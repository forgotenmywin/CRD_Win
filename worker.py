import os, sys, json, time, socket, platform, traceback, subprocess

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests

# ── injected by CI (sed replaces placeholders) ──────────
API_URL      = "%%API_URL%%".rstrip("/")
SESSION_ID   = "%%SESSION_ID%%"
WORKER_TOKEN = "%%WORKER_TOKEN%%"
# ─────────────────────────────────────────────────────────

HEADERS = {"Content-Type": "application/json", "X-Worker-Token": WORKER_TOKEN}

print("=" * 60)
print("KAGGLE GPU WORKER")
print("SESSION  :", SESSION_ID)
print("API      :", API_URL)
print("TOKEN len:", len(WORKER_TOKEN))
print("=" * 60)


def api(method, path, body=None, timeout=30, retries=3):
    url = f"{API_URL}{path}"
    for attempt in range(retries):
        try:
            if method == "GET":
                r = requests.get(url, headers=HEADERS, timeout=timeout)
            else:
                r = requests.post(url, headers=HEADERS, json=body or {}, timeout=timeout)
            print(f"[API] {method} {path} → {r.status_code}")
            return r
        except Exception as e:
            print(f"[API] {method} {path} attempt {attempt+1} error: {e}")
            if attempt < retries - 1:
                time.sleep(3)
    return None


result = {
    "session_id":         SESSION_ID,
    "status":             "starting",
    "gpu":                None,
    "compute_capability": None,
    "cuda_available":     False,
    "test":               None,
    "error":              None,
}

try:
    # ── nvidia-smi ──────────────────────────────────────────
    print("\n=== nvidia-smi ===")
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
    print(smi.stdout)
    if smi.returncode != 0:
        raise RuntimeError("nvidia-smi failed — no GPU")

    gpu_csv = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    lines = gpu_csv.stdout.strip().splitlines()
    if lines:
        result["gpu"] = lines[0]
    print("GPU:", result["gpu"])

    # ── install / import numba ───────────────────────────────
    print("\n=== CUDA via numba ===")
    try:
        import numpy as np
        from numba import cuda
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "numba", "numpy"])
        import numpy as np
        from numba import cuda

    result["cuda_available"] = bool(cuda.is_available())
    if not result["cuda_available"]:
        raise RuntimeError("CUDA not available")

    cap = cuda.get_current_device().compute_capability
    result["compute_capability"] = [int(cap[0]), int(cap[1])]
    print("Compute capability:", cap)

    # ── kernel smoke-test ────────────────────────────────────
    N = 1024 * 1024

    @cuda.jit
    def _add(a, b, c):
        i = cuda.grid(1)
        if i < a.size:
            c[i] = a[i] + b[i]

    a = np.ones(N, np.float32)
    b = np.ones(N, np.float32)
    c = np.zeros(N, np.float32)
    da, db, dc = cuda.to_device(a), cuda.to_device(b), cuda.to_device(c)

    t0 = time.perf_counter()
    _add[(N + 255) // 256, 256](da, db, dc)
    cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out = dc.copy_to_host()
    if abs(float(out.sum()) - N * 2) > 0.01:
        raise RuntimeError("GPU result verification failed")

    result["test"]   = {"elements": N, "time_seconds": elapsed}
    result["status"] = "READY"
    print(f"Kernel OK — {N} elems in {elapsed:.4f}s")

    # ── notify API: worker-ready (with retry) ────────────────
    print("\n=== Notifying API: worker-ready ===")
    ready_payload = {
        "gpu":                result["gpu"],
        "compute_capability": result["compute_capability"],
        "cuda_available":     result["cuda_available"],
    }

    notified = False
    for attempt in range(5):
        r = api("POST", f"/gpu/session/{SESSION_ID}/worker-ready",
                ready_payload, retries=1)
        if r and r.status_code in (200, 202):
            print("API acknowledged worker-ready.")
            notified = True
            break
        print(f"Retry worker-ready {attempt+1}/5 ...")
        time.sleep(5)

    if not notified:
        print("WARNING: API never acknowledged worker-ready. Continuing anyway.")

    # ── save result ──────────────────────────────────────────
    with open("/kaggle/working/session_result.json", "w") as f:
        json.dump(result, f, indent=2)

    # ── command poll loop ────────────────────────────────────
    print("\n=== COMMAND LOOP (max 10 min) ===")
    for tick in range(600):
        time.sleep(1)

        # heartbeat every 30s
        if tick > 0 and tick % 30 == 0:
            hb = api("POST", f"/gpu/session/{SESSION_ID}/heartbeat")
            if hb and hb.status_code == 410:
                print("Session expired. Exiting.")
                break

        # poll for command every 5s
        if tick % 5 != 0:
            continue

        try:
            cr = api("GET", f"/internal/session/{SESSION_ID}/command", retries=1)
            if not (cr and cr.status_code == 200):
                continue

            data = cr.json()
            if data.get("expired"):
                print("Session expired. Exiting.")
                break
            cmd = data.get("command")
            if not cmd:
                continue

            op, params, cid = cmd.get("operation",""), cmd.get("parameters",{}), cmd["command_id"]
            print(f"\n>>> {cid}  op={op}")
            t_start = time.time()

            try:
                if op == "execute_python":
                    g = {"__builtins__": __builtins__, "np": np, "cuda": cuda}
                    try:
                        import torch; g["torch"] = torch
                    except ImportError:
                        pass
                    loc = {}
                    exec(params.get("code",""), g, loc)
                    cmd_out = {
                        "status":         "ok",
                        "output":         {k: str(v) for k, v in loc.items()
                                           if not k.startswith("_")},
                        "execution_time": time.time() - t_start,
                    }

                elif op == "nvidia_smi":
                    s2 = subprocess.run(
                        ["nvidia-smi"], capture_output=True, text=True, timeout=30)
                    cmd_out = {"status":"ok","stdout":s2.stdout,
                               "stderr":s2.stderr,"returncode":s2.returncode}

                elif op == "shell":
                    sh = subprocess.run(
                        params.get("command",""), shell=True,
                        capture_output=True, text=True, timeout=60)
                    cmd_out = {"status":"ok","stdout":sh.stdout,
                               "stderr":sh.stderr,"returncode":sh.returncode}

                elif op == "info":
                    mem = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=name,memory.total,memory.used,"
                         "memory.free,temperature.gpu,utilization.gpu",
                         "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=30)
                    cmd_out = {
                        "status":             "ok",
                        "gpu":                result["gpu"],
                        "compute_capability": result["compute_capability"],
                        "cuda_available":     result["cuda_available"],
                        "gpu_details":        mem.stdout.strip(),
                        "hostname":           socket.gethostname(),
                        "python":             sys.version,
                        "platform":           platform.platform(),
                    }

                else:
                    cmd_out = {"status":"error","error":f"Unknown operation: {op}"}

            except Exception as ex:
                cmd_out = {"status":"error","error":str(ex),
                           "traceback":traceback.format_exc()}

            api("POST", f"/internal/session/{SESSION_ID}/result",
                {"command_id": cid, **cmd_out})
            print(f"<<< {cid} done.")

        except Exception as poll_err:
            print(f"[POLL] {poll_err}")

    print("Worker loop finished.")

except Exception as e:
    result.update({
        "status":    "ERROR",
        "error":     str(e),
        "exception": type(e).__name__,
        "traceback": traceback.format_exc(),
    })
    traceback.print_exc()
    try:
        with open("/kaggle/working/session_result.json", "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass
    raise
