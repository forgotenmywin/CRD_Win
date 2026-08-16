import os
import sys
import json
import time
import socket
import platform
import traceback
import subprocess

try:
    import requests
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "requests"
    ])
    import requests


# ============================================================
# CONFIG
# These values are injected by GitHub Actions.
# ============================================================

API_URL = os.environ.get("GPU_API_URL", "")
SESSION_ID = os.environ.get("GPU_SESSION_ID", "")
WORKER_TOKEN = os.environ.get("GPU_WORKER_TOKEN", "")

API_URL = API_URL.rstrip("/")


# ============================================================
# VALIDATION
# ============================================================

if not API_URL:
    raise RuntimeError("GPU_API_URL is missing")

if not SESSION_ID:
    raise RuntimeError("GPU_SESSION_ID is missing")

if not WORKER_TOKEN:
    raise RuntimeError("GPU_WORKER_TOKEN is missing")


HEADERS = {
    "Content-Type": "application/json",
    "X-Worker-Token": WORKER_TOKEN,
}


print("=" * 60)
print("KAGGLE GPU WORKER")
print("=" * 60)
print("SESSION :", SESSION_ID)
print("API     :", API_URL)
print("TOKEN   :", len(WORKER_TOKEN))
print("=" * 60)


# ============================================================
# API HELPER
# ============================================================

def api(method, path, body=None, timeout=30, retries=3):

    url = f"{API_URL}{path}"

    for attempt in range(1, retries + 1):

        try:

            if method == "GET":
                r = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=timeout
                )
            else:
                r = requests.post(
                    url,
                    headers=HEADERS,
                    json=body or {},
                    timeout=timeout
                )

            print(
                f"[API] {method} {path} "
                f"→ {r.status_code}"
            )

            return r

        except Exception as e:

            print(
                f"[API] {method} {path} "
                f"attempt={attempt} "
                f"error={e}"
            )

            if attempt < retries:
                time.sleep(3)

    return None


# ============================================================
# RESULT OBJECT
# ============================================================

result = {
    "session_id": SESSION_ID,
    "status": "starting",
    "gpu": None,
    "compute_capability": None,
    "cuda_available": False,
    "test": None,
    "error": None,
}


# ============================================================
# GPU INITIALIZATION
# ============================================================

try:

    print()
    print("=" * 60)
    print("NVIDIA SMI")
    print("=" * 60)

    smi = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=30
    )

    print(smi.stdout)

    if smi.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed - GPU unavailable"
        )


    # --------------------------------------------------------
    # GPU INFORMATION
    # --------------------------------------------------------

    gpu_csv = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader"
        ],
        capture_output=True,
        text=True,
        timeout=30
    )

    lines = gpu_csv.stdout.strip().splitlines()

    if lines:
        result["gpu"] = lines[0]

    print("GPU:", result["gpu"])


    # ========================================================
    # NUMBA / CUDA
    # ========================================================

    print()
    print("=" * 60)
    print("CUDA TEST")
    print("=" * 60)

    try:

        import numpy as np
        from numba import cuda

    except ImportError:

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "numba",
            "numpy"
        ])

        import numpy as np
        from numba import cuda


    result["cuda_available"] = bool(
        cuda.is_available()
    )

    print(
        "CUDA available:",
        result["cuda_available"]
    )

    if not result["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available"
        )


    device = cuda.get_current_device()

    cap = device.compute_capability

    result["compute_capability"] = [
        int(cap[0]),
        int(cap[1])
    ]

    print(
        "Compute capability:",
        cap
    )


    # ========================================================
    # GPU KERNEL SMOKE TEST
    # ========================================================

    N = 1024 * 1024


    @cuda.jit
    def gpu_add(a, b, c):

        i = cuda.grid(1)

        if i < a.size:
            c[i] = a[i] + b[i]


    a = np.ones(
        N,
        dtype=np.float32
    )

    b = np.ones(
        N,
        dtype=np.float32
    )

    c = np.zeros(
        N,
        dtype=np.float32
    )


    da = cuda.to_device(a)
    db = cuda.to_device(b)
    dc = cuda.to_device(c)


    t0 = time.perf_counter()

    gpu_add[
        (N + 255) // 256,
        256
    ](
        da,
        db,
        dc
    )

    cuda.synchronize()

    elapsed = time.perf_counter() - t0


    out = dc.copy_to_host()


    if abs(
        float(out.sum()) - N * 2
    ) > 0.01:

        raise RuntimeError(
            "GPU result verification failed"
        )


    result["test"] = {
        "elements": N,
        "time_seconds": elapsed
    }

    result["status"] = "READY"


    print(
        f"GPU kernel OK: "
        f"{N} elements "
        f"in {elapsed:.4f}s"
    )


    # ========================================================
    # WORKER READY
    # ========================================================

    print()
    print("=" * 60)
    print("NOTIFYING RAILWAY")
    print("=" * 60)


    ready_payload = {
        "gpu": result["gpu"],
        "compute_capability": result["compute_capability"],
        "cuda_available": result["cuda_available"],
    }


    notified = False


    for attempt in range(1, 11):

        r = api(
            "POST",
            f"/gpu/session/{SESSION_ID}/worker-ready",
            ready_payload,
            retries=1
        )

        if r and r.status_code in (200, 202):

            print(
                "WORKER READY accepted by Railway."
            )

            notified = True
            break

        print(
            f"worker-ready retry {attempt}/10"
        )

        time.sleep(5)


    if not notified:

        raise RuntimeError(
            "Railway did not accept worker-ready"
        )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    with open(
        "/kaggle/working/session_result.json",
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


    # ========================================================
    # COMMAND LOOP
    # ========================================================

    print()
    print("=" * 60)
    print("COMMAND LOOP")
    print("=" * 60)


    last_heartbeat = 0
    started = time.time()


    while True:

        # ----------------------------------------------------
        # Session maximum lifetime
        # Worker can stay alive while Railway heartbeats
        # ----------------------------------------------------

        if time.time() - started > 1100:
            print(
                "Maximum worker runtime reached."
            )
            break


        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if time.time() - last_heartbeat >= 30:

            hb = api(
                "POST",
                f"/gpu/session/{SESSION_ID}/heartbeat",
                retries=2
            )

            last_heartbeat = time.time()

            if hb and hb.status_code == 410:

                print(
                    "Railway says session expired."
                )

                break


        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        try:

            cr = api(
                "GET",
                f"/internal/session/{SESSION_ID}/command",
                retries=2
            )

            if not cr:
                time.sleep(2)
                continue


            if cr.status_code != 200:

                time.sleep(2)
                continue


            data = cr.json()


            if data.get("expired"):

                print(
                    "Session expired."
                )

                break


            cmd = data.get("command")


            if not cmd:

                time.sleep(2)
                continue


            command_id = cmd.get(
                "command_id"
            )

            operation = cmd.get(
                "operation"
            )

            parameters = cmd.get(
                "parameters",
                {}
            )


            print()
            print("=" * 60)
            print("COMMAND RECEIVED")
            print("ID:", command_id)
            print("OP:", operation)
            print("=" * 60)


            start_time = time.time()


            # =================================================
            # EXECUTE PYTHON
            # =================================================

            if operation == "execute_python":

                code = parameters.get(
                    "code",
                    ""
                )


                namespace = {
                    "__builtins__": __builtins__,
                    "np": np,
                    "cuda": cuda,
                }


                try:

                    import torch

                    namespace["torch"] = torch

                except Exception:

                    pass


                local_vars = {}


                try:

                    exec(
                        code,
                        namespace,
                        local_vars
                    )


                    output = {}

                    for key, value in local_vars.items():

                        if not key.startswith("_"):

                            try:
                                output[key] = str(value)
                            except Exception:
                                output[key] = repr(value)


                    cmd_out = {
                        "status": "ok",
                        "output": output,
                        "execution_time":
                            time.time() - start_time
                    }


                except Exception as e:

                    cmd_out = {
                        "status": "error",
                        "error": str(e),
                        "traceback":
                            traceback.format_exc()
                    }


            # =================================================
            # NVIDIA SMI
            # =================================================

            elif operation == "nvidia_smi":

                s2 = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )


                cmd_out = {
                    "status": "ok",
                    "stdout": s2.stdout,
                    "stderr": s2.stderr,
                    "returncode": s2.returncode,
                }


            # =================================================
            # SHELL
            # =================================================

            elif operation == "shell":

                command = parameters.get(
                    "command",
                    ""
                )


                if not command:

                    cmd_out = {
                        "status": "error",
                        "error":
                            "command missing"
                    }

                else:

                    try:

                        sh = subprocess.run(
                            command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=120
                        )


                        cmd_out = {
                            "status": "ok",
                            "stdout": sh.stdout,
                            "stderr": sh.stderr,
                            "returncode":
                                sh.returncode
                        }


                    except Exception as e:

                        cmd_out = {
                            "status": "error",
                            "error": str(e),
                            "traceback":
                                traceback.format_exc()
                        }


            # =================================================
            # INFO
            # =================================================

            elif operation == "info":

                mem = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu="
                        "name,memory.total,memory.used,"
                        "memory.free,temperature.gpu,"
                        "utilization.gpu",
                        "--format=csv,noheader"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )


                cmd_out = {
                    "status": "ok",
                    "gpu": result["gpu"],
                    "compute_capability":
                        result["compute_capability"],
                    "cuda_available":
                        result["cuda_available"],
                    "gpu_details":
                        mem.stdout.strip(),
                    "hostname":
                        socket.gethostname(),
                    "python":
                        sys.version,
                    "platform":
                        platform.platform(),
                }


            else:

                cmd_out = {
                    "status": "error",
                    "error":
                        f"Unknown operation: {operation}"
                }


            # =================================================
            # SEND RESULT
            # =================================================

            api(
                "POST",
                f"/internal/session/{SESSION_ID}/result",
                {
                    "command_id": command_id,
                    **cmd_out
                },
                retries=3
            )


            print(
                "COMMAND FINISHED:",
                command_id
            )


        except Exception as e:

            print(
                "[COMMAND LOOP ERROR]",
                e
            )

            traceback.print_exc()


        time.sleep(1)


    print(
        "Worker finished."
    )


except Exception as e:

    result.update({
        "status": "ERROR",
        "error": str(e),
        "exception": type(e).__name__,
        "traceback": traceback.format_exc(),
    })


    print()
    print("=" * 60)
    print("WORKER ERROR")
    print("=" * 60)

    traceback.print_exc()


    try:

        with open(
            "/kaggle/working/session_result.json",
            "w"
        ) as f:

            json.dump(
                result,
                f,
                indent=2
            )

    except Exception:

        pass


    raise
