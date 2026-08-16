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
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "requests"
    ])
    import requests


API_URL = "%%API_URL%%".rstrip("/")
SESSION_ID = "%%SESSION_ID%%"
WORKER_TOKEN = "%%WORKER_TOKEN%%"


# Make sure GitHub replaced the placeholders
if "%%API_URL%%" in API_URL:
    raise RuntimeError("API_URL placeholder was not replaced")

if "%%SESSION_ID%%" in SESSION_ID:
    raise RuntimeError("SESSION_ID placeholder was not replaced")

if "%%WORKER_TOKEN%%" in WORKER_TOKEN:
    raise RuntimeError("WORKER_TOKEN placeholder was not replaced")


if not API_URL.startswith(("http://", "https://")):
    raise RuntimeError(f"Invalid API_URL: {API_URL}")


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
                f"attempt={attempt} error={e}"
            )

            if attempt < retries:
                time.sleep(3)

    return None


result = {
    "session_id": SESSION_ID,
    "status": "starting",
    "gpu": None,
    "compute_capability": None,
    "cuda_available": False,
    "test": None,
    "error": None,
}


try:

    # ========================================================
    # NVIDIA GPU
    # ========================================================

    print("\n=== NVIDIA-SMI ===")

    smi = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=30
    )

    print(smi.stdout)

    if smi.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed"
        )


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
        result["gpu"] = lines[0].strip()

    print("GPU:", result["gpu"])


    # ========================================================
    # NUMPY / NUMBA
    # ========================================================

    print("\n=== CUDA ===")

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
            "numpy",
            "numba"
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
        result["compute_capability"]
    )


    # ========================================================
    # GPU TEST
    # ========================================================

    print("\n=== GPU TEST ===")

    N = 1024 * 1024


    @cuda.jit
    def add_kernel(a, b, c):

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

    add_kernel[
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


    if abs(float(out.sum()) - N * 2) > 0.01:
        raise RuntimeError(
            "GPU result verification failed"
        )


    result["test"] = {
        "elements": N,
        "time_seconds": elapsed
    }

    result["status"] = "READY"


    print(
        f"GPU TEST OK: {elapsed:.4f}s"
    )


    # ========================================================
    # WORKER READY
    # ========================================================

    print("\n=== WORKER READY ===")

    ready_payload = {
        "gpu": result["gpu"],
        "compute_capability":
            result["compute_capability"],
        "cuda_available":
            result["cuda_available"]
    }


    notified = False


    for attempt in range(1, 11):

        r = api(
            "POST",
            f"/gpu/session/{SESSION_ID}/worker-ready",
            ready_payload,
            retries=1
        )

        if r and r.status_code == 200:

            print(
                "Railway accepted worker-ready."
            )

            notified = True
            break

        print(
            f"worker-ready retry "
            f"{attempt}/10"
        )

        time.sleep(5)


    if not notified:
        raise RuntimeError(
            "Railway did not acknowledge worker-ready"
        )


    # ========================================================
    # COMMAND LOOP
    # ========================================================

    print("\n=== COMMAND LOOP ===")

    tick = 0


    while True:

        time.sleep(1)

        tick += 1


        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if tick % 30 == 0:

            hb = api(
                "POST",
                f"/gpu/session/{SESSION_ID}/heartbeat",
                retries=2
            )

            if hb and hb.status_code == 410:

                print(
                    "Session expired."
                )

                break


        # ----------------------------------------------------
        # COMMAND POLL
        # ----------------------------------------------------

        if tick % 3 != 0:
            continue


        try:

            cr = api(
                "GET",
                f"/internal/session/{SESSION_ID}/command",
                retries=2
            )

            if not cr:
                continue

            if cr.status_code != 200:
                continue


            data = cr.json()


            if data.get("expired"):

                print(
                    "Session expired."
                )

                break


            cmd = data.get("command")

            if not cmd:
                continue


            command_id = cmd["command_id"]

            operation = cmd.get(
                "operation",
                ""
            )

            parameters = cmd.get(
                "parameters",
                {}
            )


            print(
                f">>> {command_id} "
                f"operation={operation}"
            )


            started = time.time()


            # =================================================
            # EXECUTE PYTHON
            # =================================================

            if operation == "execute_python":

                code = parameters.get(
                    "code",
                    ""
                )


                namespace = {
                    "__builtins__":
                        __builtins__,
                    "np":
                        np,
                    "cuda":
                        cuda
                }


                try:

                    import torch

                    namespace["torch"] = torch

                except ImportError:

                    pass


                local_vars = {}


                try:

                    exec(
                        code,
                        namespace,
                        local_vars
                    )


                    output = {
                        k: str(v)
                        for k, v
                        in local_vars.items()
                        if not k.startswith("_")
                    }


                    cmd_out = {
                        "status": "ok",
                        "output": output,
                        "execution_time":
                            time.time() - started
                    }


                except Exception as ex:

                    cmd_out = {
                        "status": "error",
                        "error": str(ex),
                        "traceback":
                            traceback.format_exc()
                    }


            # =================================================
            # NVIDIA SMI
            # =================================================

            elif operation == "nvidia_smi":

                p = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )


                cmd_out = {
                    "status": "ok",
                    "stdout": p.stdout,
                    "stderr": p.stderr,
                    "returncode": p.returncode
                }


            # =================================================
            # SHELL
            # =================================================

            elif operation == "shell":

                command = parameters.get(
                    "command",
                    ""
                )


                try:

                    p = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )


                    cmd_out = {
                        "status": "ok",
                        "stdout": p.stdout,
                        "stderr": p.stderr,
                        "returncode": p.returncode
                    }


                except subprocess.TimeoutExpired:

                    cmd_out = {
                        "status": "error",
                        "error":
                            "Command timed out"
                    }


            # =================================================
            # INFO
            # =================================================

            elif operation == "info":

                p = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu="
                        "name,"
                        "memory.total,"
                        "memory.used,"
                        "memory.free,"
                        "temperature.gpu,"
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
                        p.stdout.strip(),
                    "hostname":
                        socket.gethostname(),
                    "python":
                        sys.version,
                    "platform":
                        platform.platform()
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

            payload = {
                "command_id": command_id,
                **cmd_out
            }


            rr = api(
                "POST",
                f"/internal/session/{SESSION_ID}/result",
                payload,
                retries=5
            )


            if rr and rr.status_code == 200:

                print(
                    f"<<< {command_id} result sent"
                )

            else:

                print(
                    f"<<< {command_id} "
                    f"result send FAILED"
                )


        except Exception as e:

            print(
                "[COMMAND LOOP ERROR]",
                e
            )


except Exception as e:

    result["status"] = "ERROR"
    result["error"] = str(e)
    result["traceback"] = traceback.format_exc()

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
