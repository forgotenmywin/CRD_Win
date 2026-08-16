import os
import sys
import json
import time
import socket
import platform
import traceback
import subprocess


# ============================================================
# REQUESTS
# ============================================================

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


# ============================================================
# INJECTED BY GITHUB ACTIONS
# ============================================================

API_URL = "%%API_URL%%".rstrip("/")
SESSION_ID = "%%SESSION_ID%%"
WORKER_TOKEN = "%%WORKER_TOKEN%%"


HEADERS = {
    "Content-Type": "application/json",
    "X-Worker-Token": WORKER_TOKEN,
}


print("=" * 70)
print("KAGGLE GPU WORKER")
print("=" * 70)
print("SESSION:", SESSION_ID)
print("API:", API_URL)
print("TOKEN:", len(WORKER_TOKEN))
print("=" * 70)


# ============================================================
# API HELPER
# ============================================================

def api(method, path, body=None, timeout=30, retries=3):

    url = f"{API_URL}{path}"

    for attempt in range(retries):

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
                f"[API] {method} {path} -> {r.status_code}"
            )

            return r

        except Exception as e:

            print(
                f"[API] {method} {path} "
                f"attempt={attempt + 1} "
                f"error={e}"
            )

            if attempt < retries - 1:
                time.sleep(3)

    return None


# ============================================================
# GPU INFORMATION
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
# GPU TEST
# ============================================================

try:

    print()
    print("=" * 70)
    print("NVIDIA-SMI")
    print("=" * 70)

    smi = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=30
    )

    print(smi.stdout)

    if smi.stderr:
        print("STDERR:")
        print(smi.stderr)

    if smi.returncode != 0:

        raise RuntimeError(
            "nvidia-smi failed"
        )


    # ========================================================
    # GPU INFO
    # ========================================================

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
    # NUMPY / NUMBA
    # ========================================================

    print()
    print("=" * 70)
    print("CUDA / NUMBA")
    print("=" * 70)


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


    # ========================================================
    # COMPUTE CAPABILITY
    # ========================================================

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
    # CUDA TEST
    # ========================================================

    N = 1024 * 1024


    @cuda.jit
    def add_kernel(a, b, c):

        i = cuda.grid(1)

        if i < a.size:

            c[i] = a[i] + b[i]


    print("Allocating GPU memory...")


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


    print("Running CUDA kernel...")


    start = time.perf_counter()


    add_kernel[
        (N + 255) // 256,
        256
    ](
        da,
        db,
        dc
    )


    cuda.synchronize()


    elapsed = (
        time.perf_counter() - start
    )


    out = dc.copy_to_host()


    expected = N * 2

    actual = float(out.sum())


    print("Expected:", expected)

    print("Actual:", actual)

    print("Time:", elapsed)


    if abs(actual - expected) > 0.01:

        raise RuntimeError(
            "GPU result verification failed"
        )


    result["test"] = {

        "elements": N,

        "time_seconds": elapsed,

    }


    result["status"] = "READY"


    print()
    print("=" * 70)
    print("GPU TEST SUCCESS")
    print("=" * 70)


    # ========================================================
    # WORKER READY
    # ========================================================

    ready_payload = {

        "gpu": result["gpu"],

        "compute_capability":
            result["compute_capability"],

        "cuda_available":
            result["cuda_available"],

    }


    print()
    print("Sending worker-ready...")


    notified = False


    for attempt in range(10):

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
            f"worker-ready retry "
            f"{attempt + 1}/10"
        )

        time.sleep(5)


    if not notified:

        raise RuntimeError(
            "Railway did not acknowledge worker-ready"
        )


    # ========================================================
    # SAVE LOCAL RESULT
    # ========================================================

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

    except Exception as e:

        print(
            "Could not save result:",
            e
        )


    # ========================================================
    # COMMAND LOOP
    # ========================================================

    print()
    print("=" * 70)
    print("COMMAND LOOP")
    print("=" * 70)


    last_heartbeat = time.time()

    last_command_poll = 0


    while True:

        now = time.time()


        # ====================================================
        # HEARTBEAT
        # ====================================================

        if now - last_heartbeat >= 30:

            hb = api(
                "POST",
                f"/gpu/session/{SESSION_ID}/heartbeat"
            )


            last_heartbeat = now


            if hb and hb.status_code == 410:

                print(
                    "Session expired."
                )

                break


        # ====================================================
        # COMMAND POLL
        # ====================================================

        if now - last_command_poll < 2:

            time.sleep(0.2)

            continue


        last_command_poll = now


        try:

            cr = api(

                "GET",

                f"/internal/session/{SESSION_ID}/command",

                retries=1

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


            command = data.get(
                "command"
            )


            if not command:

                continue


            command_id = command.get(
                "command_id"
            )

            operation = command.get(
                "operation"
            )

            params = command.get(
                "parameters",
                {}
            )


            print()
            print("=" * 70)
            print(
                "COMMAND:",
                command_id
            )
            print(
                "OPERATION:",
                operation
            )
            print("=" * 70)


            started = time.time()


            # =================================================
            # EXECUTE PYTHON
            # =================================================

            if operation == "execute_python":

                code = params.get(
                    "code",
                    ""
                )


                namespace = {

                    "__builtins__":
                        __builtins__,

                    "np":
                        np,

                    "cuda":
                        cuda,

                }


                try:

                    import torch

                    namespace["torch"] = torch

                except Exception:

                    pass


                local_vars = {}


                exec(
                    code,
                    namespace,
                    local_vars
                )


                output = {

                    k: str(v)

                    for k, v in local_vars.items()

                    if not k.startswith("_")

                }


                cmd_result = {

                    "status": "ok",

                    "output": output,

                    "execution_time":
                        time.time() - started,

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


                cmd_result = {

                    "status": "ok",

                    "stdout": p.stdout,

                    "stderr": p.stderr,

                    "returncode":
                        p.returncode,

                }


            # =================================================
            # SHELL
            # =================================================

            elif operation == "shell":

                command_line = params.get(
                    "command",
                    ""
                )


                p = subprocess.run(

                    command_line,

                    shell=True,

                    capture_output=True,

                    text=True,

                    timeout=60

                )


                cmd_result = {

                    "status": "ok",

                    "stdout": p.stdout,

                    "stderr": p.stderr,

                    "returncode":
                        p.returncode,

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


                cmd_result = {

                    "status": "ok",

                    "gpu":
                        result["gpu"],

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
                        platform.platform(),

                }


            # =================================================
            # UNKNOWN
            # =================================================

            else:

                cmd_result = {

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

                    "command_id":
                        command_id,

                    **cmd_result,

                }

            )


            print(
                "COMMAND FINISHED:",
                command_id
            )


        except Exception as e:

            print(
                "COMMAND ERROR:",
                e
            )

            traceback.print_exc()


            try:

                api(

                    "POST",

                    f"/internal/session/{SESSION_ID}/result",

                    {

                        "command_id":
                            command_id,

                        "status":
                            "error",

                        "error":
                            str(e),

                        "traceback":
                            traceback.format_exc(),

                    }

                )

            except Exception:

                pass


except Exception as e:

    result.update({

        "status": "ERROR",

        "error": str(e),

        "exception":
            type(e).__name__,

        "traceback":
            traceback.format_exc(),

    })


    print()
    print("=" * 70)
    print("WORKER FAILED")
    print("=" * 70)

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
