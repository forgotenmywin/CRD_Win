import os
import sys
import json
import time
import socket
import platform
import traceback
import subprocess
import io
import contextlib

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

# Session is 20 minutes by default.
# Worker stays slightly shorter to exit cleanly.
WORKER_MAX_SECONDS = 1100

HEARTBEAT_INTERVAL = 30
COMMAND_POLL_INTERVAL = 5

HEADERS = {
    "Content-Type": "application/json",
    "X-Worker-Token": WORKER_TOKEN,
}

print("=" * 70)
print("KAGGLE GPU WORKER")
print("SESSION   :", SESSION_ID)
print("API       :", API_URL)
print("TOKEN LEN :", len(WORKER_TOKEN))
print("=" * 70)


# ============================================================
# API HELPER
# ============================================================

def api(
    method,
    path,
    body=None,
    timeout=30,
    retries=3
):
    url = f"{API_URL}{path}"

    for attempt in range(retries):

        try:

            if method == "GET":
                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=timeout
                )

            else:
                response = requests.post(
                    url,
                    headers=HEADERS,
                    json=body or {},
                    timeout=timeout
                )

            print(
                f"[API] {method} {path} "
                f"-> {response.status_code}"
            )

            return response

        except Exception as e:

            print(
                f"[API] {method} {path} "
                f"attempt {attempt + 1} "
                f"error: {e}"
            )

            if attempt < retries - 1:
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
# NOTIFY ERROR
# ============================================================

def notify_error(error_text):

    try:

        api(
            "POST",
            f"/gpu/session/{SESSION_ID}/worker-error",
            {
                "error": error_text
            },
            retries=2
        )

    except Exception:
        pass


# ============================================================
# SAFE STRING LIMIT
# ============================================================

def limit_text(value, max_chars=50000):

    value = str(value)

    if len(value) <= max_chars:
        return value

    return (
        value[:max_chars]
        + "\n...[TRUNCATED]..."
    )


# ============================================================
# MAIN WORKER
# ============================================================

try:

    # ========================================================
    # NVIDIA SMI
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
            "nvidia-smi failed -- no usable GPU"
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

    lines = (
        gpu_csv.stdout
        .strip()
        .splitlines()
    )

    if lines:
        result["gpu"] = lines[0]

    print("GPU:", result["gpu"])


    # ========================================================
    # NUMPY + NUMBA
    # ========================================================

    print("\n=== CUDA VIA NUMBA ===")

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

    if not result["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available"
        )


    # ========================================================
    # COMPUTE CAPABILITY
    # ========================================================

    cap = (
        cuda
        .get_current_device()
        .compute_capability
    )

    result["compute_capability"] = [
        int(cap[0]),
        int(cap[1])
    ]

    print(
        "Compute capability:",
        result["compute_capability"]
    )


    # ========================================================
    # CUDA SMOKE TEST
    # ========================================================

    print("\n=== CUDA SMOKE TEST ===")

    N = 1024 * 1024


    @cuda.jit
    def _add(a, b, c):

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

    _add[
        (N + 255) // 256,
        256
    ](
        da,
        db,
        dc
    )

    cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - t0
    )


    out = dc.copy_to_host()

    expected = N * 2

    if abs(
        float(out.sum()) - expected
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
        f"Kernel OK - "
        f"{N} elements - "
        f"{elapsed:.4f}s"
    )


    # ========================================================
    # WORKER READY
    # ========================================================

    print("\n=== WORKER READY ===")

    ready_payload = {
        "gpu": result["gpu"],
        "compute_capability": (
            result["compute_capability"]
        ),
        "cuda_available": (
            result["cuda_available"]
        ),
    }

    notified = False

    for attempt in range(5):

        response = api(
            "POST",
            f"/gpu/session/{SESSION_ID}/worker-ready",
            ready_payload,
            retries=1
        )

        if response and response.status_code in (
            200,
            202
        ):

            print(
                "API acknowledged worker-ready."
            )

            notified = True
            break

        print(
            f"worker-ready retry "
            f"{attempt + 1}/5"
        )

        time.sleep(5)


    if not notified:

        raise RuntimeError(
            "Railway never acknowledged worker-ready"
        )


    # ========================================================
    # SAVE INITIAL RESULT
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

    except Exception:
        pass


    # ========================================================
    # COMMAND LOOP
    # ========================================================

    print("\n=== COMMAND LOOP ===")
    print(
        f"Maximum runtime: "
        f"{WORKER_MAX_SECONDS}s"
    )

    worker_started = time.time()
    last_heartbeat = time.time()
    last_poll = 0

    while (
        time.time() - worker_started
        < WORKER_MAX_SECONDS
    ):

        now = time.time()


        # ====================================================
        # HEARTBEAT
        # ====================================================

        if (
            now - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            hb = api(
                "POST",
                f"/gpu/session/{SESSION_ID}/heartbeat",
                retries=2
            )

            last_heartbeat = now

            if hb and hb.status_code == 410:

                print(
                    "Session expired."
                )

                break


            if hb and hb.status_code == 404:

                print(
                    "Session disappeared."
                )

                break


        # ====================================================
        # POLL COMMAND
        # ====================================================

        if (
            now - last_poll
            < COMMAND_POLL_INTERVAL
        ):

            time.sleep(0.5)
            continue

        last_poll = now


        try:

            response = api(
                "GET",
                f"/internal/session/"
                f"{SESSION_ID}/command",
                retries=1
            )

            if not response:
                continue

            if response.status_code != 200:
                continue

            data = response.json()


            if data.get("expired"):

                print(
                    "Session expired."
                )

                break


            command = data.get("command")

            if not command:
                continue


            operation = command.get(
                "operation",
                ""
            )

            parameters = command.get(
                "parameters",
                {}
            )

            command_id = command[
                "command_id"
            ]


            print(
                f"\n>>> {command_id} "
                f"operation={operation}"
            )


            start_time = time.time()


            # =================================================
            # EXECUTE PYTHON
            # =================================================

            if operation == "execute_python":

                code = parameters.get(
                    "code",
                    ""
                )

                if not code:
                    raise ValueError(
                        "code is empty"
                    )


                # -----------------------------
                # stdout / stderr capture
                # -----------------------------

                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()


                # -----------------------------
                # globals
                # -----------------------------

                globals_dict = {
                    "__builtins__": __builtins__,
                    "np": np,
                    "cuda": cuda,
                }


                # Try PyTorch.
                try:

                    import torch

                    globals_dict["torch"] = torch

                except Exception:

                    pass


                locals_dict = {}


                # -----------------------------
                # execute
                # -----------------------------

                with contextlib.redirect_stdout(
                    stdout_buffer
                ), contextlib.redirect_stderr(
                    stderr_buffer
                ):

                    exec(
                        code,
                        globals_dict,
                        locals_dict
                    )


                cmd_out = {
                    "status": "ok",

                    "stdout": limit_text(
                        stdout_buffer.getvalue()
                    ),

                    "stderr": limit_text(
                        stderr_buffer.getvalue()
                    ),

                    "variables": {
                        key: limit_text(value, 10000)
                        for key, value
                        in locals_dict.items()
                        if not key.startswith("_")
                    },

                    "execution_time": (
                        time.time()
                        - start_time
                    ),
                }


            # =================================================
            # NVIDIA SMI
            # =================================================

            elif operation == "nvidia_smi":

                smi_result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                cmd_out = {
                    "status": "ok",
                    "stdout": limit_text(
                        smi_result.stdout
                    ),
                    "stderr": limit_text(
                        smi_result.stderr
                    ),
                    "returncode": (
                        smi_result.returncode
                    ),
                    "execution_time": (
                        time.time()
                        - start_time
                    ),
                }


            # =================================================
            # SHELL
            # =================================================

            elif operation == "shell":

                shell_command = parameters.get(
                    "command",
                    ""
                )

                if not shell_command:
                    raise ValueError(
                        "command is empty"
                    )


                shell_result = subprocess.run(
                    shell_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120
                )


                cmd_out = {
                    "status": "ok",
                    "stdout": limit_text(
                        shell_result.stdout
                    ),
                    "stderr": limit_text(
                        shell_result.stderr
                    ),
                    "returncode": (
                        shell_result.returncode
                    ),
                    "execution_time": (
                        time.time()
                        - start_time
                    ),
                }


            # =================================================
            # INFO
            # =================================================

            elif operation == "info":

                mem = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu="
                        "name,memory.total,"
                        "memory.used,memory.free,"
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

                    "compute_capability": (
                        result[
                            "compute_capability"
                        ]
                    ),

                    "cuda_available": (
                        result[
                            "cuda_available"
                        ]
                    ),

                    "gpu_details": (
                        mem.stdout.strip()
                    ),

                    "hostname": (
                        socket.gethostname()
                    ),

                    "python": sys.version,

                    "platform": (
                        platform.platform()
                    ),
                }


            # =================================================
            # UNKNOWN
            # =================================================

            else:

                cmd_out = {
                    "status": "error",
                    "error": (
                        f"Unknown operation: "
                        f"{operation}"
                    )
                }


        except Exception as exc:

            cmd_out = {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "execution_time": (
                    time.time()
                    - start_time
                ),
            }


        # ====================================================
        # SEND RESULT
        # ====================================================

        result_payload = {
            "command_id": command_id,
            **cmd_out
        }

        post_result = api(
            "POST",
            f"/internal/session/"
            f"{SESSION_ID}/result",
            result_payload,
            retries=3
        )


        if post_result and post_result.status_code in (
            200,
            201,
            202
        ):

            print(
                f"<<< {command_id} result sent"
            )

        else:

            print(
                f"<<< {command_id} "
                f"result send failed"
            )


    print(
        "\nWorker loop finished."
    )


except Exception as exc:

    result.update({
        "status": "ERROR",
        "error": str(exc),
        "exception": type(exc).__name__,
        "traceback": traceback.format_exc(),
    })


    traceback.print_exc()


    notify_error(
        f"{type(exc).__name__}: {exc}"
    )


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
