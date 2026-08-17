import sys
import time
import json
import socket
import platform
import traceback
import subprocess
import io
import contextlib

import requests


# ============================================================
# RUNTIME CONFIG
# These 3 variables are prepended by GitHub Actions.
# DO NOT define them again in this file.
# ============================================================

# SESSION_ID
# API_URL
# WORKER_TOKEN


# ============================================================
# TEST CONFIG
# ============================================================

WORKER_RUNTIME_SECONDS = 120

HEARTBEAT_INTERVAL = 10
COMMAND_INTERVAL = 3

REQUEST_TIMEOUT = 15
COMMAND_TIMEOUT = 120

MAX_OUTPUT = 50000


# ============================================================
# LOG
# ============================================================

def log(message=""):
    print(message, flush=True)


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config():

    required = [
        "SESSION_ID",
        "API_URL",
        "WORKER_TOKEN",
    ]

    missing = []

    for name in required:

        if name not in globals():
            missing.append(name)

    if missing:

        raise RuntimeError(
            "Runtime config missing: "
            + ", ".join(missing)
        )

    if not SESSION_ID:

        raise RuntimeError(
            "SESSION_ID is empty"
        )

    if not API_URL:

        raise RuntimeError(
            "API_URL is empty"
        )

    if not WORKER_TOKEN:

        raise RuntimeError(
            "WORKER_TOKEN is empty"
        )

    if not API_URL.startswith(
        ("http://", "https://")
    ):

        raise RuntimeError(
            f"Invalid API_URL: {API_URL}"
        )


# ============================================================
# HTTP HEADERS
# ============================================================

def headers():

    return {
        "Content-Type": "application/json",
        "X-Worker-Token": WORKER_TOKEN,
    }


# ============================================================
# HTTP GET
# ============================================================

def api_get(path):

    url = (
        API_URL.rstrip("/")
        + path
    )

    try:

        response = requests.get(
            url,
            headers=headers(),
            timeout=REQUEST_TIMEOUT,
        )

        log(
            f"[API] GET {path} "
            f"-> {response.status_code}"
        )

        return response

    except Exception as exc:

        log(
            f"[API] GET {path} ERROR: "
            f"{exc}"
        )

        return None


# ============================================================
# HTTP POST
# ============================================================

def api_post(path, payload=None):

    url = (
        API_URL.rstrip("/")
        + path
    )

    try:

        response = requests.post(
            url,
            headers=headers(),
            json=payload or {},
            timeout=REQUEST_TIMEOUT,
        )

        log(
            f"[API] POST {path} "
            f"-> {response.status_code}"
        )

        if response.status_code >= 400:

            log(
                response.text[:2000]
            )

        return response

    except Exception as exc:

        log(
            f"[API] POST {path} ERROR: "
            f"{exc}"
        )

        return None


# ============================================================
# NVIDIA SMI
# ============================================================

def show_nvidia_smi():

    log()
    log("=" * 60)
    log("NVIDIA-SMI")
    log("=" * 60)

    process = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    log(process.stdout)

    if process.stderr:

        log(process.stderr)

    if process.returncode != 0:

        raise RuntimeError(
            "nvidia-smi failed"
        )


def get_gpu_info():

    process = subprocess.run(
        [
            "nvidia-smi",

            "--query-gpu="
            "name,memory.total,driver_version",

            "--format=csv,noheader",
        ],

        capture_output=True,

        text=True,

        timeout=30,
    )

    if process.returncode != 0:

        return None

    return process.stdout.strip()


# ============================================================
# CUDA TEST
# Uses Numba, NOT PyTorch.
# This is required because current Kaggle PyTorch does not
# support Tesla P100 / compute capability 6.0.
# ============================================================

def cuda_test():

    log()
    log("=" * 60)
    log("CUDA / NUMBA TEST")
    log("=" * 60)

    try:

        import numpy as np
        from numba import cuda

    except ImportError:

        log(
            "Installing numpy and numba..."
        )

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "numpy",
                "numba",
            ]
        )

        import numpy as np
        from numba import cuda


    available = cuda.is_available()

    log(
        f"CUDA available: {available}"
    )

    if not available:

        return {
            "ok": False,
            "error": "CUDA unavailable",
            "compute_capability": None,
        }


    device = cuda.get_current_device()

    capability = (
        device.compute_capability
    )

    log(
        f"GPU: {device.name}"
    )

    log(
        f"Compute capability: {capability}"
    )


    # --------------------------------------------------------
    # Actual CUDA kernel
    # --------------------------------------------------------

    n = 1024 * 1024


    a = np.ones(
        n,
        dtype=np.float32,
    )

    b = np.ones(
        n,
        dtype=np.float32,
    )

    c = np.zeros(
        n,
        dtype=np.float32,
    )


    @cuda.jit
    def add_kernel(a, b, c):

        i = cuda.grid(1)

        if i < c.size:

            c[i] = (
                a[i]
                + b[i]
            )


    threads = 256

    blocks = (
        n + threads - 1
    ) // threads


    device_a = cuda.to_device(a)

    device_b = cuda.to_device(b)

    device_c = cuda.to_device(c)


    start = time.perf_counter()


    add_kernel[
        blocks,
        threads
    ](
        device_a,
        device_b,
        device_c,
    )


    cuda.synchronize()


    elapsed = (
        time.perf_counter()
        - start
    )


    result = (
        device_c.copy_to_host()
    )


    if not np.allclose(
        result,
        2.0
    ):

        return {
            "ok": False,
            "error":
                "CUDA result verification failed",
            "compute_capability": [
                int(capability[0]),
                int(capability[1]),
            ],
        }


    log(
        f"GPU kernel OK: "
        f"{n} elements "
        f"in {elapsed:.4f}s"
    )


    return {

        "ok": True,

        "compute_capability": [
            int(capability[0]),
            int(capability[1]),
        ],

        "elements": n,

        "time_seconds": elapsed,
    }


# ============================================================
# WORKER READY
# ============================================================

def worker_ready(
    gpu_info,
    cuda_result,
):

    payload = {

        "gpu": gpu_info,

        "compute_capability":
            cuda_result.get(
                "compute_capability"
            ),

        "cuda_available":
            bool(
                cuda_result.get(
                    "ok"
                )
            ),
    }


    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )


    log()
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)


    for attempt in range(1, 11):

        log(
            f"worker-ready attempt "
            f"{attempt}/10"
        )


        response = api_post(
            path,
            payload,
        )


        if response is not None:

            if response.status_code in (
                200,
                202,
            ):

                log(
                    "WORKER READY accepted "
                    "by Railway."
                )

                return True


        time.sleep(3)


    return False


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():

    response = api_post(
        f"/gpu/session/"
        f"{SESSION_ID}/heartbeat",
        {
            "timestamp": time.time(),
        },
    )


    if response is None:

        return False


    if response.status_code == 410:

        log(
            "Railway says session expired."
        )

        return False


    return response.status_code in (
        200,
        202,
    )


# ============================================================
# COMMAND FETCH
# ============================================================

def get_command():

    response = api_get(
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )


    if response is None:

        return None


    if response.status_code != 200:

        return None


    try:

        data = response.json()

    except Exception:

        return None


    if not isinstance(
        data,
        dict
    ):

        return None


    if data.get(
        "expired"
    ):

        return {
            "_expired": True
        }


    command = data.get(
        "command"
    )


    if not command:

        return None


    return command


# ============================================================
# RESULT
# ============================================================

def send_result(
    command_id,
    result,
):

    response = api_post(
        f"/internal/session/"
        f"{SESSION_ID}/result",
        {
            "command_id":
                command_id,

            **result,
        },
    )


    if response is None:

        return False


    return response.status_code in (
        200,
        201,
        202,
    )


# ============================================================
# LIMIT OUTPUT
# ============================================================

def limit_output(
    value,
    maximum=MAX_OUTPUT,
):

    value = str(value)

    if len(value) <= maximum:

        return value

    return (
        value[:maximum]
        + "\n...[TRUNCATED]..."
    )


# ============================================================
# PYTHON COMMAND
# ============================================================

def execute_python(
    parameters
):

    code = parameters.get(
        "code",
        "",
    )


    if not code:

        return {
            "status": "error",
            "error":
                "Python code is empty",
        }


    namespace = {
        "__builtins__":
            __builtins__,
    }


    try:

        import numpy as np

        namespace["np"] = np

    except Exception:

        pass


    try:

        from numba import cuda

        namespace["cuda"] = cuda

    except Exception:

        pass


    try:

        import torch

        namespace["torch"] = torch

    except Exception:

        pass


    local_vars = {}

    stdout = io.StringIO()

    stderr = io.StringIO()


    started = time.perf_counter()


    try:

        with contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(
            stderr
        ):

            exec(
                code,
                namespace,
                local_vars,
            )


        variables = {}


        for key, value in local_vars.items():

            if key.startswith(
                "_"
            ):

                continue


            try:

                variables[key] = (
                    limit_output(
                        repr(value),
                        10000,
                    )
                )

            except Exception:

                variables[key] = (
                    "<unprintable>"
                )


        return {

            "status": "ok",

            "stdout":
                limit_output(
                    stdout.getvalue()
                ),

            "stderr":
                limit_output(
                    stderr.getvalue()
                ),

            "variables":
                variables,

            "execution_time":
                time.perf_counter()
                - started,
        }


    except Exception as exc:

        return {

            "status": "error",

            "stdout":
                limit_output(
                    stdout.getvalue()
                ),

            "stderr":
                limit_output(
                    stderr.getvalue()
                ),

            "error":
                str(exc),

            "traceback":
                traceback.format_exc(),

            "execution_time":
                time.perf_counter()
                - started,
        }


# ============================================================
# NVIDIA-SMI COMMAND
# ============================================================

def execute_nvidia_smi():

    try:

        process = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=60,
        )


        return {

            "status": "ok",

            "stdout":
                limit_output(
                    process.stdout
                ),

            "stderr":
                limit_output(
                    process.stderr
                ),

            "returncode":
                process.returncode,
        }


    except Exception as exc:

        return {

            "status": "error",

            "error":
                str(exc),
        }


# ============================================================
# SHELL COMMAND
# ============================================================

def execute_shell(
    parameters
):

    command = parameters.get(
        "command",
        "",
    )


    if not command:

        return {

            "status": "error",

            "error":
                "Shell command is empty",
        }


    try:

        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )


        return {

            "status": "ok",

            "stdout":
                limit_output(
                    process.stdout
                ),

            "stderr":
                limit_output(
                    process.stderr
                ),

            "returncode":
                process.returncode,
        }


    except subprocess.TimeoutExpired:

        return {

            "status": "error",

            "error":
                "Shell command timed out",
        }


    except Exception as exc:

        return {

            "status": "error",

            "error":
                str(exc),
        }


# ============================================================
# INFO
# ============================================================

def execute_info():

    try:

        process = subprocess.run(
            [
                "nvidia-smi",

                "--query-gpu="
                "name,"
                "memory.total,"
                "memory.used,"
                "memory.free,"
                "temperature.gpu,"
                "utilization.gpu",

                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )


        capability = None


        try:

            from numba import cuda

            if cuda.is_available():

                cap = (
                    cuda
                    .get_current_device()
                    .compute_capability
                )

                capability = [
                    int(cap[0]),
                    int(cap[1]),
                ]

        except Exception:

            pass


        return {

            "status": "ok",

            "gpu_details":
                process.stdout.strip(),

            "compute_capability":
                capability,

            "cuda_available":
                capability is not None,

            "hostname":
                socket.gethostname(),

            "python":
                sys.version,

            "platform":
                platform.platform(),
        }


    except Exception as exc:

        return {

            "status": "error",

            "error":
                str(exc),

            "traceback":
                traceback.format_exc(),
        }


# ============================================================
# EXECUTE OPERATION
# ============================================================

def execute_command(
    command
):

    command_id = command.get(
        "command_id"
    )

    operation = command.get(
        "operation",
        "",
    )

    parameters = command.get(
        "parameters",
        {},
    )


    log()
    log("=" * 60)
    log("COMMAND RECEIVED")
    log("=" * 60)

    log(
        f"ID: {command_id}"
    )

    log(
        f"OPERATION: {operation}"
    )


    try:

        if operation == "execute_python":

            result = execute_python(
                parameters
            )


        elif operation == "nvidia_smi":

            result = execute_nvidia_smi()


        elif operation == "shell":

            result = execute_shell(
                parameters
            )


        elif operation == "info":

            result = execute_info()


        else:

            result = {

                "status": "error",

                "error":
                    f"Unknown operation: "
                    f"{operation}",
            }


    except Exception as exc:

        result = {

            "status": "error",

            "error":
                str(exc),

            "traceback":
                traceback.format_exc(),
        }


    send_result(
        command_id,
        result
    )


# ============================================================
# 120-SECOND TEST LOOP
# ============================================================

def run_test_loop():

    log()
    log("=" * 60)
    log(
        f"RUNNING FOR "
        f"{WORKER_RUNTIME_SECONDS} SECONDS"
    )
    log("=" * 60)


    started = time.monotonic()

    last_heartbeat = 0

    loop_number = 0


    while True:

        elapsed = (
            time.monotonic()
            - started
        )


        # ----------------------------------------------------
        # End after exactly ~120 seconds
        # ----------------------------------------------------

        if (
            elapsed
            >= WORKER_RUNTIME_SECONDS
        ):

            log()
            log("=" * 60)
            log(
                "120-SECOND TEST FINISHED"
            )
            log("=" * 60)

            break


        loop_number += 1


        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if (
            time.monotonic()
            - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            ok = heartbeat()

            last_heartbeat = (
                time.monotonic()
            )


            remaining = max(
                0,
                int(
                    WORKER_RUNTIME_SECONDS
                    - elapsed
                )
            )


            if ok:

                log(
                    f"[KEEP-ALIVE] "
                    f"OK | "
                    f"remaining={remaining}s"
                )

            else:

                log(
                    f"[KEEP-ALIVE] "
                    f"FAILED | "
                    f"remaining={remaining}s"
                )


        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        try:

            command = get_command()


            if command:

                if command.get(
                    "_expired"
                ):

                    log(
                        "Railway says "
                        "session expired."
                    )

                    break


                execute_command(
                    command
                )


        except Exception as exc:

            log(
                f"[COMMAND LOOP ERROR] "
                f"{exc}"
            )

            traceback.print_exc()


        # ----------------------------------------------------
        # Sleep only up to 3 seconds
        # ----------------------------------------------------

        time.sleep(
            COMMAND_INTERVAL
        )


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 60)
    log("KAGGLE GPU WORKER - 120 SECOND TEST")
    log("=" * 60)


    validate_config()


    log(
        f"SESSION : {SESSION_ID}"
    )

    log(
        f"API     : {API_URL}"
    )

    # Never print actual token.
    log(
        f"TOKEN   : {len(WORKER_TOKEN)} chars"
    )


    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    show_nvidia_smi()


    gpu_info = get_gpu_info()


    if not gpu_info:

        raise RuntimeError(
            "Could not read GPU info"
        )


    log(
        f"GPU: {gpu_info}"
    )


    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    cuda_result = cuda_test()


    if not cuda_result.get(
        "ok"
    ):

        raise RuntimeError(
            "CUDA test failed: "
            + str(
                cuda_result.get(
                    "error"
                )
            )
        )


    # --------------------------------------------------------
    # READY
    # --------------------------------------------------------

    if not worker_ready(
        gpu_info,
        cuda_result,
    ):

        raise RuntimeError(
            "worker-ready failed"
        )


    # --------------------------------------------------------
    # 120 SECOND LOOP
    # --------------------------------------------------------

    run_test_loop()


    log(
        "Worker exiting normally."
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        log(
            "Worker interrupted."
        )

    except Exception as exc:

        log()
        log("=" * 60)
        log("WORKER ERROR")
        log("=" * 60)

        log(
            str(exc)
        )

        traceback.print_exc()

        sys.exit(1)
