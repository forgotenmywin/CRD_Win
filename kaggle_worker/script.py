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
#
# GitHub Actions prepends the REAL values before pushing
# this file to Kaggle.
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"


# ============================================================
# CONFIG
# ============================================================

HEARTBEAT_INTERVAL = 15
COMMAND_INTERVAL = 3
REQUEST_TIMEOUT = 20
MAX_OUTPUT = 50000


# ============================================================
# VALIDATION
# ============================================================

def validate_config():

    if not SESSION_ID:
        raise RuntimeError("SESSION_ID is empty")

    if SESSION_ID == "__SESSION_ID__":
        raise RuntimeError("SESSION_ID was not injected")

    if not API_URL:
        raise RuntimeError("API_URL is empty")

    if API_URL == "__API_URL__":
        raise RuntimeError("API_URL was not injected")

    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN is empty")

    if WORKER_TOKEN == "__WORKER_TOKEN__":
        raise RuntimeError("WORKER_TOKEN was not injected")

    if not API_URL.startswith(
        ("http://", "https://")
    ):
        raise RuntimeError(
            f"Invalid API_URL: {API_URL}"
        )


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message, flush=True)


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "Content-Type": "application/json",
    "X-Worker-Token": WORKER_TOKEN,
}


def api_get(path, retries=3):

    url = f"{API_URL}{path}"

    for attempt in range(1, retries + 1):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            log(
                f"[API] GET {path} "
                f"→ {response.status_code}"
            )

            return response

        except Exception as exc:

            log(
                f"[API] GET {path} "
                f"attempt={attempt} "
                f"error={exc}"
            )

            if attempt < retries:
                time.sleep(2)

    return None


def api_post(path, payload=None, retries=3):

    url = f"{API_URL}{path}"

    for attempt in range(1, retries + 1):

        try:

            response = requests.post(
                url,
                headers=HEADERS,
                json=payload or {},
                timeout=REQUEST_TIMEOUT,
            )

            log(
                f"[API] POST {path} "
                f"→ {response.status_code}"
            )

            if response.status_code >= 400:
                log(
                    response.text[:2000]
                )

            return response

        except Exception as exc:

            log(
                f"[API] POST {path} "
                f"attempt={attempt} "
                f"error={exc}"
            )

            if attempt < retries:
                time.sleep(2)

    return None


# ============================================================
# OUTPUT LIMIT
# ============================================================

def limit_output(value, limit=MAX_OUTPUT):

    value = str(value)

    if len(value) <= limit:
        return value

    return (
        value[:limit]
        + "\n...[OUTPUT TRUNCATED]..."
    )


# ============================================================
# NVIDIA SMI
# ============================================================

def run_nvidia_smi():

    log()
    log("=" * 60)
    log("NVIDIA-SMI")
    log("=" * 60)

    try:

        process = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        log(process.stdout)

        if process.stderr:
            log(process.stderr)

        return process

    except Exception as exc:

        log(
            f"nvidia-smi error: {exc}"
        )

        return None


def get_gpu_info():

    try:

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

    except Exception:
        return None


# ============================================================
# CUDA / NUMBA TEST
#
# IMPORTANT:
# No PyTorch CUDA test.
#
# Kaggle currently provides a PyTorch build that does not
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
            "Numba/Numpy missing. Installing..."
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

    try:

        device = cuda.get_current_device()

        capability = device.compute_capability

        log(
            f"GPU: {device.name}"
        )

        log(
            f"Compute capability: {capability}"
        )

    except Exception as exc:

        return {
            "ok": False,
            "error": str(exc),
            "compute_capability": None,
        }


    # ========================================================
    # REAL CUDA KERNEL TEST
    # ========================================================

    try:

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

            raise RuntimeError(
                "CUDA result verification failed"
            )


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


    except Exception as exc:

        log(
            "CUDA kernel test failed."
        )

        traceback.print_exc()

        return {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "compute_capability": [
                int(capability[0]),
                int(capability[1]),
            ],
        }


# ============================================================
# WORKER READY
# ============================================================

def notify_worker_ready(
    gpu_info,
    cuda_result,
):

    log()
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)


    payload = {

        "gpu": gpu_info,

        "compute_capability":
            cuda_result.get(
                "compute_capability"
            ),

        "cuda_available":
            bool(
                cuda_result.get("ok")
            ),
    }


    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )


    for attempt in range(1, 11):

        log(
            f"worker-ready "
            f"attempt {attempt}/10"
        )


        response = api_post(
            path,
            payload,
            retries=1,
        )


        if response is not None:

            if response.status_code in (
                200,
                202,
            ):

                log(
                    "WORKER READY accepted by Railway."
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
            "timestamp": time.time()
        },
        retries=2,
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
# GET COMMAND
# ============================================================

def get_command():

    response = api_get(
        f"/internal/session/"
        f"{SESSION_ID}/command",
        retries=2,
    )


    if response is None:
        return None


    if response.status_code != 200:
        return None


    try:

        data = response.json()

    except Exception:

        return None


    if not isinstance(data, dict):
        return None


    if data.get("expired"):

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
# SEND RESULT
# ============================================================

def send_result(
    command_id,
    result,
):

    payload = {
        "command_id": command_id,
        **result,
    }


    response = api_post(
        f"/internal/session/"
        f"{SESSION_ID}/result",
        payload,
        retries=4,
    )


    if response is not None:

        return response.status_code in (
            200,
            201,
            202,
        )


    return False


# ============================================================
# EXECUTE PYTHON
# ============================================================

def execute_python(parameters):

    code = parameters.get(
        "code",
        "",
    )


    if not code:

        return {
            "status": "error",
            "error": "Python code is empty",
        }


    namespace = {
        "__builtins__": __builtins__,
    }


    # Numpy / Numba
    try:

        import numpy as np
        from numba import cuda

        namespace["np"] = np
        namespace["cuda"] = cuda

    except Exception:

        pass


    # PyTorch is available for CPU-side code,
    # but do NOT use CUDA tensors with this Kaggle build
    # on the P100.
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

            if key.startswith("_"):
                continue

            try:

                variables[key] = limit_output(
                    repr(value),
                    10000,
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

            "variables": variables,

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
# NVIDIA SMI COMMAND
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

def execute_shell(parameters):

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
            timeout=120,
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
                "Command timed out",
        }


    except Exception as exc:

        return {

            "status": "error",

            "error":
                str(exc),
        }


# ============================================================
# INFO COMMAND
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
# EXECUTE COMMAND
# ============================================================

def execute_command(command):

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
        result,
    )


# ============================================================
# KEEP-ALIVE + COMMAND LOOP
# ============================================================

def command_loop():

    log()
    log("=" * 60)
    log("KEEP-ALIVE + COMMAND LOOP")
    log("=" * 60)


    last_heartbeat = 0
    counter = 0


    while True:

        counter += 1

        now = time.time()


        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if (
            now - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            log(
                f"[KEEP-ALIVE] "
                f"heartbeat #{counter}"
            )


            ok = heartbeat()


            last_heartbeat = now


            if ok:

                log(
                    "[KEEP-ALIVE] "
                    "heartbeat OK"
                )

            else:

                log(
                    "[KEEP-ALIVE] "
                    "heartbeat failed"
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
                        "Session expired."
                    )

                    break


                execute_command(
                    command
                )


        except Exception as exc:

            log(
                f"[COMMAND ERROR] {exc}"
            )

            traceback.print_exc()


        time.sleep(
            COMMAND_INTERVAL
        )


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 60)
    log("KAGGLE GPU WORKER")
    log("=" * 60)


    validate_config()


    log()
    log(
        f"SESSION : {SESSION_ID}"
    )

    log(
        f"API     : {API_URL}"
    )

    # Never print the actual token.
    log(
        f"TOKEN   : {len(WORKER_TOKEN)} chars"
    )


    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    gpu_test_process = run_nvidia_smi()


    if gpu_test_process is None:

        raise RuntimeError(
            "Unable to run nvidia-smi"
        )


    if gpu_test_process.returncode != 0:

        raise RuntimeError(
            "nvidia-smi failed"
        )


    gpu_info = get_gpu_info()


    if not gpu_info:

        raise RuntimeError(
            "Could not read GPU information"
        )


    log(
        f"GPU: {gpu_info}"
    )


    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    cuda_result = cuda_test()


    if not cuda_result.get("ok"):

        raise RuntimeError(
            "CUDA test failed: "
            + str(
                cuda_result.get(
                    "error"
                )
            )
        )


    # --------------------------------------------------------
    # WORKER READY
    # --------------------------------------------------------

    if not notify_worker_ready(
        gpu_info,
        cuda_result,
    ):

        raise RuntimeError(
            "Railway did not accept worker-ready"
        )


    # --------------------------------------------------------
    # KEEP ALIVE
    # --------------------------------------------------------

    command_loop()


# ============================================================
# ENTRY
# ============================================================

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
