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
# These 3 variables are prepended by GitHub Actions.
# Do NOT define them again later in this file.
# ============================================================

# SESSION_ID
# API_URL
# WORKER_TOKEN


HEARTBEAT_INTERVAL = 15
COMMAND_INTERVAL = 3
REQUEST_TIMEOUT = 20
MAX_OUTPUT = 50000


def log(message=""):
    print(message, flush=True)


def validate_config():
    missing = []

    if "SESSION_ID" not in globals():
        missing.append("SESSION_ID")

    if "API_URL" not in globals():
        missing.append("API_URL")

    if "WORKER_TOKEN" not in globals():
        missing.append("WORKER_TOKEN")

    if missing:
        raise RuntimeError(
            "Runtime config missing: "
            + ", ".join(missing)
        )

    if not SESSION_ID:
        raise RuntimeError("SESSION_ID is empty")

    if not API_URL:
        raise RuntimeError("API_URL is empty")

    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN is empty")

    if not API_URL.startswith(("http://", "https://")):
        raise RuntimeError(
            f"Invalid API_URL: {API_URL}"
        )


def headers():
    return {
        "Content-Type": "application/json",
        "X-Worker-Token": WORKER_TOKEN,
        "Authorization": f"Bearer {WORKER_TOKEN}",
    }


def api_get(path, retries=3):
    url = f"{API_URL}{path}"

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                url,
                headers=headers(),
                timeout=REQUEST_TIMEOUT,
            )

            log(
                f"[API] GET {path} -> {r.status_code}"
            )

            return r

        except Exception as e:
            log(
                f"[API] GET {path} "
                f"attempt={attempt} error={e}"
            )

            if attempt < retries:
                time.sleep(2)

    return None


def api_post(path, payload=None, retries=3):
    url = f"{API_URL}{path}"

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                url,
                headers=headers(),
                json=payload or {},
                timeout=REQUEST_TIMEOUT,
            )

            log(
                f"[API] POST {path} -> {r.status_code}"
            )

            if r.status_code >= 400:
                log(r.text[:2000])

            return r

        except Exception as e:
            log(
                f"[API] POST {path} "
                f"attempt={attempt} error={e}"
            )

            if attempt < retries:
                time.sleep(2)

    return None


def limit_output(value, limit=MAX_OUTPUT):
    value = str(value)

    if len(value) <= limit:
        return value

    return value[:limit] + "\n...[TRUNCATED]..."


def show_nvidia_smi():
    log()
    log("=" * 60)
    log("NVIDIA-SMI")
    log("=" * 60)

    try:
        p = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        log(p.stdout)

        if p.stderr:
            log(p.stderr)

        if p.returncode != 0:
            raise RuntimeError(
                "nvidia-smi failed"
            )

    except Exception:
        traceback.print_exc()
        raise


def get_gpu_info():
    try:
        p = subprocess.run(
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

        if p.returncode != 0:
            return None

        return p.stdout.strip()

    except Exception:
        return None


def cuda_test():
    log()
    log("=" * 60)
    log("CUDA / NUMBA TEST")
    log("=" * 60)

    try:
        import numpy as np
        from numba import cuda

    except ImportError:
        log("Installing numba/numpy...")

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

    if not cuda.is_available():
        return {
            "ok": False,
            "error": "CUDA unavailable",
            "compute_capability": None,
        }

    device = cuda.get_current_device()
    capability = device.compute_capability

    log(f"CUDA available: True")
    log(f"GPU: {device.name}")
    log(f"Compute capability: {capability}")

    try:
        n = 1024 * 1024

        a = np.ones(n, np.float32)
        b = np.ones(n, np.float32)
        c = np.zeros(n, np.float32)

        @cuda.jit
        def add_kernel(a, b, c):
            i = cuda.grid(1)

            if i < c.size:
                c[i] = a[i] + b[i]

        threads = 256
        blocks = (n + threads - 1) // threads

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dc = cuda.to_device(c)

        start = time.perf_counter()

        add_kernel[blocks, threads](da, db, dc)

        cuda.synchronize()

        elapsed = time.perf_counter() - start

        result = dc.copy_to_host()

        if not np.allclose(result, 2.0):
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

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "compute_capability": [
                int(capability[0]),
                int(capability[1]),
            ],
        }


def notify_ready(gpu_info, cuda_result):
    log()
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)

    payload = {
        "gpu": gpu_info,
        "cuda_available": bool(
            cuda_result.get("ok")
        ),
        "compute_capability": cuda_result.get(
            "compute_capability"
        ),
    }

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):
        log(
            f"worker-ready attempt "
            f"{attempt}/10"
        )

        r = api_post(
            path,
            payload,
            retries=1,
        )

        if r is not None and r.status_code in (200, 202):
            log(
                "WORKER READY accepted by Railway."
            )
            return True

        time.sleep(3)

    return False


def heartbeat():
    r = api_post(
        f"/gpu/session/{SESSION_ID}/heartbeat",
        {
            "timestamp": time.time(),
        },
        retries=2,
    )

    if r is None:
        return False

    if r.status_code == 410:
        log("Session expired.")
        return False

    return r.status_code in (200, 202)


def get_command():
    r = api_get(
        f"/internal/session/{SESSION_ID}/command",
        retries=2,
    )

    if r is None or r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    if data.get("expired"):
        return {
            "_expired": True
        }

    command = data.get("command")

    if not command:
        return None

    return command


def send_result(command_id, result):
    r = api_post(
        f"/internal/session/{SESSION_ID}/result",
        {
            "command_id": command_id,
            **result,
        },
        retries=4,
    )

    return (
        r is not None
        and r.status_code in (200, 201, 202)
    )


def execute_python(parameters):
    code = parameters.get("code", "")

    if not code:
        return {
            "status": "error",
            "error": "Python code is empty",
        }

    namespace = {
        "__builtins__": __builtins__,
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
            if key.startswith("_"):
                continue

            try:
                variables[key] = limit_output(
                    repr(value),
                    10000,
                )
            except Exception:
                variables[key] = "<unprintable>"

        return {
            "status": "ok",
            "stdout": limit_output(
                stdout.getvalue()
            ),
            "stderr": limit_output(
                stderr.getvalue()
            ),
            "variables": variables,
            "execution_time":
                time.perf_counter() - started,
        }

    except Exception as e:
        return {
            "status": "error",
            "stdout": limit_output(
                stdout.getvalue()
            ),
            "stderr": limit_output(
                stderr.getvalue()
            ),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_time":
                time.perf_counter() - started,
        }


def execute_nvidia_smi():
    try:
        p = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        return {
            "status": "ok",
            "stdout": limit_output(
                p.stdout
            ),
            "stderr": limit_output(
                p.stderr
            ),
            "returncode": p.returncode,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def execute_shell(parameters):
    command = parameters.get("command", "")

    if not command:
        return {
            "status": "error",
            "error": "Shell command is empty",
        }

    try:
        p = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        return {
            "status": "ok",
            "stdout": limit_output(
                p.stdout
            ),
            "stderr": limit_output(
                p.stderr
            ),
            "returncode": p.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Command timed out",
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def execute_info():
    try:
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
            "gpu_details": p.stdout.strip(),
            "compute_capability": capability,
            "cuda_available": (
                capability is not None
            ),
            "hostname": socket.gethostname(),
            "python": sys.version,
            "platform": platform.platform(),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def execute_command(command):
    command_id = command.get("command_id")
    operation = command.get("operation", "")
    parameters = command.get("parameters", {})

    log()
    log("=" * 60)
    log("COMMAND RECEIVED")
    log("=" * 60)
    log(f"ID: {command_id}")
    log(f"OPERATION: {operation}")

    try:
        if operation == "execute_python":
            result = execute_python(parameters)

        elif operation == "nvidia_smi":
            result = execute_nvidia_smi()

        elif operation == "shell":
            result = execute_shell(parameters)

        elif operation == "info":
            result = execute_info()

        else:
            result = {
                "status": "error",
                "error": (
                    f"Unknown operation: {operation}"
                ),
            }

    except Exception as e:
        result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    send_result(
        command_id,
        result,
    )


def command_loop():
    log()
    log("=" * 60)
    log("KEEP-ALIVE + COMMAND LOOP")
    log("=" * 60)

    last_heartbeat = 0
    consecutive_heartbeat_failures = 0

    while True:
        now = time.time()

        # Heartbeat
        if (
            now - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):
            ok = heartbeat()

            last_heartbeat = now

            if ok:
                consecutive_heartbeat_failures = 0
                log("[KEEP-ALIVE] heartbeat OK")
            else:
                consecutive_heartbeat_failures += 1
                log(
                    "[KEEP-ALIVE] "
                    f"heartbeat failed "
                    f"({consecutive_heartbeat_failures})"
                )

        # Command
        try:
            command = get_command()

            if command:
                if command.get("_expired"):
                    log("Session expired.")
                    return

                execute_command(command)

        except Exception as e:
            log(
                f"[COMMAND ERROR] {e}"
            )
            traceback.print_exc()

        time.sleep(COMMAND_INTERVAL)


def main():
    log("=" * 60)
    log("KAGGLE GPU WORKER")
    log("=" * 60)

    validate_config()

    log(
        f"SESSION : {SESSION_ID}"
    )
    log(
        f"API     : {API_URL}"
    )
    log(
        f"TOKEN   : {len(WORKER_TOKEN)} chars"
    )

    # GPU
    show_nvidia_smi()

    gpu_info = get_gpu_info()

    if not gpu_info:
        raise RuntimeError(
            "GPU information unavailable"
        )

    log(f"GPU: {gpu_info}")

    # CUDA
    cuda_result = cuda_test()

    if not cuda_result.get("ok"):
        raise RuntimeError(
            "CUDA test failed: "
            + str(
                cuda_result.get("error")
            )
        )

    # Ready
    if not notify_ready(
        gpu_info,
        cuda_result,
    ):
        raise RuntimeError(
            "Railway did not accept worker-ready"
        )

    # Keep alive
    command_loop()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        log("Worker interrupted.")

    except Exception as e:
        log("=" * 60)
        log("WORKER ERROR")
        log("=" * 60)
        log(str(e))
        traceback.print_exc()
        sys.exit(1)
