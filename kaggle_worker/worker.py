import os
import sys
import time
import traceback
import subprocess
import requests


# ============================================================
# CONFIG
# ============================================================

SESSION_ID = os.environ.get("SESSION_ID", "")
API_URL = os.environ.get("API_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

TEST_SECONDS = 600

HEARTBEAT_INTERVAL = 30
COMMAND_INTERVAL = 30

REQUEST_TIMEOUT = 15


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "X-Worker-Token": WORKER_TOKEN,
    "Content-Type": "application/json",
}


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message, flush=True)


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config():

    if not SESSION_ID:
        raise RuntimeError("SESSION_ID was not injected")

    if not API_URL:
        raise RuntimeError("API_URL was not injected")

    if not API_URL.startswith("http://") and not API_URL.startswith("https://"):
        raise RuntimeError(
            f"Invalid API_URL: {API_URL}"
        )

    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN was not injected")


# ============================================================
# API REQUEST
# ============================================================

def api_post(path, payload=None):

    url = f"{API_URL}{path}"

    try:

        response = requests.post(
            url,
            headers=HEADERS,
            json=payload or {},
            timeout=REQUEST_TIMEOUT,
        )

        log(
            f"[API] POST {path} -> "
            f"{response.status_code}"
        )

        try:
            data = response.json()
        except Exception:
            data = response.text

        log(str(data))

        return response, data

    except Exception as e:

        log(
            f"[API] POST ERROR {path}: "
            f"{e}"
        )

        return None, None


def api_get(path):

    url = f"{API_URL}{path}"

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        log(
            f"[API] GET {path} -> "
            f"{response.status_code}"
        )

        try:
            data = response.json()
        except Exception:
            data = response.text

        return response, data

    except Exception as e:

        log(
            f"[API] GET ERROR {path}: "
            f"{e}"
        )

        return None, None


# ============================================================
# GPU INFORMATION
# ============================================================

def show_gpu():

    log()
    log("=" * 60)
    log("NVIDIA-SMI")
    log("=" * 60)

    try:

        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=20,
        )

        print(
            result.stdout,
            flush=True
        )

        if result.returncode != 0:
            print(
                result.stderr,
                flush=True
            )

    except Exception as e:

        log(
            f"nvidia-smi error: {e}"
        )


# ============================================================
# CUDA TEST
# ============================================================

def gpu_test():

    log()
    log("=" * 60)
    log("CUDA TEST")
    log("=" * 60)

    try:

        import torch

        if not torch.cuda.is_available():

            log("CUDA available: False")

            return False

        device = torch.device("cuda")

        gpu_name = torch.cuda.get_device_name(0)

        capability = torch.cuda.get_device_capability(0)

        log(
            f"GPU: {gpu_name}"
        )

        log(
            f"Compute capability: "
            f"{capability}"
        )

        # Real GPU operation
        size = 1024 * 1024

        start = time.time()

        x = torch.ones(
            size,
            device=device,
            dtype=torch.float32,
        )

        y = x * 2

        torch.cuda.synchronize()

        elapsed = time.time() - start

        result = int(
            y.sum().item()
        )

        expected = size * 2

        del x
        del y

        torch.cuda.empty_cache()

        if result != expected:

            log(
                f"GPU result incorrect: "
                f"{result} != {expected}"
            )

            return False

        log(
            f"GPU JOB SUCCESS: "
            f"{size} elements"
        )

        log(
            f"GPU kernel time: "
            f"{elapsed:.4f}s"
        )

        return True

    except Exception as e:

        log(
            f"GPU TEST ERROR: {e}"
        )

        traceback.print_exc()

        return False


# ============================================================
# WORKER READY
# ============================================================

def worker_ready():

    log()
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):

        log(
            f"worker-ready attempt "
            f"{attempt}/10"
        )

        response, data = api_post(
            path,
            {
                "gpu": "Tesla P100-PCIE-16GB",
                "cuda_available": True,
            },
        )

        if response is not None:

            if response.status_code == 200:

                log(
                    "WORKER READY accepted."
                )

                return True

            if response.status_code == 401:

                log(
                    "ERROR: Railway rejected "
                    "WORKER_TOKEN (401)."
                )

        time.sleep(3)

    return False


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/heartbeat"
    )

    response, data = api_post(
        path
    )

    if response is None:
        return None

    if response.status_code != 200:
        return None

    return data


# ============================================================
# COMMAND
# ============================================================

def get_command():

    path = (
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )

    response, data = api_get(
        path
    )

    if response is None:
        return None

    if response.status_code != 200:
        return None

    if isinstance(data, dict):

        command = data.get(
            "command"
        )

        log(
            f"[COMMAND] "
            f"{data}"
        )

        return command

    return None


# ============================================================
# EXECUTE COMMAND
# ============================================================

def execute_command(command):

    if not command:
        return

    log()
    log("=" * 60)
    log("EXECUTING GPU COMMAND")
    log("=" * 60)

    log(
        f"COMMAND: {command}"
    )

    try:

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
        )

        log(
            f"EXIT CODE: "
            f"{result.returncode}"
        )

        if result.stdout:
            log(result.stdout)

        if result.stderr:
            log(result.stderr)

    except Exception as e:

        log(
            f"COMMAND ERROR: {e}"
        )


# ============================================================
# WORKER LOOP
# ============================================================

def worker_loop():

    log()
    log("=" * 60)
    log("GPU WORKER IS ACTIVE")
    log("=" * 60)

    start_time = time.time()

    last_heartbeat = 0
    last_command = 0

    while True:

        elapsed = int(
            time.time() - start_time
        )

        if elapsed >= TEST_SECONDS:

            log()
            log("=" * 60)
            log(
                f"WORKER FINISHED "
                f"{TEST_SECONDS}s"
            )
            log("=" * 60)

            break

        now = time.time()

        # -----------------------------
        # HEARTBEAT
        # -----------------------------

        if (
            now - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            data = heartbeat()

            last_heartbeat = now

            if data:

                remaining = data.get(
                    "remaining_seconds"
                )

                log(
                    f"[KEEP-ALIVE] "
                    f"remaining={remaining}"
                )

        # -----------------------------
        # COMMAND
        # -----------------------------

        if (
            now - last_command
            >= COMMAND_INTERVAL
        ):

            command = get_command()

            last_command = now

            if command:

                execute_command(
                    command
                )

        log(
            f"[WORKER] "
            f"{elapsed}/{TEST_SECONDS}s"
        )

        time.sleep(1)


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 60)
    log("KAGGLE GPU WORKER")
    log("=" * 60)

    log(
        f"SESSION: {SESSION_ID}"
    )

    log(
        f"API: {API_URL}"
    )

    log(
        f"TOKEN: "
        f"{len(WORKER_TOKEN)} chars"
    )

    validate_config()

    show_gpu()

    if not gpu_test():

        raise RuntimeError(
            "GPU test failed"
        )

    if not worker_ready():

        raise RuntimeError(
            "Could not notify Railway"
        )

    worker_loop()


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

        sys.exit(0)

    except Exception as e:

        log()
        log("=" * 60)
        log("WORKER ERROR")
        log("=" * 60)

        log(
            repr(e)
        )

        traceback.print_exc()

        sys.exit(1)
