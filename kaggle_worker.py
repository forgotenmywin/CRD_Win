import os
import sys
import time
import json
import traceback
import subprocess

import requests
import numpy as np
from numba import cuda


# ============================================================
# CONFIG
# ============================================================

SESSION_ID = os.environ.get("SESSION_ID", "").strip()
API_URL = os.environ.get("API_URL", "").strip().rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "").strip()

# مدت کار Worker
TEST_SECONDS = 600

# هر چند ثانیه heartbeat
HEARTBEAT_INTERVAL = 30

# هر چند ثانیه command را بررسی کنیم
COMMAND_INTERVAL = 30

# هر چند ثانیه یک کار کوچک واقعی روی GPU
GPU_KEEPALIVE_INTERVAL = 10

REQUEST_TIMEOUT = 15


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "X-Worker-Token": WORKER_TOKEN,
    "Content-Type": "application/json",
}


# ============================================================
# PRINT
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

    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN was not injected")

    if not API_URL.startswith("http://") and not API_URL.startswith("https://"):
        raise RuntimeError(
            f"API_URL must start with http:// or https://, got: {API_URL}"
        )


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
            timeout=20
        )

        log(result.stdout)

        if result.stderr:
            log(result.stderr)

    except Exception as e:
        log(f"nvidia-smi error: {e}")


# ============================================================
# GPU KEEP-ALIVE KERNEL
# ============================================================

@cuda.jit
def keep_alive_kernel(x):

    i = cuda.grid(1)

    if i < x.size:
        x[i] = x[i] * 1.0001


# ============================================================
# GPU KEEP-ALIVE
# ============================================================

def gpu_keepalive():

    try:

        n = 1024

        host_data = np.ones(
            n,
            dtype=np.float32
        )

        device_data = cuda.to_device(host_data)

        keep_alive_kernel[4, 256](device_data)

        cuda.synchronize()

        # برگرداندن نتیجه تا مطمئن شویم kernel واقعاً اجرا شده
        device_data.copy_to_host(host_data)

        return True

    except Exception as e:

        log()
        log("[GPU KEEP-ALIVE ERROR]")
        log(repr(e))

        return False


# ============================================================
# INITIAL GPU TEST
# ============================================================

def gpu_test():

    log()
    log("=" * 60)
    log("CUDA TEST")
    log("=" * 60)

    try:

        device = cuda.get_current_device()

        log(
            f"GPU: {device.name.decode() if isinstance(device.name, bytes) else device.name}"
        )

        try:

            cc = device.compute_capability

            log(
                f"Compute capability: {cc}"
            )

        except Exception:

            cc = None

        # یک تست کوچک واقعی با Numba
        n = 1024 * 1024

        data = np.ones(
            n,
            dtype=np.float32
        )

        device_data = cuda.to_device(data)

        start = time.perf_counter()

        keep_alive_kernel[4096, 256](device_data)

        cuda.synchronize()

        elapsed = time.perf_counter() - start

        device_data.copy_to_host(data)

        log(
            f"GPU JOB SUCCESS: {n} elements"
        )

        log(
            f"GPU kernel time: {elapsed:.4f}s"
        )

        return True

    except Exception as e:

        log()
        log("CUDA ERROR:")
        log(repr(e))

        traceback.print_exc()

        return False


# ============================================================
# API REQUEST
# ============================================================

def api_post(path, payload=None):

    url = API_URL + path

    try:

        response = requests.post(
            url,
            headers=HEADERS,
            json=payload or {},
            timeout=REQUEST_TIMEOUT
        )

        log(
            f"[API] POST {path} -> {response.status_code}"
        )

        try:
            data = response.json()
        except Exception:
            data = response.text

        log(
            str(data)
        )

        return response.status_code, data

    except Exception as e:

        log(
            f"[API] POST ERROR {path}: {e}"
        )

        return None, None


# ============================================================
# API GET
# ============================================================

def api_get(path):

    url = API_URL + path

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        log(
            f"[API] GET {path} -> {response.status_code}"
        )

        try:
            data = response.json()
        except Exception:
            data = response.text

        return response.status_code, data

    except Exception as e:

        log(
            f"[API] GET ERROR {path}: {e}"
        )

        return None, None


# ============================================================
# WORKER READY
# ============================================================

def worker_ready():

    log()
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)

    path = (
        f"/gpu/session/{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):

        log(
            f"worker-ready attempt {attempt}/10"
        )

        status, data = api_post(path)

        if status == 200:

            log()
            log("WORKER READY accepted by Railway.")

            return True

        if status == 401:

            log()
            log(
                "ERROR: Railway rejected WORKER_TOKEN (401)."
            )

            # ادامه دادن بی‌فایده است
            return False

        time.sleep(3)

    return False


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():

    path = (
        f"/gpu/session/{SESSION_ID}/heartbeat"
    )

    status, data = api_post(path)

    if status == 200:

        remaining = None

        if isinstance(data, dict):

            remaining = data.get(
                "remaining_seconds"
            )

        if remaining is not None:

            log(
                f"[KEEP-ALIVE] remaining={remaining}s"
            )

        return True

    return False


# ============================================================
# COMMAND
# ============================================================

def get_command():

    path = (
        f"/internal/session/{SESSION_ID}/command"
    )

    status, data = api_get(path)

    if status == 200:

        log(
            f"[COMMAND] {data}"
        )

        if isinstance(data, dict):

            command = data.get("command")

            if command:
                return command

    return None


# ============================================================
# EXECUTE COMMAND
# ============================================================

def execute_command(command):

    log()
    log("=" * 60)
    log("COMMAND RECEIVED")
    log("=" * 60)

    log(
        str(command)
    )

    # فعلاً فقط برای تست
    # بعداً می‌توان این قسمت را به job واقعی GPU وصل کرد

    if command == "gpu_test":

        return gpu_keepalive()

    return True


# ============================================================
# WORKER LOOP
# ============================================================

def worker_loop():

    log()
    log("=" * 60)
    log("GPU WORKER IS ACTIVE")
    log("=" * 60)

    start_time = time.monotonic()

    last_heartbeat = 0.0
    last_command = 0.0
    last_gpu_keepalive = 0.0

    while True:

        now = time.monotonic()

        elapsed = int(
            now - start_time
        )

        # ----------------------------------------------------
        # SESSION TIME
        # ----------------------------------------------------

        if elapsed >= TEST_SECONDS:

            log()
            log("=" * 60)
            log("WORKER SESSION FINISHED")
            log("=" * 60)

            log(
                f"Runtime: {elapsed}s"
            )

            break

        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if (
            now - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            heartbeat()

            last_heartbeat = now

        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        if (
            now - last_command
            >= COMMAND_INTERVAL
        ):

            command = get_command()

            if command:

                execute_command(
                    command
                )

            last_command = now

        # ----------------------------------------------------
        # GPU KEEP ALIVE
        # ----------------------------------------------------

        if (
            now - last_gpu_keepalive
            >= GPU_KEEPALIVE_INTERVAL
        ):

            ok = gpu_keepalive()

            if ok:

                log(
                    f"[GPU] keep-alive OK at {elapsed}s"
                )

            else:

                log(
                    f"[GPU] keep-alive FAILED at {elapsed}s"
                )

            last_gpu_keepalive = now

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        log(
            f"[WORKER] {elapsed}/{TEST_SECONDS}s"
        )

        # ----------------------------------------------------
        # SMALL SLEEP
        # ----------------------------------------------------

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
        f"TOKEN: {len(WORKER_TOKEN)} chars"
    )

    log()

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    validate_config()

    # --------------------------------------------------------
    # GPU INFO
    # --------------------------------------------------------

    show_gpu()

    # --------------------------------------------------------
    # GPU TEST
    # --------------------------------------------------------

    if not gpu_test():

        raise RuntimeError(
            "GPU test failed"
        )

    # --------------------------------------------------------
    # RAILWAY READY
    # --------------------------------------------------------

    if not worker_ready():

        raise RuntimeError(
            "Could not notify Railway"
        )

    # --------------------------------------------------------
    # START LOOP
    # --------------------------------------------------------

    worker_loop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        log()
        log("Worker stopped by user.")

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
