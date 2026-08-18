import os
import time
import traceback
import requests
import subprocess

# ============================================================
# CONFIG
# ============================================================

SESSION_ID = os.environ.get("SESSION_ID", "").strip()
API_URL = os.environ.get("API_URL", "").strip().rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "").strip()

TEST_SECONDS = 600
HEARTBEAT_INTERVAL = 30
COMMAND_INTERVAL = 30

REQUEST_TIMEOUT = 15
READY_RETRIES = 10
READY_RETRY_DELAY = 3

# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    if not SESSION_ID:
        raise RuntimeError("SESSION_ID was not injected")

    if not API_URL:
        raise RuntimeError("API_URL was not injected")

    if not API_URL.startswith(("http://", "https://")):
        raise RuntimeError(f"Invalid API_URL: {API_URL}")

    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN was not injected")


# ============================================================
# HEADERS
# ============================================================

def headers():
    return {
        "X-Worker-Token": WORKER_TOKEN,
        "Content-Type": "application/json",
        "User-Agent": "kaggle-gpu-worker/1.0",
    }


# ============================================================
# URLS
# ============================================================

def worker_ready_url():
    return f"{API_URL}/gpu/session/{SESSION_ID}/worker-ready"


def heartbeat_url():
    return f"{API_URL}/gpu/session/{SESSION_ID}/heartbeat"


def command_url():
    return f"{API_URL}/internal/session/{SESSION_ID}/command"


# ============================================================
# HTTP HELPERS
# ============================================================

def post(url, payload=None):
    try:
        r = requests.post(
            url,
            headers=headers(),
            json=payload or {},
            timeout=REQUEST_TIMEOUT,
        )

        print(f"[API] POST {url} -> {r.status_code}")

        try:
            data = r.json()
            print(data)
        except Exception:
            data = {"raw": r.text}

        return r.status_code, data

    except Exception as e:
        print(f"[API] POST ERROR {url}: {e}")
        return None, None


def get(url):
    try:
        r = requests.get(
            url,
            headers=headers(),
            timeout=REQUEST_TIMEOUT,
        )

        print(f"[API] GET {url} -> {r.status_code}")

        try:
            data = r.json()
            print(data)
        except Exception:
            data = {"raw": r.text}

        return r.status_code, data

    except Exception as e:
        print(f"[API] GET ERROR {url}: {e}")
        return None, None


# ============================================================
# NVIDIA-SMI
# ============================================================

def nvidia_smi():
    print()
    print("=" * 60)
    print("NVIDIA-SMI")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=20,
        )

        print(result.stdout)

        if result.stderr:
            print(result.stderr)

    except Exception as e:
        print("nvidia-smi error:", e)


# ============================================================
# CUDA TEST
# ============================================================

def cuda_test():
    print()
    print("=" * 60)
    print("CUDA TEST")
    print("=" * 60)

    try:
        import numba
        from numba import cuda

        available = cuda.is_available()

        print("CUDA available:", available)

        if not available:
            raise RuntimeError("CUDA is not available")

        device = cuda.get_current_device()

        print("GPU:", device.name)
        print("Compute capability:", device.compute_capability)

        # واقعی GPU kernel
        n = 1024 * 1024

        import numpy as np

        arr = np.ones(n, dtype=np.float32)

        d_arr = cuda.to_device(arr)

        @cuda.jit
        def gpu_kernel(x):
            i = cuda.grid(1)
            if i < x.size:
                x[i] = x[i] * 2.0

        threads = 256
        blocks = (n + threads - 1) // threads

        start = time.time()

        gpu_kernel[blocks, threads](d_arr)

        cuda.synchronize()

        elapsed = time.time() - start

        result = d_arr.copy_to_host()

        if not np.all(result == 2.0):
            raise RuntimeError("GPU result verification failed")

        print(f"GPU JOB SUCCESS: {n} elements")
        print(f"GPU kernel time: {elapsed:.4f}s")

        return True

    except Exception:
        print("CUDA TEST FAILED")
        traceback.print_exc()
        return False


# ============================================================
# WORKER READY
# ============================================================

def notify_worker_ready():
    print()
    print("=" * 60)
    print("NOTIFYING RAILWAY")
    print("=" * 60)

    url = worker_ready_url()

    for attempt in range(1, READY_RETRIES + 1):

        print(f"worker-ready attempt {attempt}/{READY_RETRIES}")

        status_code, data = post(
            url,
            {
                "session_id": SESSION_ID,
                "gpu": "Tesla P100-PCIE-16GB",
                "worker": "kaggle",
            },
        )

        if status_code == 200:

            print()
            print("WORKER READY accepted.")

            if data:
                print(data)

            return True

        if status_code == 401:
            print("ERROR: unauthorized")
            print("Check WORKER_TOKEN injection.")

        time.sleep(READY_RETRY_DELAY)

    return False


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():
    status_code, data = post(
        heartbeat_url(),
        {
            "session_id": SESSION_ID,
            "timestamp": time.time(),
        },
    )

    if status_code == 200:

        remaining = None

        if isinstance(data, dict):
            remaining = data.get("remaining_seconds")

        print(
            f"[KEEP-ALIVE] OK"
            + (
                f" remaining={remaining}s"
                if remaining is not None
                else ""
            )
        )

        return True

    print("[KEEP-ALIVE] FAILED")

    return False


# ============================================================
# COMMAND
# ============================================================

def get_command():

    status_code, data = get(command_url())

    if status_code != 200:
        print("[COMMAND] request failed")
        return None

    if not isinstance(data, dict):
        return None

    command = data.get("command")

    print("[COMMAND]", data)

    return command


# ============================================================
# COMMAND HANDLER
# ============================================================

def handle_command(command):

    if not command:
        return False

    print()
    print("=" * 60)
    print("COMMAND RECEIVED")
    print("=" * 60)

    print(command)

    command_type = None

    if isinstance(command, dict):
        command_type = command.get("type")

    elif isinstance(command, str):
        command_type = command

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if command_type in ("stop", "shutdown", "terminate"):

        print("STOP command received.")

        return True

    # --------------------------------------------------------
    # TEST GPU
    # --------------------------------------------------------

    if command_type in ("gpu_test", "test_gpu"):

        print("Running GPU test...")

        cuda_test()

        return False

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    print("Unknown command.")

    return False


# ============================================================
# WORKER LOOP
# ============================================================

def worker_loop():

    print()
    print("=" * 60)
    print("GPU WORKER IS ACTIVE")
    print("=" * 60)

    start_time = time.monotonic()

    last_heartbeat = 0
    last_command = 0

    while True:

        elapsed = int(time.monotonic() - start_time)

        # ----------------------------------------------------
        # TIME LIMIT
        # ----------------------------------------------------

        if elapsed >= TEST_SECONDS:

            print()
            print("=" * 60)
            print("TEST TIME FINISHED")
            print("=" * 60)

            print(f"Elapsed: {elapsed}s")
            print(f"Target : {TEST_SECONDS}s")

            break

        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if (
            last_heartbeat == 0
            or time.monotonic() - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            heartbeat()

            last_heartbeat = time.monotonic()

        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        if (
            last_command == 0
            or time.monotonic() - last_command
            >= COMMAND_INTERVAL
        ):

            command = get_command()

            last_command = time.monotonic()

            should_stop = handle_command(command)

            if should_stop:

                print("Worker stopping by command.")

                break

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        print(
            f"[WORKER] {elapsed}/{TEST_SECONDS}s"
        )

        # ----------------------------------------------------
        # SHORT SLEEP
        # ----------------------------------------------------

        time.sleep(3)

    print()
    print("=" * 60)
    print("GPU WORKER TEST FINISHED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("KAGGLE GPU WORKER")
    print("=" * 60)

    print("SESSION:", SESSION_ID)
    print("API:", API_URL)
    print("TOKEN:", len(WORKER_TOKEN), "chars")

    validate_config()

    # --------------------------------------------------------
    # GPU INFORMATION
    # --------------------------------------------------------

    nvidia_smi()

    # --------------------------------------------------------
    # REAL CUDA TEST
    # --------------------------------------------------------

    if not cuda_test():

        raise RuntimeError("CUDA test failed")

    # --------------------------------------------------------
    # RAILWAY READY
    # --------------------------------------------------------

    if not notify_worker_ready():

        raise RuntimeError(
            "Could not notify Railway"
        )

    # --------------------------------------------------------
    # 10 MINUTE WORKER
    # --------------------------------------------------------

    worker_loop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Worker interrupted.")

    except Exception as e:

        print()
        print("=" * 60)
        print("WORKER ERROR")
        print("=" * 60)

        print(repr(e))

        traceback.print_exc()

        raise
