import time
import traceback
import requests

# ============================================================
# THESE VALUES ARE INJECTED BY GITHUB ACTIONS
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"

API_URL = API_URL.rstrip("/")

HEARTBEAT_INTERVAL = 3
COMMAND_INTERVAL = 3

# فقط برای تست
TEST_SECONDS = 120


# ============================================================
# HEADERS
# ============================================================

HEADERS = {
    "X-Worker-Token": WORKER_TOKEN,
    "Content-Type": "application/json",
}


# ============================================================
# VALIDATION
# ============================================================

def validate_config():

    if not SESSION_ID:
        raise RuntimeError("SESSION_ID is empty")

    if SESSION_ID.startswith("__"):
        raise RuntimeError(
            f"SESSION_ID was not injected: {SESSION_ID}"
        )

    if not API_URL:
        raise RuntimeError("API_URL is empty")

    if not API_URL.startswith("http"):
        raise RuntimeError(
            f"Invalid API_URL: {API_URL}"
        )

    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN is empty")


# ============================================================
# API
# ============================================================

def api_post(path, data=None):

    url = API_URL + path

    try:

        r = requests.post(
            url,
            headers=HEADERS,
            json=data or {},
            timeout=15,
        )

        print(
            f"[API] POST {path} -> {r.status_code}"
        )

        try:
            print(r.json())
        except Exception:
            print(r.text)

        return r

    except Exception as e:

        print(
            f"[API] POST ERROR {path}: {e}"
        )

        return None


def api_get(path):

    url = API_URL + path

    try:

        r = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
        )

        print(
            f"[API] GET {path} -> {r.status_code}"
        )

        try:
            return r.json()
        except Exception:
            return {}

    except Exception as e:

        print(
            f"[API] GET ERROR {path}: {e}"
        )

        return {}


# ============================================================
# GPU TEST
# ============================================================

def gpu_test():

    print("=" * 60)
    print("CUDA TEST")
    print("=" * 60)

    try:

        from numba import cuda

        if not cuda.is_available():
            raise RuntimeError(
                "CUDA is not available"
            )

        device = cuda.get_current_device()

        print(
            "GPU:",
            device.name
        )

        print(
            "Compute capability:",
            device.compute_capability
        )

        import numpy as np

        n = 1024 * 1024

        data = np.ones(n, dtype=np.float32)

        d_data = cuda.to_device(data)

        @cuda.jit
        def kernel(x):

            i = cuda.grid(1)

            if i < x.size:
                x[i] = x[i] * 2.0

        threads = 256
        blocks = (n + threads - 1) // threads

        start = time.time()

        kernel[blocks, threads](d_data)

        cuda.synchronize()

        elapsed = time.time() - start

        result = d_data.copy_to_host()

        if not np.allclose(result, 2.0):
            raise RuntimeError(
                "GPU calculation produced invalid result"
            )

        print(
            f"GPU JOB SUCCESS: {n} elements"
        )

        print(
            f"GPU kernel time: {elapsed:.4f}s"
        )

        return True

    except Exception:

        traceback.print_exc()

        return False


# ============================================================
# WORKER READY
# ============================================================

def notify_ready():

    print("=" * 60)
    print("NOTIFYING RAILWAY")
    print("=" * 60)

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):

        print(
            f"worker-ready attempt "
            f"{attempt}/10"
        )

        r = api_post(path)

        if r is not None and r.status_code == 200:

            print(
                "WORKER READY accepted by Railway."
            )

            return True

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

    r = api_post(path)

    if r is not None and r.status_code == 200:

        try:

            data = r.json()

            remaining = data.get(
                "remaining_seconds"
            )

            if remaining is not None:
                print(
                    f"[KEEP-ALIVE] "
                    f"remaining={remaining}s"
                )
            else:
                print(
                    "[KEEP-ALIVE] OK"
                )

        except Exception:

            print(
                "[KEEP-ALIVE] OK"
            )

        return True

    print(
        "[KEEP-ALIVE] FAILED"
    )

    return False


# ============================================================
# COMMAND
# ============================================================

def check_command():

    path = (
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )

    return api_get(path)


# ============================================================
# MAIN WORKER LOOP
# ============================================================

def main():

    print("=" * 60)
    print("KAGGLE GPU WORKER")
    print("=" * 60)

    print(
        "SESSION:",
        SESSION_ID
    )

    print(
        "API:",
        API_URL
    )

    print(
        "TOKEN:",
        len(WORKER_TOKEN),
        "chars"
    )

    print()

    validate_config()

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    success = gpu_test()

    if not success:

        raise RuntimeError(
            "GPU test failed"
        )

    # --------------------------------------------------------
    # READY
    # --------------------------------------------------------

    if not notify_ready():

        raise RuntimeError(
            "Could not notify Railway"
        )

    print()

    print("=" * 60)
    print("GPU WORKER IS ACTIVE")
    print("=" * 60)

    print(
        f"Running for {TEST_SECONDS} seconds..."
    )

    start = time.time()

    last_heartbeat = 0

    # --------------------------------------------------------
    # KEEP WORKER ALIVE
    # --------------------------------------------------------

    while True:

        elapsed = time.time() - start

        if elapsed >= TEST_SECONDS:

            break

        # heartbeat
        if (
            time.time() - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            heartbeat()

            last_heartbeat = time.time()

        # command polling
        try:

            command = check_command()

            if command:

                print(
                    "[COMMAND]",
                    command
                )

        except Exception as e:

            print(
                "[COMMAND ERROR]",
                e
            )

        print(
            f"[WORKER] "
            f"{int(elapsed)}/{TEST_SECONDS}s"
        )

        time.sleep(COMMAND_INTERVAL)

    print()

    print("=" * 60)
    print("GPU WORKER TEST FINISHED")
    print("=" * 60)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print()
        print("=" * 60)
        print("WORKER ERROR")
        print("=" * 60)

        print(
            repr(e)
        )

        traceback.print_exc()

        raise
