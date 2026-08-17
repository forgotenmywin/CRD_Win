import time
import traceback
import requests


# ============================================================
# THESE ARE REPLACED BY GITHUB ACTIONS
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"

API_URL = API_URL.rstrip("/")

TEST_SECONDS = 120
HEARTBEAT_INTERVAL = 3
COMMAND_INTERVAL = 3


HEADERS = {
    "X-Worker-Token": WORKER_TOKEN,
    "Content-Type": "application/json",
}


# ============================================================
# VALIDATION
# ============================================================

def validate_config():

    if not SESSION_ID:
        raise RuntimeError(
            "SESSION_ID is empty"
        )

    if SESSION_ID.startswith("__"):
        raise RuntimeError(
            "SESSION_ID was not injected"
        )

    if not API_URL.startswith("http"):
        raise RuntimeError(
            f"Invalid API_URL: {API_URL}"
        )

    if not WORKER_TOKEN:
        raise RuntimeError(
            "WORKER_TOKEN is empty"
        )

    if WORKER_TOKEN.startswith("__"):
        raise RuntimeError(
            "WORKER_TOKEN was not injected"
        )


# ============================================================
# API
# ============================================================

def post(path):

    url = API_URL + path

    try:

        r = requests.post(
            url,
            headers=HEADERS,
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


def get(path):

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
# GPU JOB
# ============================================================

def gpu_job():

    print("=" * 60)
    print("CUDA TEST")
    print("=" * 60)

    from numba import cuda
    import numpy as np

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

    n = 1024 * 1024

    data = np.ones(
        n,
        dtype=np.float32
    )

    d_data = cuda.to_device(data)

    @cuda.jit
    def kernel(x):

        i = cuda.grid(1)

        if i < x.size:
            x[i] = x[i] * 2

    threads = 256
    blocks = (
        n + threads - 1
    ) // threads

    start = time.time()

    kernel[
        blocks,
        threads
    ](d_data)

    cuda.synchronize()

    elapsed = (
        time.time() - start
    )

    result = d_data.copy_to_host()

    if not np.allclose(
        result,
        2
    ):

        raise RuntimeError(
            "GPU result invalid"
        )

    print(
        f"GPU JOB SUCCESS: {n} elements"
    )

    print(
        f"GPU kernel time: {elapsed:.4f}s"
    )


# ============================================================
# READY
# ============================================================

def notify_ready():

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):

        print(
            f"worker-ready attempt "
            f"{attempt}/10"
        )

        r = post(path)

        if (
            r is not None
            and r.status_code == 200
        ):

            print(
                "WORKER READY accepted."
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

    r = post(path)

    if (
        r is not None
        and r.status_code == 200
    ):

        try:

            data = r.json()

            print(
                "[KEEP-ALIVE] "
                f"remaining="
                f"{data.get('remaining_seconds')}s"
            )

        except Exception:

            print(
                "[KEEP-ALIVE] OK"
            )

        return True

    return False


# ============================================================
# COMMAND
# ============================================================

def check_command():

    return get(
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )


# ============================================================
# MAIN
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

    validate_config()

    # --------------------------------------------------------
    # REAL GPU JOB
    # --------------------------------------------------------

    gpu_job()

    # --------------------------------------------------------
    # REGISTER WORKER
    # --------------------------------------------------------

    if not notify_ready():

        raise RuntimeError(
            "Could not notify Railway"
        )

    print("=" * 60)
    print("GPU WORKER IS ACTIVE")
    print("=" * 60)

    start = time.time()
    last_heartbeat = 0

    # --------------------------------------------------------
    # KEEP GPU JOB ALIVE
    # --------------------------------------------------------

    while True:

        elapsed = (
            time.time() - start
        )

        if elapsed >= TEST_SECONDS:
            break

        if (
            time.time()
            - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            heartbeat()

            last_heartbeat = time.time()

        command = check_command()

        if command:

            print(
                "[COMMAND]",
                command
            )

        print(
            f"[WORKER] "
            f"{int(elapsed)}/"
            f"{TEST_SECONDS}s"
        )

        time.sleep(
            COMMAND_INTERVAL
        )

    print("=" * 60)
    print("GPU WORKER TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        print("=" * 60)
        print("WORKER ERROR")
        print("=" * 60)

        print(repr(e))

        traceback.print_exc()

        raise
