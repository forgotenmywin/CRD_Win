import os
import sys
import time
import json
import signal
import traceback
import subprocess

import requests
from numba import cuda


# ============================================================
# CONFIG
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"

HEARTBEAT_INTERVAL = 10
STATUS_INTERVAL = 10

STOP = False


# ============================================================
# SIGNAL HANDLER
# ============================================================

def handle_signal(signum, frame):
    global STOP
    STOP = True
    print(f"[SIGNAL] Received signal {signum}", flush=True)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# ============================================================
# LOGGING
# ============================================================

def log(msg):
    print(msg, flush=True)


def section(title):
    log("")
    log("=" * 60)
    log(title)
    log("=" * 60)


# ============================================================
# HTTP
# ============================================================

def headers():
    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "kaggle-gpu-worker/1.0",
    }


def post(path, payload=None, timeout=10):
    url = API_URL.rstrip("/") + path

    try:
        r = requests.post(
            url,
            headers=headers(),
            json=payload or {},
            timeout=timeout,
        )

        log(
            f"[API] POST {path} -> {r.status_code}"
        )

        if r.text:
            log(r.text[:1000])

        return r

    except Exception as e:
        log(f"[API ERROR] POST {path}: {e}")
        return None


def get(path, timeout=10):
    url = API_URL.rstrip("/") + path

    try:
        r = requests.get(
            url,
            headers=headers(),
            timeout=timeout,
        )

        log(
            f"[API] GET {path} -> {r.status_code}"
        )

        if r.text:
            log(r.text[:1000])

        return r

    except Exception as e:
        log(f"[API ERROR] GET {path}: {e}")
        return None


# ============================================================
# GPU INFORMATION
# ============================================================

def nvidia_smi():
    section("NVIDIA-SMI")

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        print(result.stdout, flush=True)

        if result.stderr:
            print(result.stderr, flush=True)

    except Exception as e:
        log(f"nvidia-smi failed: {e}")


# ============================================================
# CUDA KERNEL
# ============================================================

@cuda.jit
def gpu_kernel(a, b, c):
    i = cuda.grid(1)

    if i < a.size:
        c[i] = a[i] * b[i] + 1.0


# ============================================================
# REAL GPU TEST
# ============================================================

def initialize_gpu():
    section("CUDA INITIALIZATION")

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = cuda.get_current_device()

    log(f"GPU: {device.name}")
    log(
        f"Compute capability: "
        f"{device.compute_capability}"
    )

    return device


def create_gpu_job():
    section("CREATING REAL GPU JOB")

    # Small enough to avoid excessive memory usage.
    # 4 million float32 values ≈ 16 MB per array.
    N = 4_000_000

    log(f"Elements: {N}")

    import numpy as np

    a = np.ones(N, dtype=np.float32)
    b = np.ones(N, dtype=np.float32)
    c = np.empty(N, dtype=np.float32)

    log("Copying data to GPU...")

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(c)

    threads = 256
    blocks = (N + threads - 1) // threads

    log(f"Threads per block: {threads}")
    log(f"Blocks: {blocks}")

    # First launch compiles the Numba CUDA kernel.
    log("Compiling / warming up CUDA kernel...")

    gpu_kernel[blocks, threads](
        d_a,
        d_b,
        d_c,
    )

    cuda.synchronize()

    log("GPU kernel warm-up OK")

    return d_a, d_b, d_c, blocks, threads


# ============================================================
# SESSION STATUS
# ============================================================

def get_session_status():
    path = (
        f"/gpu/session/"
        f"{SESSION_ID}"
    )

    r = get(path)

    if r is None:
        return None

    try:
        return r.json()
    except Exception:
        return None


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():
    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/heartbeat"
    )

    r = post(path)

    if r is None:
        return False

    return r.status_code == 200


# ============================================================
# WORKER READY
# ============================================================

def notify_ready():
    section("NOTIFYING RAILWAY")

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):

        log(
            f"worker-ready attempt "
            f"{attempt}/10"
        )

        r = post(path)

        if r is not None and r.status_code == 200:
            log("WORKER READY accepted by Railway.")
            return True

        time.sleep(3)

    raise RuntimeError(
        "Railway did not accept worker-ready"
    )


# ============================================================
# REAL GPU JOB
# ============================================================

def run_gpu_job():
    section("REAL GPU JOB")

    import time as pytime

    d_a, d_b, d_c, blocks, threads = create_gpu_job()

    start_time = pytime.time()
    last_heartbeat = 0
    last_status = 0
    iterations = 0

    log("")
    log("GPU JOB STARTED")
    log("The GPU will continuously execute CUDA work")
    log("until the Railway session expires.")
    log("")

    while not STOP:

        now = pytime.time()

        # ----------------------------------------------------
        # Check Railway status
        # ----------------------------------------------------

        if now - last_status >= STATUS_INTERVAL:

            status = get_session_status()
            last_status = now

            if status:

                current_status = status.get(
                    "status"
                )

                remaining = status.get(
                    "remaining_seconds"
                )

                log(
                    f"[SESSION] "
                    f"status={current_status} "
                    f"remaining={remaining}s "
                    f"iterations={iterations}"
                )

                if current_status in (
                    "expired",
                    "stopped",
                    "failed",
                    "error",
                ):
                    log(
                        "Session is no longer active."
                    )
                    break

                if (
                    remaining is not None
                    and remaining <= 0
                ):
                    log(
                        "Session time finished."
                    )
                    break

        # ----------------------------------------------------
        # Heartbeat
        # ----------------------------------------------------

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            ok = heartbeat()
            last_heartbeat = now

            if ok:
                log(
                    "[HEARTBEAT] OK"
                )
            else:
                log(
                    "[HEARTBEAT] FAILED"
                )

        # ----------------------------------------------------
        # REAL CUDA WORK
        # ----------------------------------------------------

        gpu_kernel[blocks, threads](
            d_a,
            d_b,
            d_c,
        )

        # This waits for the actual GPU calculation.
        cuda.synchronize()

        iterations += 1

        # Print progress every 25 iterations.
        if iterations % 25 == 0:

            elapsed = pytime.time() - start_time

            log(
                f"[GPU JOB] "
                f"iterations={iterations} "
                f"elapsed={elapsed:.1f}s"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    log("")
    log("GPU JOB STOPPED")

    try:
        del d_a
        del d_b
        del d_c
    except Exception:
        pass

    try:
        cuda.synchronize()
    except Exception:
        pass

    return iterations


# ============================================================
# MAIN
# ============================================================

def main():

    section("KAGGLE GPU WORKER")

    log(f"SESSION : {SESSION_ID}")
    log(f"API     : {API_URL}")
    log(
        f"TOKEN   : "
        f"{len(WORKER_TOKEN)} chars"
    )

    # --------------------------------------------------------
    # Validate generated configuration
    # --------------------------------------------------------

    if (
        not SESSION_ID
        or SESSION_ID.startswith("__")
    ):
        raise RuntimeError(
            "SESSION_ID was not injected"
        )

    if (
        not API_URL
        or API_URL.startswith("__")
    ):
        raise RuntimeError(
            "API_URL was not injected"
        )

    if (
        not WORKER_TOKEN
        or WORKER_TOKEN.startswith("__")
    ):
        raise RuntimeError(
            "WORKER_TOKEN was not injected"
        )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    nvidia_smi()

    initialize_gpu()

    # --------------------------------------------------------
    # Tell Railway we are ready
    # --------------------------------------------------------

    notify_ready()

    # --------------------------------------------------------
    # Real GPU workload
    # --------------------------------------------------------

    iterations = run_gpu_job()

    section("WORKER FINISHED")

    log(
        f"Total GPU iterations: {iterations}"
    )

    log(
        "Worker exited normally."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        log("Interrupted.")

    except Exception as e:

        section("WORKER ERROR")

        log(str(e))
        traceback.print_exc()

        sys.exit(1)
