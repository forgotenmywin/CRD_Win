import os
import sys
import time
import json
import signal
import traceback
import subprocess
import threading

import requests


# ============================================================
# RUNTIME CONFIG
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__".rstrip("/")
WORKER_TOKEN = "__WORKER_TOKEN__"


# ============================================================
# SETTINGS
# ============================================================

HEARTBEAT_INTERVAL = 5
COMMAND_INTERVAL = 3

# Workload size.
# P100 16GB can handle this comfortably.
MATRIX_SIZE = 4096

STOP_EVENT = threading.Event()


# ============================================================
# VALIDATION
# ============================================================

def validate_config():

    if not SESSION_ID or SESSION_ID.startswith("__"):
        raise RuntimeError("SESSION_ID was not injected")

    if not API_URL or API_URL.startswith("__"):
        raise RuntimeError("API_URL was not injected")

    if not WORKER_TOKEN or WORKER_TOKEN.startswith("__"):
        raise RuntimeError("WORKER_TOKEN was not injected")


# ============================================================
# API
# ============================================================

def headers():

    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "kaggle-gpu-worker/1.0",
    }


def api_post(path, payload=None):

    url = API_URL + path

    try:

        response = requests.post(
            url,
            headers=headers(),
            json=payload or {},
            timeout=15,
        )

        print(
            f"[API] POST {path} -> {response.status_code}",
            flush=True,
        )

        if response.text:
            print(response.text[:2000], flush=True)

        return response

    except Exception as e:

        print(
            f"[API] POST ERROR {path}: {e}",
            flush=True,
        )

        return None


def api_get(path):

    url = API_URL + path

    try:

        response = requests.get(
            url,
            headers=headers(),
            timeout=15,
        )

        print(
            f"[API] GET {path} -> {response.status_code}",
            flush=True,
        )

        if response.text:
            print(response.text[:2000], flush=True)

        return response

    except Exception as e:

        print(
            f"[API] GET ERROR {path}: {e}",
            flush=True,
        )

        return None


# ============================================================
# NVIDIA
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
            timeout=15,
        )

        print(result.stdout, flush=True)

        if result.stderr:
            print(result.stderr, flush=True)

    except Exception as e:

        print(f"nvidia-smi error: {e}", flush=True)


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

        print(
            "CUDA available:",
            cuda.is_available(),
            flush=True,
        )

        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")

        gpu = cuda.get_current_device()

        print(
            "GPU:",
            gpu.name,
            flush=True,
        )

        print(
            "Compute capability:",
            gpu.compute_capability,
            flush=True,
        )

        @cuda.jit
        def kernel(a, b, c):

            i = cuda.grid(1)

            if i < c.size:
                c[i] = a[i] + b[i]

        n = 1024 * 1024

        import numpy as np

        a = np.ones(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32)
        c = np.zeros(n, dtype=np.float32)

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dc = cuda.to_device(c)

        threads = 256
        blocks = (n + threads - 1) // threads

        start = time.time()

        kernel[blocks, threads](da, db, dc)

        cuda.synchronize()

        elapsed = time.time() - start

        print(
            f"GPU test OK: {n} elements in {elapsed:.4f}s",
            flush=True,
        )

        return True

    except Exception:

        traceback.print_exc()

        return False


# ============================================================
# CONTINUOUS GPU WORKLOAD
# ============================================================

def gpu_worker():

    """
    Keeps the P100 actively performing GPU computation.

    This is NOT just a one-time test.
    It continuously performs matrix multiplication until
    STOP_EVENT is set or the process is terminated.
    """

    print()
    print("=" * 60)
    print("CONTINUOUS GPU WORKLOAD")
    print("=" * 60)

    try:

        import torch

        print(
            "PyTorch:",
            torch.__version__,
            flush=True,
        )

        print(
            "CUDA:",
            torch.version.cuda,
            flush=True,
        )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "PyTorch CUDA is unavailable"
            )

        device = torch.device("cuda")

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
            flush=True,
        )

        print(
            "Matrix size:",
            MATRIX_SIZE,
            flush=True,
        )

        print(
            "Starting continuous GPU workload...",
            flush=True,
        )

        # Allocate tensors ON GPU.

        a = torch.randn(
            MATRIX_SIZE,
            MATRIX_SIZE,
            device=device,
            dtype=torch.float32,
        )

        b = torch.randn(
            MATRIX_SIZE,
            MATRIX_SIZE,
            device=device,
            dtype=torch.float32,
        )

        c = torch.empty_like(a)

        iteration = 0

        last_print = time.time()

        while not STOP_EVENT.is_set():

            # GPU computation.

            torch.mm(a, b, out=c)

            # Another operation keeps the GPU pipeline busy.

            torch.relu_(c)

            iteration += 1

            # Synchronize periodically so exceptions aren't hidden.

            if iteration % 10 == 0:

                torch.cuda.synchronize()

            now = time.time()

            if now - last_print >= 5:

                torch.cuda.synchronize()

                allocated = (
                    torch.cuda.memory_allocated()
                    / 1024
                    / 1024
                )

                reserved = (
                    torch.cuda.memory_reserved()
                    / 1024
                    / 1024
                )

                print(
                    f"[GPU] iteration={iteration} "
                    f"memory={allocated:.0f}MB "
                    f"reserved={reserved:.0f}MB",
                    flush=True,
                )

                last_print = now

        print(
            "GPU workload stopping...",
            flush=True,
        )

    except Exception as e:

        print(
            "GPU WORKLOAD ERROR:",
            repr(e),
            flush=True,
        )

        traceback.print_exc()

        STOP_EVENT.set()


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat_loop():

    count = 0

    while not STOP_EVENT.is_set():

        count += 1

        print(
            f"[HEARTBEAT] #{count}",
            flush=True,
        )

        response = api_post(
            f"/gpu/session/{SESSION_ID}/heartbeat"
        )

        if response is not None:

            try:

                data = response.json()

                status = data.get("status")

                print(
                    f"[HEARTBEAT] status={status}",
                    flush=True,
                )

                if status in (
                    "expired",
                    "stopped",
                    "error",
                ):

                    print(
                        "Session is no longer active.",
                        flush=True,
                    )

                    STOP_EVENT.set()
                    return

            except Exception:
                pass

        STOP_EVENT.wait(HEARTBEAT_INTERVAL)


# ============================================================
# COMMAND LOOP
# ============================================================

def command_loop():

    while not STOP_EVENT.is_set():

        response = api_get(
            f"/internal/session/{SESSION_ID}/command"
        )

        if response is not None:

            try:

                if response.status_code == 200:

                    data = response.json()

                    command = data.get("command")

                    if command:

                        print(
                            "[COMMAND]",
                            command,
                            flush=True,
                        )

                        if command in (
                            "stop",
                            "shutdown",
                            "terminate",
                        ):

                            STOP_EVENT.set()
                            return

            except Exception as e:

                print(
                    "[COMMAND] parse error:",
                    e,
                    flush=True,
                )

        STOP_EVENT.wait(COMMAND_INTERVAL)


# ============================================================
# SIGNALS
# ============================================================

def signal_handler(signum, frame):

    print(
        f"Received signal {signum}",
        flush=True,
    )

    STOP_EVENT.set()


signal.signal(
    signal.SIGTERM,
    signal_handler,
)

signal.signal(
    signal.SIGINT,
    signal_handler,
)


# ============================================================
# WORKER READY
# ============================================================

def notify_ready():

    print()
    print("=" * 60)
    print("NOTIFYING RAILWAY")
    print("=" * 60)

    for attempt in range(1, 11):

        print(
            f"worker-ready attempt {attempt}/10",
            flush=True,
        )

        response = api_post(
            f"/gpu/session/{SESSION_ID}/worker-ready"
        )

        if response is not None:

            if response.status_code == 200:

                print(
                    "WORKER READY accepted by Railway.",
                    flush=True,
                )

                return True

        time.sleep(3)

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("KAGGLE GPU WORKER")
    print("=" * 60)

    validate_config()

    print(
        f"SESSION : {SESSION_ID}",
        flush=True,
    )

    print(
        f"API     : {API_URL}",
        flush=True,
    )

    print(
        f"TOKEN   : {len(WORKER_TOKEN)} chars",
        flush=True,
    )

    # --------------------------------------------------------
    # GPU INFORMATION
    # --------------------------------------------------------

    nvidia_smi()

    # --------------------------------------------------------
    # CUDA TEST
    # --------------------------------------------------------

    if not cuda_test():

        raise RuntimeError(
            "CUDA test failed"
        )

    # --------------------------------------------------------
    # TELL RAILWAY WE ARE READY
    # --------------------------------------------------------

    if not notify_ready():

        raise RuntimeError(
            "Could not notify Railway"
        )

    # --------------------------------------------------------
    # START HEARTBEAT
    # --------------------------------------------------------

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        daemon=True,
    )

    heartbeat_thread.start()

    # --------------------------------------------------------
    # START COMMAND LOOP
    # --------------------------------------------------------

    command_thread = threading.Thread(
        target=command_loop,
        daemon=True,
    )

    command_thread.start()

    # --------------------------------------------------------
    # START REAL CONTINUOUS GPU WORK
    # --------------------------------------------------------

    gpu_thread = threading.Thread(
        target=gpu_worker,
        daemon=True,
    )

    gpu_thread.start()

    print()
    print("=" * 60)
    print("WORKER IS NOW RUNNING")
    print("CONTINUOUS GPU WORKLOAD ACTIVE")
    print("=" * 60)

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    while not STOP_EVENT.is_set():

        time.sleep(1)

    print()
    print("=" * 60)
    print("WORKER STOPPING")
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
            repr(e),
            flush=True,
        )

        traceback.print_exc()

        # Do not silently report success.

        sys.exit(1)
