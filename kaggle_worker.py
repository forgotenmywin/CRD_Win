import os
import sys
import time
import json
import traceback
import subprocess
import threading

import requests


# ============================================================
# INJECTED CONFIG
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"

API_URL = API_URL.rstrip("/")

# Test duration
TEST_SECONDS = 120

# Heartbeat interval
HEARTBEAT_INTERVAL = 3

# Command polling interval
COMMAND_INTERVAL = 3


# ============================================================
# VALIDATION
# ============================================================

def validate_config():
    if SESSION_ID == "__SESSION_ID__":
        raise RuntimeError("SESSION_ID was not injected")

    if API_URL == "__API_URL__":
        raise RuntimeError("API_URL was not injected")

    if WORKER_TOKEN == "__WORKER_TOKEN__":
        raise RuntimeError("WORKER_TOKEN was not injected")

    if not SESSION_ID:
        raise RuntimeError("SESSION_ID is empty")

    if not API_URL:
        raise RuntimeError("API_URL is empty")

    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN is empty")

    if not API_URL.startswith("http://") and not API_URL.startswith("https://"):
        raise RuntimeError(f"Invalid API_URL: {API_URL}")


# ============================================================
# API
# ============================================================

def headers():
    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "kaggle-gpu-worker",
    }


def api_post(path, payload=None, timeout=15):
    url = f"{API_URL}{path}"

    try:
        r = requests.post(
            url,
            json=payload or {},
            headers=headers(),
            timeout=timeout,
        )

        print(
            f"[API] POST {path} -> {r.status_code}",
            flush=True,
        )

        if r.text:
            print(r.text[:2000], flush=True)

        return r

    except Exception as e:
        print(
            f"[API] POST ERROR {path}: {e}",
            flush=True,
        )

        return None


def api_get(path, timeout=15):
    url = f"{API_URL}{path}"

    try:
        r = requests.get(
            url,
            headers=headers(),
            timeout=timeout,
        )

        print(
            f"[API] GET {path} -> {r.status_code}",
            flush=True,
        )

        if r.text:
            print(r.text[:2000], flush=True)

        return r

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
        print("nvidia-smi error:", e, flush=True)


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

        device = cuda.get_current_device()

        print(
            "GPU:",
            device.name,
            flush=True,
        )

        print(
            "Compute capability:",
            device.compute_capability,
            flush=True,
        )

        import numpy as np

        n = 1024 * 1024

        a = np.ones(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32)
        c = np.zeros(n, dtype=np.float32)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.to_device(c)

        @cuda.jit
        def add_kernel(a, b, c):
            i = cuda.grid(1)

            if i < a.size:
                c[i] = a[i] + b[i]

        threads = 256
        blocks = (n + threads - 1) // threads

        start = time.time()

        add_kernel[blocks, threads](
            d_a,
            d_b,
            d_c,
        )

        cuda.synchronize()

        elapsed = time.time() - start

        result = d_c.copy_to_host()

        if not np.allclose(result, 2.0):
            raise RuntimeError("CUDA result is incorrect")

        print(
            f"GPU kernel OK: {n} elements in {elapsed:.4f}s",
            flush=True,
        )

        return True

    except Exception:
        traceback.print_exc()
        return False


# ============================================================
# WORKER READY
# ============================================================

def notify_ready():
    print()
    print("=" * 60)
    print("NOTIFYING RAILWAY")
    print("=" * 60)

    endpoint = f"/gpu/session/{SESSION_ID}/worker-ready"

    for attempt in range(1, 11):

        print(
            f"worker-ready attempt {attempt}/10",
            flush=True,
        )

        response = api_post(
            endpoint,
            {
                "gpu": "Tesla P100-PCIE-16GB",
                "compute_capability": [6, 0],
                "cuda_available": True,
            },
        )

        if response is not None:

            if response.status_code == 200:
                print(
                    "WORKER READY accepted by Railway.",
                    flush=True,
                )

                return True

            if response.status_code == 401:
                print(
                    "ERROR: Railway rejected WORKER_TOKEN.",
                    flush=True,
                )

        time.sleep(3)

    return False


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():
    endpoint = f"/gpu/session/{SESSION_ID}/heartbeat"

    response = api_post(endpoint)

    return (
        response is not None
        and response.status_code == 200
    )


# ============================================================
# COMMAND
# ============================================================

def get_command():
    endpoint = (
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )

    response = api_get(endpoint)

    if response is None:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()

        if not data:
            return None

        return data

    except Exception:
        return None


# ============================================================
# COMMAND EXECUTION
# ============================================================

def execute_command(command):
    print()
    print("=" * 60)
    print("GPU COMMAND")
    print("=" * 60)

    print(
        json.dumps(
            command,
            indent=2,
        ),
        flush=True,
    )

    command_id = command.get("id")
    command_type = command.get("type")

    # --------------------------------------------------------
    # TEST GPU
    # --------------------------------------------------------

    if command_type == "gpu_test":

        ok = cuda_test()

        result = {
            "command_id": command_id,
            "success": ok,
            "type": "gpu_test",
        }

        api_post(
            f"/internal/session/{SESSION_ID}/result",
            result,
        )

        return

    # --------------------------------------------------------
    # SHELL
    # --------------------------------------------------------

    if command_type == "shell":

        cmd = command.get("command")

        if not cmd:
            return

        try:
            p = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )

            result = {
                "command_id": command_id,
                "success": p.returncode == 0,
                "returncode": p.returncode,
                "stdout": p.stdout[-10000:],
                "stderr": p.stderr[-10000:],
            }

        except Exception as e:

            result = {
                "command_id": command_id,
                "success": False,
                "error": str(e),
            }

        api_post(
            f"/internal/session/{SESSION_ID}/result",
            result,
        )

        return


# ============================================================
# MAIN WORKER LOOP
# ============================================================

def worker_loop():
    print()
    print("=" * 60)
    print("GPU WORKER ACTIVE")
    print("=" * 60)

    start_time = time.time()
    end_time = start_time + TEST_SECONDS

    last_heartbeat = 0

    while True:

        now = time.time()

        remaining = int(
            max(
                0,
                end_time - now,
            )
        )

        if remaining <= 0:
            print()
            print("=" * 60)
            print("120 SECOND TEST FINISHED")
            print("=" * 60)
            break

        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            ok = heartbeat()

            print(
                f"[KEEP-ALIVE] "
                f"{'OK' if ok else 'FAILED'} "
                f"remaining={remaining}s",
                flush=True,
            )

            last_heartbeat = now

        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        command = get_command()

        if command:

            try:
                execute_command(command)

            except Exception:
                traceback.print_exc()

        time.sleep(COMMAND_INTERVAL)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("KAGGLE GPU WORKER")
    print("=" * 60)

    print(
        "SESSION :",
        SESSION_ID,
        flush=True,
    )

    print(
        "API     :",
        API_URL,
        flush=True,
    )

    print(
        "TOKEN   :",
        len(WORKER_TOKEN),
        "chars",
        flush=True,
    )

    validate_config()

    nvidia_smi()

    if not cuda_test():
        raise RuntimeError(
            "CUDA test failed"
        )

    if not notify_ready():
        raise RuntimeError(
            "Could not notify Railway"
        )

    print()
    print("=" * 60)
    print("WORKER READY")
    print("=" * 60)

    worker_loop()

    print()
    print("=" * 60)
    print("WORKER FINISHED")
    print("=" * 60)


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

        sys.exit(1)
