import time
import json
import traceback
import subprocess
import requests


# ============================================================
# THESE VALUES ARE REPLACED BY GITHUB ACTIONS
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"

API_URL = API_URL.rstrip("/")

TEST_SECONDS = 120
HEARTBEAT_INTERVAL = 3
COMMAND_INTERVAL = 3


# ============================================================
# CONFIG CHECK
# ============================================================

def validate_config():

    if not SESSION_ID or SESSION_ID.startswith("__"):
        raise RuntimeError(
            f"SESSION_ID was not injected: {SESSION_ID!r}"
        )

    if not API_URL or API_URL.startswith("__"):
        raise RuntimeError(
            f"API_URL was not injected: {API_URL!r}"
        )

    if not WORKER_TOKEN or WORKER_TOKEN.startswith("__"):
        raise RuntimeError(
            "WORKER_TOKEN was not injected"
        )

    if not API_URL.startswith("http"):
        raise RuntimeError(
            f"Invalid API_URL: {API_URL!r}"
        )


# ============================================================
# HTTP
# ============================================================

def get_headers():

    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "kaggle-gpu-worker",
    }


def post(path, data=None):

    url = API_URL + path

    try:

        r = requests.post(
            url,
            json=data or {},
            headers=get_headers(),
            timeout=15,
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


def get(path):

    url = API_URL + path

    try:

        r = requests.get(
            url,
            headers=get_headers(),
            timeout=15,
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
# GPU INFORMATION
# ============================================================

def nvidia_smi():

    print()
    print("=" * 60)
    print("NVIDIA-SMI")
    print("=" * 60)

    try:

        p = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        print(p.stdout, flush=True)

    except Exception as e:

        print(
            "nvidia-smi failed:",
            e,
            flush=True,
        )


# ============================================================
# CUDA REAL TEST
# ============================================================

def cuda_test():

    print()
    print("=" * 60)
    print("CUDA TEST")
    print("=" * 60)

    try:

        import numpy as np
        from numba import cuda

        available = cuda.is_available()

        print(
            "CUDA available:",
            available,
            flush=True,
        )

        if not available:
            raise RuntimeError(
                "CUDA is not available"
            )

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

        da = cuda.to_device(a)
        db = cuda.to_device(b)
        dc = cuda.to_device(c)

        @cuda.jit
        def kernel(a, b, c):

            i = cuda.grid(1)

            if i < a.size:
                c[i] = a[i] + b[i]

        threads = 256
        blocks = (n + threads - 1) // threads

        start = time.time()

        kernel[
            blocks,
            threads
        ](
            da,
            db,
            dc,
        )

        cuda.synchronize()

        elapsed = time.time() - start

        result = dc.copy_to_host()

        if not np.allclose(
            result,
            2.0,
        ):
            raise RuntimeError(
                "GPU calculation returned wrong result"
            )

        print(
            f"GPU kernel OK: {n} elements "
            f"in {elapsed:.4f}s",
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

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):

        print(
            f"worker-ready attempt "
            f"{attempt}/10",
            flush=True,
        )

        r = post(
            path,
            {
                "gpu": "Tesla P100-PCIE-16GB",
                "compute_capability": [6, 0],
                "cuda_available": True,
            },
        )

        if r is not None:

            if r.status_code == 200:

                print(
                    "WORKER READY accepted by Railway.",
                    flush=True,
                )

                return True

            if r.status_code == 401:

                print(
                    "ERROR: WORKER_TOKEN rejected.",
                    flush=True,
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

    r = post(path)

    return (
        r is not None
        and r.status_code == 200
    )


# ============================================================
# COMMAND
# ============================================================

def get_command():

    path = (
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )

    r = get(path)

    if r is None:
        return None

    if r.status_code != 200:
        return None

    try:
        return r.json()

    except Exception:
        return None


# ============================================================
# COMMAND EXECUTION
# ============================================================

def execute_command(command):

    print()
    print("=" * 60)
    print("COMMAND RECEIVED")
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
    # GPU TEST
    # --------------------------------------------------------

    if command_type == "gpu_test":

        success = cuda_test()

        post(
            f"/internal/session/"
            f"{SESSION_ID}/result",
            {
                "command_id": command_id,
                "success": success,
                "type": "gpu_test",
            },
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

        post(
            f"/internal/session/"
            f"{SESSION_ID}/result",
            result,
        )


# ============================================================
# REAL WORKER LOOP
# ============================================================

def worker_loop():

    print()
    print("=" * 60)
    print("GPU WORKER ACTIVE")
    print("=" * 60)

    start = time.time()

    end = (
        start +
        TEST_SECONDS
    )

    last_heartbeat = 0

    while True:

        now = time.time()

        remaining = int(
            max(
                0,
                end - now,
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

        if (
            now - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

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

        time.sleep(
            COMMAND_INTERVAL
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("KAGGLE GPU WORKER")
    print("=" * 60)

    validate_config()

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

        raise
