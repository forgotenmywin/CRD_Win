import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SESSION_ID = os.environ.get("SESSION_ID", "")
API_URL = os.environ.get("API_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")


# ============================================================
# VALIDATION
# ============================================================

def fail(message):
    print("")
    print("=" * 60)
    print("ERROR")
    print("=" * 60)
    print(message)
    sys.exit(1)


if not SESSION_ID:
    fail("SESSION_ID is missing")

if not API_URL:
    fail("API_URL is missing")

if not WORKER_TOKEN:
    fail("WORKER_TOKEN is missing")

if not KAGGLE_USERNAME:
    fail("KAGGLE_USERNAME is missing")


print("=" * 60)
print("BUILDING KAGGLE GPU WORKER")
print("=" * 60)

print(f"Session injected: {SESSION_ID}")
print(f"API injected: {API_URL}")
print("Token injected: YES")


# ============================================================
# DIRECTORIES
# ============================================================

BASE = Path("kaggle_upload")

if BASE.exists():
    shutil.rmtree(BASE)

BASE.mkdir(parents=True)


# ============================================================
# WORKER SCRIPT
# ============================================================

worker_code = r'''
import os
import sys
import time
import json
import subprocess
import traceback
import requests


# ============================================================
# RUNTIME CONFIG
# ============================================================

SESSION_ID = os.environ.get("SESSION_ID", "")
API_URL = os.environ.get("API_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

TEST_DURATION = 120


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message, flush=True)


def section(title):
    log("")
    log("=" * 60)
    log(title)
    log("=" * 60)


# ============================================================
# VALIDATE
# ============================================================

if not SESSION_ID:
    raise RuntimeError("SESSION_ID was not injected")

if not API_URL:
    raise RuntimeError("API_URL was not injected")

if not WORKER_TOKEN:
    raise RuntimeError("WORKER_TOKEN was not injected")


section("KAGGLE GPU WORKER - 120 SECOND TEST")

log(f"SESSION : {SESSION_ID}")
log(f"API     : {API_URL}")
log(f"TOKEN   : {len(WORKER_TOKEN)} chars")


# ============================================================
# API
# ============================================================

def api_headers():
    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "X-Worker-Token": WORKER_TOKEN,
        "Content-Type": "application/json",
    }


def post(path, data=None, timeout=10):

    url = API_URL + path

    try:
        r = requests.post(
            url,
            headers=api_headers(),
            json=data or {},
            timeout=timeout
        )

        log(
            f"[API] POST {path} -> {r.status_code}"
        )

        if r.text:
            log(r.text[:1000])

        return r

    except Exception as e:
        log(f"[API ERROR] {e}")
        return None


def get(path, timeout=10):

    url = API_URL + path

    try:
        r = requests.get(
            url,
            headers=api_headers(),
            timeout=timeout
        )

        log(
            f"[API] GET {path} -> {r.status_code}"
        )

        return r

    except Exception as e:
        log(f"[API ERROR] {e}")
        return None


# ============================================================
# NVIDIA
# ============================================================

section("NVIDIA-SMI")

try:

    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=15
    )

    log(result.stdout)

except Exception as e:
    log(f"nvidia-smi failed: {e}")


# ============================================================
# GPU INFORMATION
# ============================================================

gpu_name = "Unknown"
gpu_memory = 0

try:

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader"
        ],
        capture_output=True,
        text=True,
        timeout=10
    )

    line = result.stdout.strip()

    if line:
        parts = line.split(",")

        gpu_name = parts[0].strip()

        if len(parts) > 1:
            gpu_memory = parts[1].strip()

except Exception as e:
    log(f"GPU query failed: {e}")


log(f"GPU: {gpu_name}, {gpu_memory}")


# ============================================================
# CUDA / NUMBA TEST
# ============================================================

section("CUDA / NUMBA TEST")

cuda_ok = False
compute_capability = None

try:

    import numba
    from numba import cuda

    cuda_ok = cuda.is_available()

    log(f"CUDA available: {cuda_ok}")

    if cuda_ok:

        device = cuda.get_current_device()

        compute_capability = (
            device.compute_capability
        )

        log(f"GPU: {device.name}")
        log(
            f"Compute capability: "
            f"{compute_capability}"
        )

        @cuda.jit
        def test_kernel(a):

            i = cuda.grid(1)

            if i < a.size:
                a[i] += 1

        import numpy as np

        n = 1024 * 1024

        data = np.zeros(n, dtype=np.float32)

        d_data = cuda.to_device(data)

        threads = 256
        blocks = (n + threads - 1) // threads

        start = time.time()

        test_kernel[blocks, threads](d_data)

        cuda.synchronize()

        elapsed = time.time() - start

        log(
            f"GPU kernel OK: "
            f"{n} elements in "
            f"{elapsed:.4f}s"
        )

except Exception as e:

    log("CUDA TEST FAILED")
    traceback.print_exc()


# ============================================================
# WORKER READY
# ============================================================

section("NOTIFYING RAILWAY")

ready = False

for attempt in range(1, 11):

    log(
        f"worker-ready attempt "
        f"{attempt}/10"
    )

    response = post(
        f"/gpu/session/{SESSION_ID}/worker-ready",
        {
            "gpu": f"{gpu_name}, {gpu_memory}",
            "cuda_available": cuda_ok,
            "compute_capability": (
                list(compute_capability)
                if compute_capability
                else None
            )
        }
    )

    if response is not None:

        if response.status_code == 200:

            log(
                "WORKER READY accepted by Railway."
            )

            ready = True
            break

        elif response.status_code == 401:

            log(
                "ERROR: Worker authentication failed."
            )

    time.sleep(3)


if not ready:

    raise RuntimeError(
        "Railway did not accept worker-ready"
    )


# ============================================================
# 120 SECOND WORKER
# ============================================================

section("120 SECOND WORKER TEST")

start_time = time.time()

heartbeat_count = 0

while True:

    elapsed = time.time() - start_time

    if elapsed >= TEST_DURATION:
        break

    remaining = int(
        TEST_DURATION - elapsed
    )

    # --------------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------------

    heartbeat_count += 1

    response = post(
        f"/gpu/session/{SESSION_ID}/heartbeat",
        {
            "remaining_seconds": remaining,
            "worker_alive": True
        }
    )

    if response is not None:

        if response.status_code == 200:
            log(
                f"[KEEP-ALIVE] OK "
                f"remaining={remaining}s"
            )

        elif response.status_code == 401:

            log(
                "[KEEP-ALIVE] Unauthorized"
            )

    # --------------------------------------------------------
    # COMMAND
    # --------------------------------------------------------

    response = get(
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )

    if response is not None:

        if response.status_code == 200:

            try:

                command_data = response.json()

                command = command_data.get(
                    "command"
                )

                if command:

                    log(
                        f"COMMAND RECEIVED: "
                        f"{command}"
                    )

                    if command == "stop":

                        log(
                            "STOP command received."
                        )

                        break

            except Exception:
                pass

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    time.sleep(3)


# ============================================================
# COMPLETE
# ============================================================

section("WORKER TEST FINISHED")

post(
    f"/gpu/session/{SESSION_ID}/heartbeat",
    {
        "worker_alive": False,
        "completed": True
    }
)

log("120 second GPU worker test completed.")
'''


# ============================================================
# INJECT RUNTIME CONFIG
# ============================================================

worker_code = worker_code.replace(
    'SESSION_ID = os.environ.get("SESSION_ID", "")',
    f'SESSION_ID = {SESSION_ID!r}',
    1
)

worker_code = worker_code.replace(
    'API_URL = os.environ.get("API_URL", "").rstrip("/")',
    f'API_URL = {API_URL!r}',
    1
)

worker_code = worker_code.replace(
    'WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")',
    f'WORKER_TOKEN = {WORKER_TOKEN!r}',
    1
)


# ============================================================
# SAFETY CHECK
# ============================================================

if "__SESSION_ID__" in worker_code:
    fail("__SESSION_ID__ remains")

if "__API_URL__" in worker_code:
    fail("__API_URL__ remains")

if "__WORKER_TOKEN__" in worker_code:
    fail("__WORKER_TOKEN__ remains")


# ============================================================
# WRITE SCRIPT
# ============================================================

script_path = BASE / "script.py"

script_path.write_text(
    worker_code,
    encoding="utf-8"
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "id": f"{KAGGLE_USERNAME}/gpu-session-{SESSION_ID}",
    "title": f"GPU Session {SESSION_ID}",
    "code_file": "script.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True
}

metadata_path = BASE / "kernel-metadata.json"

metadata_path.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# SHOW FILES
# ============================================================

print("")
print("=" * 60)
print("KAGGLE FILES")
print("=" * 60)

for p in BASE.iterdir():
    print(p)


print("")
print("=" * 60)
print("KAGGLE METADATA")
print("=" * 60)

print(
    metadata_path.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# PUSH TO KAGGLE
# ============================================================

print("")
print("=" * 60)
print("PUSHING KAGGLE GPU WORKER")
print("=" * 60)

command = [
    "kaggle",
    "kernels",
    "push",
    "-p",
    str(BASE)
]

print(
    "Running:",
    " ".join(command)
)

result = subprocess.run(
    command,
    text=True
)

if result.returncode != 0:

    fail(
        f"Kaggle push failed "
        f"with exit code "
        f"{result.returncode}"
    )


print("")
print("=" * 60)
print("KAGGLE GPU WORKER STARTED")
print("=" * 60)

print(
    f"Session: {SESSION_ID}"
)

print(
    "The Kaggle GPU worker should now start."
)
