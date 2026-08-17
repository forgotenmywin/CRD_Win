import os
import sys
import time
import json
import traceback
import subprocess
import requests


# ============================================================
# CONFIG
# ============================================================

SESSION_ID = os.environ.get("SESSION_ID", "").strip()
API_URL = os.environ.get("API_URL", "").strip().rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "").strip()

HEARTBEAT_INTERVAL = 15
COMMAND_INTERVAL = 3
REQUEST_TIMEOUT = 20


# ============================================================
# VALIDATION
# ============================================================

if not SESSION_ID:
    raise RuntimeError("SESSION_ID is missing")

if not API_URL:
    raise RuntimeError("API_URL is missing")

if not WORKER_TOKEN:
    raise RuntimeError("WORKER_TOKEN is missing")


print("=" * 60)
print("KAGGLE GPU WORKER")
print("=" * 60)
print(f"SESSION : {SESSION_ID}")
print(f"API     : {API_URL}")
print(f"TOKEN   : {len(WORKER_TOKEN)}")
print("=" * 60)


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "X-Worker-Token": WORKER_TOKEN,
    "Authorization": f"Bearer {WORKER_TOKEN}",
    "Content-Type": "application/json",
}


def api_get(path):
    url = API_URL + path

    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        print(f"[API] GET {path} → {r.status_code}")

        if r.status_code != 200:
            print(r.text[:1000])

        return r

    except Exception as e:
        print(f"[API] GET ERROR {path}: {e}")
        return None


def api_post(path, data=None):
    url = API_URL + path

    try:
        r = requests.post(
            url,
            headers=HEADERS,
            json=data or {},
            timeout=REQUEST_TIMEOUT,
        )

        print(f"[API] POST {path} → {r.status_code}")

        if r.status_code >= 400:
            print(r.text[:1000])

        return r

    except Exception as e:
        print(f"[API] POST ERROR {path}: {e}")
        return None


# ============================================================
# GPU INFORMATION
# ============================================================

def print_gpu_info():

    print()
    print("=" * 60)
    print("NVIDIA SMI")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=20,
        )

        print(result.stdout)

        if result.returncode != 0:
            print(result.stderr)

    except Exception as e:
        print("nvidia-smi failed:", e)


# ============================================================
# CUDA TEST
# ============================================================

def cuda_test():

    print()
    print("=" * 60)
    print("CUDA TEST")
    print("=" * 60)

    try:
        import torch

        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())

        if not torch.cuda.is_available():
            print("CUDA is not available")
            return False

        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)

        print("GPU:", gpu_name)
        print("Compute capability:", capability)

        # Small test that works on P100
        x = torch.randn(
            1024,
            1024,
            device="cuda",
        )

        y = torch.matmul(x, x)

        torch.cuda.synchronize()

        print("GPU kernel OK")
        print("Elements:", y.numel())

        del x
        del y

        torch.cuda.empty_cache()

        return True

    except Exception:
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

    path = f"/gpu/session/{SESSION_ID}/worker-ready"

    for attempt in range(1, 11):

        print(f"worker-ready attempt={attempt}")

        response = api_post(
            path,
            {
                "session_id": SESSION_ID,
                "gpu": "Tesla P100-PCIE-16GB",
                "cuda_available": True,
            },
        )

        if response is not None and response.status_code == 200:
            print("WORKER READY accepted by Railway.")
            return True

        time.sleep(5)

    print("WORKER READY failed.")
    return False


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():

    path = f"/gpu/session/{SESSION_ID}/heartbeat"

    response = api_post(
        path,
        {
            "session_id": SESSION_ID,
            "timestamp": time.time(),
        },
    )

    return response is not None and response.status_code == 200


# ============================================================
# COMMAND
# ============================================================

def get_command():

    path = f"/internal/session/{SESSION_ID}/command"

    response = api_get(path)

    if response is None:
        return None

    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except Exception:
        return None

    if not data:
        return None

    # Different possible API formats
    if isinstance(data, dict):

        if "command" in data:
            return data

        if data.get("status") == "empty":
            return None

        if data.get("status") == "stopped":
            return {
                "command": "__STOP__"
            }

    return None


# ============================================================
# COMMAND EXECUTION
# ============================================================

def execute_command(command):

    command_id = command.get("id")
    cmd = command.get("command")

    if not cmd:
        return {
            "success": False,
            "error": "Empty command",
        }

    if cmd == "__STOP__":
        return {
            "success": True,
            "stopped": True,
        }

    print()
    print("=" * 60)
    print("EXECUTING COMMAND")
    print("=" * 60)
    print(cmd)
    print("=" * 60)

    started = time.time()

    try:

        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=900,
        )

        elapsed = time.time() - started

        result = {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout[-20000:],
            "stderr": process.stderr[-20000:],
            "elapsed": elapsed,
        }

    except subprocess.TimeoutExpired:

        result = {
            "success": False,
            "error": "Command timeout",
        }

    except Exception as e:

        result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # Send result
    path = f"/internal/session/{SESSION_ID}/result"

    payload = {
        "command_id": command_id,
        "result": result,
    }

    api_post(path, payload)

    return result


# ============================================================
# KEEP ALIVE LOOP
# ============================================================

def worker_loop():

    print()
    print("=" * 60)
    print("COMMAND LOOP / KEEP-ALIVE")
    print("=" * 60)

    last_heartbeat = 0

    while True:

        now = time.time()

        # -----------------------------------------
        # HEARTBEAT
        # -----------------------------------------

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            ok = heartbeat()

            if ok:
                print("[KEEP-ALIVE] heartbeat OK")
            else:
                print("[KEEP-ALIVE] heartbeat failed")

            last_heartbeat = now

        # -----------------------------------------
        # COMMAND
        # -----------------------------------------

        command = get_command()

        if command:

            print("[COMMAND] received")

            result = execute_command(command)

            if result.get("stopped"):
                print("STOP command received.")
                break

        time.sleep(COMMAND_INTERVAL)


# ============================================================
# MAIN
# ============================================================

def main():

    print_gpu_info()

    cuda_ok = cuda_test()

    if not cuda_ok:
        raise RuntimeError("CUDA test failed")

    if not notify_worker_ready():
        raise RuntimeError("Could not notify Railway that worker is ready")

    # Initial heartbeat
    heartbeat()

    # Infinite worker
    worker_loop()

    print("Worker stopped.")


if __name__ == "__main__":
    main()
