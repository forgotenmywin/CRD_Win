import os
import sys
import time
import json
import subprocess
import traceback

import requests


# ============================================================
# VALUES ARE REPLACED BY GITHUB ACTIONS
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"


# ============================================================
# CONFIG
# ============================================================

HEARTBEAT_INTERVAL = 15
COMMAND_INTERVAL = 3
REQUEST_TIMEOUT = 20


# ============================================================
# VALIDATION
# ============================================================

if not SESSION_ID or SESSION_ID == "__SESSION_ID__":
    raise RuntimeError("SESSION_ID was not injected")

if not API_URL or API_URL == "__API_URL__":
    raise RuntimeError("API_URL was not injected")

if not WORKER_TOKEN or WORKER_TOKEN == "__WORKER_TOKEN__":
    raise RuntimeError("WORKER_TOKEN was not injected")


API_URL = API_URL.rstrip("/")


# ============================================================
# LOG
# ============================================================

def log(message):
    print(message, flush=True)


def api_headers():
    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "kaggle-gpu-worker",
    }


# ============================================================
# API REQUEST
# ============================================================

def api_post(path, data=None):

    url = API_URL + path

    try:

        response = requests.post(
            url,
            headers=api_headers(),
            json=data or {},
            timeout=REQUEST_TIMEOUT,
        )

        log(
            f"[API] POST {path} "
            f"→ {response.status_code}"
        )

        if response.text:
            log(response.text[:2000])

        return response

    except Exception as e:

        log(
            f"[API] POST {path} ERROR: {e}"
        )

        return None


def api_get(path):

    url = API_URL + path

    try:

        response = requests.get(
            url,
            headers=api_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        log(
            f"[API] GET {path} "
            f"→ {response.status_code}"
        )

        return response

    except Exception as e:

        log(
            f"[API] GET {path} ERROR: {e}"
        )

        return None


# ============================================================
# GPU INFORMATION
# ============================================================

def get_gpu_info():

    try:

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        if result.returncode != 0:
            return None

        return result.stdout.strip()

    except Exception:
        return None


# ============================================================
# CUDA TEST
# ============================================================

def cuda_test():

    try:

        import torch

        log("")
        log("=" * 60)
        log("CUDA TEST")
        log("=" * 60)

        cuda_available = torch.cuda.is_available()

        log(
            f"CUDA available: {cuda_available}"
        )

        if not cuda_available:
            return False, None

        device = torch.device("cuda")

        capability = torch.cuda.get_device_capability()

        log(
            f"Compute capability: {capability}"
        )

        x = torch.randn(
            1024,
            1024,
            device=device,
        )

        y = torch.matmul(x, x)

        torch.cuda.synchronize()

        log(
            f"GPU kernel OK: {y.numel()} elements"
        )

        del x
        del y

        torch.cuda.empty_cache()

        return True, capability

    except Exception:

        log(
            traceback.format_exc()
        )

        return False, None


# ============================================================
# WORKER READY
# ============================================================

def notify_worker_ready():

    gpu = get_gpu_info()

    cuda_ok, capability = cuda_test()

    payload = {
        "gpu": gpu,
        "cuda_available": cuda_ok,
        "compute_capability": (
            list(capability)
            if capability
            else None
        ),
    }

    log("")
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)

    response = api_post(
        f"/gpu/session/{SESSION_ID}/worker-ready",
        payload,
    )

    if response is not None and response.ok:

        log(
            "WORKER READY accepted by Railway."
        )

        return True

    log(
        "WORKER READY request failed."
    )

    return False


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():

    response = api_post(
        f"/gpu/session/{SESSION_ID}/heartbeat",
        {
            "timestamp": time.time(),
        },
    )

    return (
        response is not None
        and response.ok
    )


# ============================================================
# GET COMMAND
# ============================================================

def get_command():

    response = api_get(
        f"/internal/session/{SESSION_ID}/command"
    )

    if response is None:
        return None

    if not response.ok:
        return None

    try:

        data = response.json()

        return data

    except Exception:
        return None


# ============================================================
# SEND RESULT
# ============================================================

def send_result(result):

    api_post(
        f"/internal/session/{SESSION_ID}/result",
        result,
    )


# ============================================================
# EXECUTE COMMAND
# ============================================================

def execute_command(command):

    if not command:
        return

    log("")
    log("=" * 60)
    log("COMMAND RECEIVED")
    log("=" * 60)

    log(json.dumps(
        command,
        indent=2,
        default=str,
    ))

    cmd = command.get("command")

    if not cmd:
        return

    try:

        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=900,
        )

        result = {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout[-10000:],
            "stderr": process.stderr[-10000:],
        }

    except Exception as e:

        result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    send_result(result)


# ============================================================
# KEEP ALIVE LOOP
# ============================================================

def worker_loop():

    log("")
    log("=" * 60)
    log("COMMAND LOOP STARTED")
    log("=" * 60)

    last_heartbeat = 0

    while True:

        now = time.time()

        # --------------------------------------------
        # HEARTBEAT
        # --------------------------------------------

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            heartbeat()

            last_heartbeat = now

        # --------------------------------------------
        # COMMAND
        # --------------------------------------------

        try:

            command = get_command()

            if command:

                execute_command(command)

        except Exception:

            log(
                traceback.format_exc()
            )

        # --------------------------------------------
        # KEEP PROCESS ALIVE
        # --------------------------------------------

        time.sleep(
            COMMAND_INTERVAL
        )


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 60)
    log("KAGGLE GPU WORKER")
    log("=" * 60)

    log(f"SESSION : {SESSION_ID}")
    log(f"API     : {API_URL}")
    log(
        f"TOKEN   : {len(WORKER_TOKEN)}"
    )

    log("")
    log("=" * 60)
    log("NVIDIA SMI")
    log("=" * 60)

    try:

        subprocess.run(
            ["nvidia-smi"],
            check=False,
        )

    except Exception as e:

        log(
            f"nvidia-smi error: {e}"
        )

    gpu = get_gpu_info()

    log("")
    log(
        f"GPU: {gpu}"
    )

    # --------------------------------------------
    # CUDA
    # --------------------------------------------

    cuda_ok, capability = cuda_test()

    if not cuda_ok:

        log(
            "CUDA test failed."
        )

    # --------------------------------------------
    # WORKER READY
    # --------------------------------------------

    if not notify_worker_ready():

        raise RuntimeError(
            "Failed to notify Railway"
        )

    # --------------------------------------------
    # KEEP ALIVE
    # --------------------------------------------

    worker_loop()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        log(
            "Worker stopped."
        )

    except Exception:

        log(
            traceback.format_exc()
        )

        sys.exit(1)
