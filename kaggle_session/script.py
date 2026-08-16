import os
import sys
import time
import json
import socket
import subprocess
import traceback

import requests


SESSION_ID = os.environ.get("SESSION_ID", "")
API_URL = os.environ.get("API_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

HEARTBEAT_INTERVAL = 20
COMMAND_INTERVAL = 2

if not SESSION_ID:
    raise RuntimeError("SESSION_ID is missing")

if not API_URL:
    raise RuntimeError("API_URL is missing")

if not WORKER_TOKEN:
    raise RuntimeError("WORKER_TOKEN is missing")


HEADERS = {
    "Authorization": f"Bearer {WORKER_TOKEN}",
    "Content-Type": "application/json",
}


def log(message):
    print(message, flush=True)


def post(path, payload=None, timeout=15):
    url = f"{API_URL}{path}"

    try:
        r = requests.post(
            url,
            headers=HEADERS,
            json=payload or {},
            timeout=timeout,
        )

        log(f"[API] POST {path} → {r.status_code}")

        return r

    except Exception as e:
        log(f"[API] POST {path} ERROR: {e}")
        return None


def get(path, timeout=15):
    url = f"{API_URL}{path}"

    try:
        r = requests.get(
            url,
            headers=HEADERS,
            timeout=timeout,
        )

        log(f"[API] GET {path} → {r.status_code}")

        return r

    except Exception as e:
        log(f"[API] GET {path} ERROR: {e}")
        return None


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
            timeout=15,
        )

        return result.stdout.strip()

    except Exception as e:
        return f"GPU info error: {e}"


def cuda_test():
    try:
        import torch

        log(f"PyTorch: {torch.__version__}")
        log(f"CUDA available: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda")

        props = torch.cuda.get_device_properties(0)

        log(f"GPU: {props.name}")
        log(f"VRAM: {props.total_memory // (1024 * 1024)} MiB")
        log(
            f"Compute capability: "
            f"{props.major}.{props.minor}"
        )

        x = torch.randn(
            1024,
            1024,
            device=device,
        )

        y = x @ x

        torch.cuda.synchronize()

        log(
            f"GPU kernel OK: "
            f"{y.numel()} elements"
        )

        del x
        del y

        torch.cuda.empty_cache()

        return True

    except Exception:
        traceback.print_exc()
        return False


def worker_ready():
    payload = {
        "session_id": SESSION_ID,
        "gpu": get_gpu_info(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }

    r = post(
        f"/gpu/session/{SESSION_ID}/worker-ready",
        payload,
    )

    if r is not None and r.ok:
        log("WORKER READY accepted by Railway.")
        return True

    log("WORKER READY was rejected.")
    return False


def heartbeat():
    payload = {
        "session_id": SESSION_ID,
        "timestamp": time.time(),
        "gpu": get_gpu_info(),
    }

    r = post(
        f"/gpu/session/{SESSION_ID}/heartbeat",
        payload,
    )

    return r is not None and r.ok


def get_command():
    r = get(
        f"/internal/session/{SESSION_ID}/command"
    )

    if r is None:
        return None

    if r.status_code != 200:
        return None

    try:
        return r.json()
    except Exception:
        return None


def send_result(command_id, result):
    payload = {
        "session_id": SESSION_ID,
        "command_id": command_id,
        "result": result,
    }

    post(
        f"/internal/session/{SESSION_ID}/result",
        payload,
    )


def execute_command(command):
    if not command:
        return

    command_id = command.get("command_id")

    cmd = command.get("command")

    if not cmd:
        return

    log("=" * 60)
    log("COMMAND RECEIVED")
    log(cmd)
    log("=" * 60)

    try:
        completed = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600,
        )

        result = {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        }

    except subprocess.TimeoutExpired:
        result = {
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timeout",
        }

    except Exception as e:
        result = {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }

    send_result(command_id, result)


def main():
    log("=" * 60)
    log("KAGGLE GPU WORKER")
    log("=" * 60)

    log(f"SESSION : {SESSION_ID}")
    log(f"API     : {API_URL}")
    log(f"TOKEN   : {len(WORKER_TOKEN)}")
    log("=" * 60)

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
        log(f"nvidia-smi error: {e}")

    log("")
    log("=" * 60)
    log("CUDA TEST")
    log("=" * 60)

    cuda_ok = cuda_test()

    if not cuda_ok:
        log("CUDA TEST FAILED")

        post(
            f"/gpu/session/{SESSION_ID}/heartbeat",
            {
                "error": "CUDA test failed",
            },
        )

        sys.exit(1)

    log("")
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)

    if not worker_ready():
        log("Could not notify Railway.")
        sys.exit(1)

    log("")
    log("=" * 60)
    log("COMMAND LOOP")
    log("=" * 60)

    last_heartbeat = 0

    while True:

        now = time.time()

        # -----------------------------
        # KEEP ALIVE
        # -----------------------------

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            ok = heartbeat()

            if ok:
                log("[KEEP-ALIVE] heartbeat OK")
            else:
                log("[KEEP-ALIVE] heartbeat FAILED")

            last_heartbeat = now

        # -----------------------------
        # CHECK COMMAND
        # -----------------------------

        command = get_command()

        if command:

            status = command.get("status")

            if status in ("stop", "stopped", "terminate"):
                log("STOP COMMAND RECEIVED")
                break

            execute_command(command)

        time.sleep(COMMAND_INTERVAL)

    log("Worker stopping.")

    try:
        post(
            f"/gpu/session/{SESSION_ID}/heartbeat",
            {
                "status": "stopping",
            },
        )
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Keyboard interrupt.")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
