import os
import sys
import time
import json
import traceback
import requests
import subprocess


SESSION_ID = os.environ.get("SESSION_ID", "")
API_URL = os.environ.get("API_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

HEARTBEAT_INTERVAL = 15
COMMAND_INTERVAL = 3


def log(message):
    print(message, flush=True)


def api_headers():
    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "Content-Type": "application/json",
    }


def api_post(path, data=None):
    url = f"{API_URL}{path}"

    try:
        r = requests.post(
            url,
            headers=api_headers(),
            json=data or {},
            timeout=20,
        )

        log(f"[API] POST {path} → {r.status_code}")

        return r

    except Exception as e:
        log(f"[API] POST ERROR {path}: {e}")
        return None


def api_get(path):
    url = f"{API_URL}{path}"

    try:
        r = requests.get(
            url,
            headers=api_headers(),
            timeout=20,
        )

        log(f"[API] GET {path} → {r.status_code}")

        return r

    except Exception as e:
        log(f"[API] GET ERROR {path}: {e}")
        return None


def gpu_info():
    log("=" * 60)
    log("NVIDIA-SMI")
    log("=" * 60)

    try:
        subprocess.run(
            ["nvidia-smi"],
            check=False,
        )
    except Exception as e:
        log(f"nvidia-smi error: {e}")


def cuda_test():
    log("=" * 60)
    log("CUDA TEST")
    log("=" * 60)

    try:
        import torch

        log(f"PyTorch: {torch.__version__}")
        log(f"CUDA available: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda")

        props = torch.cuda.get_device_properties(0)

        log(f"GPU: {props.name}")
        log(
            f"Compute capability: "
            f"({props.major}, {props.minor})"
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

        return True

    except Exception:
        traceback.print_exc()
        return False


def worker_ready():
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)

    r = api_post(
        f"/gpu/session/{SESSION_ID}/worker-ready",
        {
            "gpu": "Tesla P100-PCIE-16GB",
            "status": "ready",
        },
    )

    if r is not None and r.ok:
        log("WORKER READY accepted by Railway.")
        return True

    log("WORKER READY failed.")
    return False


def heartbeat():
    r = api_post(
        f"/gpu/session/{SESSION_ID}/heartbeat",
        {
            "status": "active",
        },
    )

    return r is not None and r.ok


def get_command():
    r = api_get(
        f"/internal/session/{SESSION_ID}/command"
    )

    if r is None or not r.ok:
        return None

    try:
        data = r.json()

        if not data:
            return None

        return data

    except Exception:
        return None


def send_result(command_id, result):
    api_post(
        f"/internal/session/{SESSION_ID}/result",
        {
            "command_id": command_id,
            "result": result,
        },
    )


def execute_command(command):
    command_id = command.get("id")

    cmd = command.get("command")

    if not cmd:
        return

    log("=" * 60)
    log("COMMAND")
    log(cmd)
    log("=" * 60)

    try:
        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=900,
        )

        result = {
            "returncode": process.returncode,
            "stdout": process.stdout[-20000:],
            "stderr": process.stderr[-20000:],
        }

    except Exception as e:
        result = {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }

    send_result(command_id, result)


def main():

    if not SESSION_ID:
        raise RuntimeError("SESSION_ID is missing")

    if not API_URL:
        raise RuntimeError("API_URL is missing")

    if not WORKER_TOKEN:
        raise RuntimeError("WORKER_TOKEN is missing")

    log("=" * 60)
    log("KAGGLE GPU WORKER")
    log("=" * 60)

    log(f"SESSION : {SESSION_ID}")
    log(f"API     : {API_URL}")
    log(f"TOKEN   : {len(WORKER_TOKEN)}")

    gpu_info()

    cuda_ok = cuda_test()

    if not cuda_ok:
        api_post(
            f"/gpu/session/{SESSION_ID}/worker-ready",
            {
                "status": "error",
                "error": "CUDA unavailable",
            },
        )

        raise RuntimeError("CUDA test failed")

    if not worker_ready():
        raise RuntimeError(
            "Railway rejected worker-ready"
        )

    log("=" * 60)
    log("COMMAND LOOP")
    log("=" * 60)

    last_heartbeat = 0

    while True:

        now = time.time()

        # KEEP ALIVE
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            heartbeat()

            last_heartbeat = now

        # COMMAND POLLING
        command = get_command()

        if command:
            execute_command(command)

        time.sleep(COMMAND_INTERVAL)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        log("Worker stopped.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)
