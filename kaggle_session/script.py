import os
import sys
import time
import json
import subprocess
import traceback

import requests


SESSION_ID = "%%SESSION_ID%%"
API_URL = "%%API_URL%%".rstrip("/")
WORKER_TOKEN = "%%WORKER_TOKEN%%"

HEARTBEAT_INTERVAL = 15
COMMAND_INTERVAL = 3


def get_headers():
    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "Content-Type": "application/json",
    }


def post_api(path, data=None):
    url = f"{API_URL}{path}"

    try:
        r = requests.post(
            url,
            json=data or {},
            headers=get_headers(),
            timeout=20,
        )

        print(
            f"[API] POST {path} → {r.status_code}",
            flush=True,
        )

        if r.text:
            print(r.text[:2000], flush=True)

        return r

    except Exception as e:
        print(
            f"[API] POST {path} ERROR: {e}",
            flush=True,
        )
        return None


def get_api(path):
    url = f"{API_URL}{path}"

    try:
        r = requests.get(
            url,
            headers=get_headers(),
            timeout=20,
        )

        print(
            f"[API] GET {path} → {r.status_code}",
            flush=True,
        )

        return r

    except Exception as e:
        print(
            f"[API] GET {path} ERROR: {e}",
            flush=True,
        )
        return None


def gpu_test():

    print("=" * 60, flush=True)
    print("CUDA TEST", flush=True)
    print("=" * 60, flush=True)

    import torch

    print(
        f"PyTorch: {torch.__version__}",
        flush=True,
    )

    cuda = torch.cuda.is_available()

    print(
        f"CUDA available: {cuda}",
        flush=True,
    )

    if not cuda:
        return False

    gpu = torch.cuda.get_device_name(0)

    capability = torch.cuda.get_device_capability(0)

    print(
        f"GPU: {gpu}",
        flush=True,
    )

    print(
        f"Compute capability: {capability}",
        flush=True,
    )

    x = torch.randn(
        1024,
        1024,
        device="cuda",
    )

    y = x @ x

    torch.cuda.synchronize()

    print(
        f"GPU kernel OK: {y.numel()} elements",
        flush=True,
    )

    del x
    del y

    torch.cuda.empty_cache()

    return True


def notify_ready():

    import torch

    gpu = torch.cuda.get_device_name(0)

    capability = list(
        torch.cuda.get_device_capability(0)
    )

    data = {
        "gpu": gpu,
        "cuda_available": True,
        "compute_capability": capability,
    }

    for attempt in range(1, 11):

        print(
            f"worker-ready attempt {attempt}/10",
            flush=True,
        )

        r = post_api(
            f"/gpu/session/{SESSION_ID}/worker-ready",
            data,
        )

        if r is not None and r.status_code == 200:

            print(
                "WORKER READY accepted by Railway.",
                flush=True,
            )

            return True

        time.sleep(3)

    return False


def heartbeat():

    r = post_api(
        f"/gpu/session/{SESSION_ID}/heartbeat",
        {
            "status": "alive",
            "timestamp": time.time(),
        },
    )

    return (
        r is not None
        and r.status_code == 200
    )


def get_session():

    r = get_api(
        f"/gpu/session/{SESSION_ID}"
    )

    if r is None:
        return None

    if r.status_code != 200:
        return None

    try:
        return r.json()
    except Exception:
        return None


def get_command():

    r = get_api(
        f"/internal/session/{SESSION_ID}/command"
    )

    if r is None:
        return None

    if r.status_code != 200:
        return None

    try:
        data = r.json()
    except Exception:
        return None

    if not data:
        return None

    return data


def send_result(command_id, result):

    post_api(
        f"/internal/session/{SESSION_ID}/result",
        {
            "command_id": command_id,
            "result": result,
        },
    )


def execute_command(command):

    print("=" * 60, flush=True)
    print("COMMAND RECEIVED", flush=True)
    print("=" * 60, flush=True)

    print(
        json.dumps(
            command,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    command_id = command.get("command_id")

    cmd = command.get("command")

    if not cmd:

        result = {
            "success": False,
            "error": "No command supplied",
        }

        if command_id:
            send_result(
                command_id,
                result,
            )

        return

    try:

        print(
            f"Executing: {cmd}",
            flush=True,
        )

        process = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600,
        )

        result = {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout[-10000:],
            "stderr": process.stderr[-10000:],
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
        }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    if command_id:
        send_result(
            command_id,
            result,
        )


def main():

    print("=" * 60, flush=True)
    print("KAGGLE GPU WORKER", flush=True)
    print("=" * 60, flush=True)

    print(
        f"SESSION : {SESSION_ID}",
        flush=True,
    )

    print(
        f"API     : {API_URL}",
        flush=True,
    )

    print(
        f"TOKEN   : {len(WORKER_TOKEN)}",
        flush=True,
    )

    print("=" * 60, flush=True)

    # -------------------------------------------------
    # GPU
    # -------------------------------------------------

    if not gpu_test():

        raise RuntimeError(
            "GPU/CUDA test failed"
        )

    # -------------------------------------------------
    # READY
    # -------------------------------------------------

    if not notify_ready():

        raise RuntimeError(
            "Railway did not accept worker-ready"
        )

    # -------------------------------------------------
    # KEEP ALIVE
    # -------------------------------------------------

    print("=" * 60, flush=True)
    print("KEEP-ALIVE LOOP STARTED", flush=True)
    print("=" * 60, flush=True)

    last_heartbeat = 0
    counter = 0

    while True:

        counter += 1

        now = time.time()

        print(
            f"[LOOP {counter}] Worker alive",
            flush=True,
        )

        # HEARTBEAT

        if (
            now - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            heartbeat()

            last_heartbeat = now

        # SESSION STATUS

        session = get_session()

        if session:

            status = session.get(
                "status"
            )

            remaining = session.get(
                "remaining_seconds"
            )

            print(
                f"[SESSION] "
                f"status={status} "
                f"remaining={remaining}",
                flush=True,
            )

            if status in (
                "stopped",
                "expired",
                "completed",
                "failed",
                "error",
            ):

                print(
                    "[SESSION] Session ended.",
                    flush=True,
                )

                break

        # COMMAND

        command = get_command()

        if command:

            execute_command(command)

        # WAIT

        time.sleep(
            COMMAND_INTERVAL
        )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "Worker stopped.",
            flush=True,
        )

    except Exception as e:

        print(
            f"WORKER ERROR: {e}",
            flush=True,
        )

        traceback.print_exc()

        sys.exit(1)
