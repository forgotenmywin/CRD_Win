import os
import sys
import time
import json
import subprocess
import requests
import traceback

SESSION_ID = os.environ.get("SESSION_ID", "")
API_URL = os.environ.get("API_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

HEARTBEAT_INTERVAL = 15
COMMAND_INTERVAL = 3

if not SESSION_ID:
    raise RuntimeError("SESSION_ID is missing")

if not API_URL:
    raise RuntimeError("API_URL is missing")

if not WORKER_TOKEN:
    raise RuntimeError("WORKER_TOKEN is missing")


def headers():
    return {
        "Authorization": f"Bearer {WORKER_TOKEN}",
        "Content-Type": "application/json",
    }


def api_post(path, data=None, timeout=20):
    url = f"{API_URL}{path}"

    try:
        r = requests.post(
            url,
            json=data or {},
            headers=headers(),
            timeout=timeout,
        )

        print(
            f"[API] POST {path} → {r.status_code}",
            flush=True
        )

        if r.text:
            print(r.text[:2000], flush=True)

        return r

    except Exception as e:
        print(
            f"[API] POST {path} ERROR: {e}",
            flush=True
        )
        return None


def api_get(path, timeout=20):
    url = f"{API_URL}{path}"

    try:
        r = requests.get(
            url,
            headers=headers(),
            timeout=timeout,
        )

        print(
            f"[API] GET {path} → {r.status_code}",
            flush=True
        )

        return r

    except Exception as e:
        print(
            f"[API] GET {path} ERROR: {e}",
            flush=True
        )
        return None


def gpu_test():
    print("=" * 60, flush=True)
    print("GPU TEST", flush=True)
    print("=" * 60, flush=True)

    try:
        import torch

        print(
            f"PyTorch: {torch.__version__}",
            flush=True
        )

        print(
            f"CUDA available: {torch.cuda.is_available()}",
            flush=True
        )

        if not torch.cuda.is_available():
            return False

        gpu = torch.cuda.get_device_name(0)

        print(
            f"GPU: {gpu}",
            flush=True
        )

        capability = torch.cuda.get_device_capability(0)

        print(
            f"Compute capability: {capability}",
            flush=True
        )

        x = torch.randn(
            1024,
            1024,
            device="cuda"
        )

        y = x @ x

        torch.cuda.synchronize()

        print(
            f"GPU kernel OK: {y.numel()} elements",
            flush=True
        )

        del x
        del y

        torch.cuda.empty_cache()

        return True

    except Exception:
        traceback.print_exc()
        return False


def worker_ready():
    print("=" * 60, flush=True)
    print("NOTIFYING RAILWAY", flush=True)
    print("=" * 60, flush=True)

    try:
        import torch

        gpu = torch.cuda.get_device_name(0)
        capability = list(
            torch.cuda.get_device_capability(0)
        )

        data = {
            "gpu": gpu,
            "compute_capability": capability,
            "cuda_available": True,
        }

        r = api_post(
            f"/gpu/session/{SESSION_ID}/worker-ready",
            data,
        )

        if r is not None and r.status_code == 200:
            print(
                "WORKER READY accepted by Railway.",
                flush=True
            )
            return True

        print(
            "WORKER READY was not accepted.",
            flush=True
        )

        return False

    except Exception:
        traceback.print_exc()
        return False


def heartbeat():
    r = api_post(
        f"/gpu/session/{SESSION_ID}/heartbeat",
        {
            "timestamp": time.time(),
            "status": "alive",
        },
    )

    return r is not None and r.status_code == 200


def get_command():
    r = api_get(
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
    return api_post(
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
            ensure_ascii=False
        ),
        flush=True
    )

    command_id = command.get("command_id")

    cmd = command.get("command")

    if not cmd:
        return {
            "success": False,
            "error": "No command supplied",
        }

    try:

        print(
            f"Executing: {cmd}",
            flush=True
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

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            ),
            flush=True
        )

        if command_id:
            send_result(
                command_id,
                result
            )

        return result

    except subprocess.TimeoutExpired:

        result = {
            "success": False,
            "error": "Command timeout",
        }

        if command_id:
            send_result(
                command_id,
                result
            )

        return result

    except Exception as e:

        result = {
            "success": False,
            "error": str(e),
        }

        if command_id:
            send_result(
                command_id,
                result
            )

        return result


def check_session():
    r = api_get(
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


def main():

    print("=" * 60, flush=True)
    print("KAGGLE GPU WORKER", flush=True)
    print("=" * 60, flush=True)

    print(
        f"SESSION : {SESSION_ID}",
        flush=True
    )

    print(
        f"API     : {API_URL}",
        flush=True
    )

    print(
        f"TOKEN   : {len(WORKER_TOKEN)}",
        flush=True
    )

    print("=" * 60, flush=True)

    # --------------------------------------------------
    # GPU TEST
    # --------------------------------------------------

    if not gpu_test():
        raise RuntimeError(
            "GPU/CUDA test failed"
        )

    # --------------------------------------------------
    # WORKER READY
    # --------------------------------------------------

    ready = False

    for attempt in range(1, 11):

        print(
            f"worker-ready attempt {attempt}/10",
            flush=True
        )

        if worker_ready():
            ready = True
            break

        time.sleep(3)

    if not ready:
        raise RuntimeError(
            "Could not notify Railway that worker is ready"
        )

    # --------------------------------------------------
    # KEEP ALIVE LOOP
    # --------------------------------------------------

    print("=" * 60, flush=True)
    print("KEEP-ALIVE LOOP STARTED", flush=True)
    print("=" * 60, flush=True)

    last_heartbeat = 0
    loop_counter = 0

    while True:

        loop_counter += 1

        now = time.time()

        print(
            f"[LOOP {loop_counter}] Worker alive",
            flush=True
        )

        # ----------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            print(
                "[KEEP-ALIVE] Sending heartbeat...",
                flush=True
            )

            heartbeat()

            last_heartbeat = now

        # ----------------------------------------------
        # SESSION STATUS
        # ----------------------------------------------

        status = check_session()

        if status:

            current_status = status.get(
                "status"
            )

            remaining = status.get(
                "remaining_seconds"
            )

            print(
                f"[SESSION] status={current_status} "
                f"remaining={remaining}",
                flush=True
            )

            # Session stopped/expired
            if current_status in (
                "stopped",
                "expired",
                "error",
                "failed",
                "completed",
            ):

                print(
                    "[SESSION] Session ended.",
                    flush=True
                )

                break

        # ----------------------------------------------
        # COMMAND
        # ----------------------------------------------

        command = get_command()

        if command:

            execute_command(command)

        # ----------------------------------------------
        # WAIT
        # ----------------------------------------------

        time.sleep(COMMAND_INTERVAL)

    print("=" * 60, flush=True)
    print("WORKER STOPPED", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "Worker interrupted.",
            flush=True
        )

    except Exception as e:

        print(
            f"WORKER ERROR: {e}",
            flush=True
        )

        traceback.print_exc()

        sys.exit(1)
