import time
import subprocess
import traceback
import requests

# These lines are replaced by GitHub Actions
SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"

HEARTBEAT_INTERVAL = 15
COMMAND_INTERVAL = 3
TIMEOUT = 20


def log(text=""):
    print(text, flush=True)


def headers():
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WORKER_TOKEN}"
    }


def post(path, data=None):
    url = API_URL.rstrip("/") + path

    try:
        r = requests.post(
            url,
            json=data or {},
            headers=headers(),
            timeout=TIMEOUT
        )

        log(f"[API] POST {path} → {r.status_code}")

        try:
            return r.json()
        except Exception:
            return {}

    except Exception as e:
        log(f"[API] POST ERROR: {e}")
        return None


def get(path):
    url = API_URL.rstrip("/") + path

    try:
        r = requests.get(
            url,
            headers=headers(),
            timeout=TIMEOUT
        )

        log(f"[API] GET {path} → {r.status_code}")

        try:
            return r.json()
        except Exception:
            return {}

    except Exception as e:
        log(f"[API] GET ERROR: {e}")
        return None


def command(command):
    try:
        p = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )

        return {
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr
        }

    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }


def gpu_info():

    p = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total,driver_version "
        "--format=csv,noheader",
        shell=True,
        capture_output=True,
        text=True
    )

    if p.returncode != 0:
        raise RuntimeError("nvidia-smi failed")

    line = p.stdout.strip()

    log(line)

    parts = [x.strip() for x in line.split(",")]

    return {
        "name": parts[0] if len(parts) > 0 else "Unknown",
        "memory": parts[1] if len(parts) > 1 else "Unknown",
        "driver": parts[2] if len(parts) > 2 else "Unknown"
    }


def heartbeat():

    post(
        f"/gpu/session/{SESSION_ID}/heartbeat",
        {
            "session_id": SESSION_ID,
            "timestamp": time.time()
        }
    )


def worker_ready(info):

    data = {
        "session_id": SESSION_ID,
        "gpu": (
            f"{info['name']}, "
            f"{info['memory']}, "
            f"{info['driver']}"
        ),
        "cuda_available": True,
        "compute_capability": [6, 0]
    }

    for attempt in range(1, 11):

        log(
            f"[API] worker-ready attempt={attempt}/10"
        )

        result = post(
            f"/gpu/session/{SESSION_ID}/worker-ready",
            data
        )

        if result is not None:
            log("WORKER READY accepted by Railway.")
            return True

        time.sleep(3)

    return False


def command_loop():

    log("=" * 60)
    log("COMMAND LOOP")
    log("=" * 60)

    last_heartbeat = 0

    while True:

        now = time.time()

        # ============================
        # KEEP ALIVE
        # ============================

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            log("[KEEP-ALIVE] heartbeat")

            heartbeat()

            last_heartbeat = now

        # ============================
        # COMMAND
        # ============================

        result = get(
            f"/internal/session/{SESSION_ID}/command"
        )

        if result:

            cmd = result.get("command")

            if cmd:

                log("=" * 60)
                log("COMMAND RECEIVED")
                log(cmd)
                log("=" * 60)

                output = command(cmd)

                post(
                    f"/internal/session/{SESSION_ID}/result",
                    {
                        "session_id": SESSION_ID,
                        "returncode": output["returncode"],
                        "stdout": output["stdout"],
                        "stderr": output["stderr"]
                    }
                )

        time.sleep(COMMAND_INTERVAL)


def main():

    log("=" * 60)
    log("KAGGLE GPU WORKER")
    log("=" * 60)

    # ============================
    # VALIDATE
    # ============================

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

    log(f"SESSION : {SESSION_ID}")
    log(f"API     : {API_URL}")
    log(f"TOKEN   : {len(WORKER_TOKEN)}")

    # ============================
    # GPU
    # ============================

    log()
    log("=" * 60)
    log("NVIDIA GPU")
    log("=" * 60)

    p = subprocess.run(
        "nvidia-smi",
        shell=True,
        capture_output=True,
        text=True
    )

    log(p.stdout)

    if p.returncode != 0:
        log(p.stderr)
        raise RuntimeError("NVIDIA GPU unavailable")

    info = gpu_info()

    log(
        f"GPU: {info['name']}, "
        f"{info['memory']}, "
        f"{info['driver']}"
    )

    # ============================
    # CUDA CHECK
    # ============================

    log()
    log("=" * 60)
    log("CUDA ENVIRONMENT")
    log("=" * 60)

    p = subprocess.run(
        "nvidia-smi --query-gpu=name,compute_cap "
        "--format=csv,noheader",
        shell=True,
        capture_output=True,
        text=True
    )

    log(p.stdout)

    if p.returncode != 0:
        raise RuntimeError("CUDA environment unavailable")

    # ============================
    # READY
    # ============================

    log()
    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)

    if not worker_ready(info):
        raise RuntimeError(
            "Could not notify Railway"
        )

    # ============================
    # LOOP
    # ============================

    command_loop()


try:
    main()

except KeyboardInterrupt:

    log("Worker stopped.")

except Exception as e:

    log("=" * 60)
    log("WORKER ERROR")
    log("=" * 60)

    log(str(e))
    traceback.print_exc()

    raise
