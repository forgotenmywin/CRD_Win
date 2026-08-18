import os
import sys
import time
import json
import traceback
import subprocess

import requests


# ============================================================
# REAL VALUES ARE INJECTED BY GITHUB ACTIONS
# ============================================================

SESSION_ID = "__SESSION_ID__"
API_URL = "__API_URL__"
WORKER_TOKEN = "__WORKER_TOKEN__"


# ============================================================
# CONFIG
# ============================================================

TEST_SECONDS = 600

HEARTBEAT_INTERVAL = 30

COMMAND_INTERVAL = 30


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "X-Worker-Token": WORKER_TOKEN,
    "Content-Type": "application/json",
}


# ============================================================
# PRINT
# ============================================================

def log(message):

    print(
        message,
        flush=True
    )


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config():

    if not SESSION_ID:
        raise RuntimeError(
            "SESSION_ID was not injected"
        )

    if SESSION_ID.startswith("__"):
        raise RuntimeError(
            "SESSION_ID placeholder was not replaced"
        )

    if not API_URL:
        raise RuntimeError(
            "API_URL was not injected"
        )

    if API_URL.startswith("__"):
        raise RuntimeError(
            "API_URL placeholder was not replaced"
        )

    if not WORKER_TOKEN:
        raise RuntimeError(
            "WORKER_TOKEN was not injected"
        )

    if WORKER_TOKEN.startswith("__"):
        raise RuntimeError(
            "WORKER_TOKEN placeholder was not replaced"
        )


# ============================================================
# SHOW CONFIG
# ============================================================

def show_config():

    log("=" * 60)
    log("KAGGLE GPU WORKER")
    log("=" * 60)

    log(
        f"SESSION: {SESSION_ID}"
    )

    log(
        f"API: {API_URL}"
    )

    log(
        f"TOKEN: {len(WORKER_TOKEN)} chars"
    )

    log("")


# ============================================================
# NVIDIA-SMI
# ============================================================

def show_gpu():

    log("=" * 60)
    log("NVIDIA-SMI")
    log("=" * 60)

    try:

        result = subprocess.run(
            [
                "nvidia-smi"
            ],
            capture_output=True,
            text=True
        )

        log(
            result.stdout
        )

        if result.stderr:

            log(
                result.stderr
            )

    except Exception as e:

        log(
            f"nvidia-smi error: {e}"
        )


# ============================================================
# CUDA TEST
# ============================================================

def gpu_test():

    log("=" * 60)
    log("CUDA TEST")
    log("=" * 60)

    try:

        import torch

        log(
            f"CUDA available: {torch.cuda.is_available()}"
        )

        if not torch.cuda.is_available():

            return False

        gpu_name = torch.cuda.get_device_name(0)

        capability = (
            torch.cuda.get_device_capability(0)
        )

        log(
            f"GPU: {gpu_name}"
        )

        log(
            f"Compute capability: {capability}"
        )

        start = time.time()

        x = torch.randn(
            1024,
            1024,
            device="cuda"
        )

        y = torch.matmul(
            x,
            x
        )

        torch.cuda.synchronize()

        elapsed = (
            time.time() - start
        )

        log(
            f"GPU JOB SUCCESS: {y.numel()} elements"
        )

        log(
            f"GPU kernel time: {elapsed:.4f}s"
        )

        del x
        del y

        return True

    except Exception as e:

        log(
            f"CUDA ERROR: {repr(e)}"
        )

        traceback.print_exc()

        return False


# ============================================================
# API REQUEST
# ============================================================

def api_post(path):

    url = (
        API_URL.rstrip("/")
        + "/"
        + path.lstrip("/")
    )

    try:

        response = requests.post(
            url,
            headers=HEADERS,
            timeout=20
        )

        log(
            f"[API] POST {path} -> {response.status_code}"
        )

        try:

            data = response.json()

            log(
                str(data)
            )

        except Exception:

            log(
                response.text
            )

        return response

    except Exception as e:

        log(
            f"[API] POST ERROR {url}: {e}"
        )

        return None


# ============================================================
# API GET
# ============================================================

def api_get(path):

    url = (
        API_URL.rstrip("/")
        + "/"
        + path.lstrip("/")
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        log(
            f"[API] GET {path} -> {response.status_code}"
        )

        try:

            return response.json()

        except Exception:

            return {}

    except Exception as e:

        log(
            f"[API] GET ERROR {url}: {e}"
        )

        return {}


# ============================================================
# WORKER READY
# ============================================================

def worker_ready():

    log("=" * 60)
    log("NOTIFYING RAILWAY")
    log("=" * 60)

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):

        log(
            f"worker-ready attempt "
            f"{attempt}/10"
        )

        response = api_post(
            path
        )

        if response is not None:

            if response.status_code == 200:

                log(
                    "WORKER READY accepted."
                )

                return True

            if response.status_code == 401:

                log(
                    "ERROR: Railway rejected WORKER_TOKEN."
                )

                return False

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

    response = api_post(
        path
    )

    if response is None:

        return None

    try:

        return response.json()

    except Exception:

        return None


# ============================================================
# COMMAND
# ============================================================

def get_command():

    path = (
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )

    return api_get(
        path
    )


# ============================================================
# WORKER LOOP
# ============================================================

def worker_loop():

    log("=" * 60)
    log("GPU WORKER IS ACTIVE")
    log("=" * 60)

    started = time.time()

    last_heartbeat = 0

    last_command = 0

    while True:

        now = time.time()

        elapsed = int(
            now - started
        )

        # ----------------------------------------------------
        # SESSION TEST TIME
        # ----------------------------------------------------

        if elapsed >= TEST_SECONDS:

            log(
                "TEST_SECONDS reached."
            )

            break

        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if (
            now - last_heartbeat
            >= HEARTBEAT_INTERVAL
        ):

            data = heartbeat()

            last_heartbeat = now

            if data:

                remaining = data.get(
                    "remaining_seconds"
                )

                log(
                    f"[KEEP-ALIVE] "
                    f"remaining={remaining}"
                )

                status = data.get(
                    "status"
                )

                if status in (
                    "expired",
                    "completed",
                    "error"
                ):

                    log(
                        f"Session status: {status}"
                    )

                    break

        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        if (
            now - last_command
            >= COMMAND_INTERVAL
        ):

            command = get_command()

            last_command = now

            log(
                f"[COMMAND] {command}"
            )

            if command:

                cmd = command.get(
                    "command"
                )

                if cmd:

                    log(
                        f"[COMMAND RECEIVED] {cmd}"
                    )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        log(
            f"[WORKER] "
            f"{elapsed}/{TEST_SECONDS}s"
        )

        time.sleep(1)

    log("=" * 60)
    log("GPU WORKER FINISHED")
    log("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    validate_config()

    show_config()

    show_gpu()

    if not gpu_test():

        raise RuntimeError(
            "GPU test failed"
        )

    if not worker_ready():

        raise RuntimeError(
            "Could not notify Railway"
        )

    worker_loop()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        log("")
        log("=" * 60)
        log("WORKER ERROR")
        log("=" * 60)

        log(
            repr(e)
        )

        traceback.print_exc()

        sys.exit(1)
