import os
import sys
import time
import json
import subprocess
import traceback
import requests


# ============================================================
# CONFIG
# ============================================================

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


# ============================================================
# HTTP
# ============================================================

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {WORKER_TOKEN}",
    "Content-Type": "application/json",
})


def api_url(path):
    return f"{API_URL}{path}"


def post(path, payload=None, retries=3):
    url = api_url(path)

    for attempt in range(1, retries + 1):
        try:
            r = session.post(
                url,
                json=payload or {},
                timeout=20,
            )

            print(
                f"[API] POST {path} → {r.status_code}",
                flush=True,
            )

            if r.status_code >= 400:
                print(
                    f"[API] response: {r.text[:1000]}",
                    flush=True,
                )

            return r

        except Exception as e:
            print(
                f"[API] POST {path} attempt={attempt} error={e}",
                flush=True,
            )

            if attempt < retries:
                time.sleep(2)

    return None


def get(path, retries=3):
    url = api_url(path)

    for attempt in range(1, retries + 1):
        try:
            r = session.get(
                url,
                timeout=20,
            )

            print(
                f"[API] GET {path} → {r.status_code}",
                flush=True,
            )

            return r

        except Exception as e:
            print(
                f"[API] GET {path} attempt={attempt} error={e}",
                flush=True,
            )

            if attempt < retries:
                time.sleep(2)

    return None


# ============================================================
# NVIDIA
# ============================================================

def nvidia_smi():
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

        print(result.stdout, flush=True)

        if result.stderr:
            print(result.stderr, flush=True)

        return result.returncode == 0

    except Exception as e:
        print(
            f"nvidia-smi failed: {e}",
            flush=True,
        )

        return False


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
    print()
    print("=" * 60)
    print("CUDA TEST")
    print("=" * 60)

    try:
        import torch

        print(
            f"PyTorch: {torch.__version__}",
            flush=True,
        )

        cuda_available = torch.cuda.is_available()

        print(
            f"CUDA available: {cuda_available}",
            flush=True,
        )

        if not cuda_available:
            print(
                "CUDA is not available.",
                flush=True,
            )
            return False, None

        gpu_name = torch.cuda.get_device_name(0)

        capability = torch.cuda.get_device_capability(0)

        print(
            f"GPU: {gpu_name}",
            flush=True,
        )

        print(
            f"Compute capability: {capability}",
            flush=True,
        )

        # ----------------------------------------------------
        # Tesla P100 = sm_60
        #
        # Current Kaggle PyTorch builds may not contain
        # sm_60 kernels.
        #
        # Therefore DO NOT execute torch CUDA kernels on P100
        # with an incompatible PyTorch build.
        # ----------------------------------------------------

        if capability == (6, 0):

            print(
                "P100 detected.",
                flush=True,
            )

            print(
                "Current PyTorch build may not support sm_60.",
                flush=True,
            )

            print(
                "Skipping PyTorch CUDA kernel test.",
                flush=True,
            )

            print(
                "GPU detection itself is OK.",
                flush=True,
            )

            return True, capability

        # ----------------------------------------------------
        # Newer GPUs
        # ----------------------------------------------------

        try:

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

            return True, capability

        except Exception as e:

            print(
                "CUDA kernel test failed:",
                repr(e),
                flush=True,
            )

            return False, capability

    except Exception as e:

        print(
            "CUDA test exception:",
            repr(e),
            flush=True,
        )

        traceback.print_exc()

        return False, None


# ============================================================
# WORKER READY
# ============================================================

def notify_worker_ready(gpu_info, capability):
    print()
    print("=" * 60)
    print("NOTIFYING RAILWAY")
    print("=" * 60)

    payload = {
        "gpu": gpu_info,
        "cuda_available": True,
        "compute_capability": list(capability)
        if capability
        else None,
    }

    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/worker-ready"
    )

    for attempt in range(1, 11):

        response = post(
            path,
            payload,
            retries=1,
        )

        if response is not None:

            if 200 <= response.status_code < 300:

                print(
                    "WORKER READY accepted by Railway.",
                    flush=True,
                )

                return True

        print(
            f"worker-ready retry {attempt}/10",
            flush=True,
        )

        time.sleep(3)

    print(
        "Failed to notify Railway.",
        flush=True,
    )

    return False


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat():
    path = (
        f"/gpu/session/"
        f"{SESSION_ID}/heartbeat"
    )

    response = post(
        path,
        {
            "timestamp": time.time(),
            "status": "alive",
        },
        retries=2,
    )

    if response is None:
        return False

    return 200 <= response.status_code < 300


# ============================================================
# COMMAND
# ============================================================

def get_command():
    path = (
        f"/internal/session/"
        f"{SESSION_ID}/command"
    )

    response = get(
        path,
        retries=2,
    )

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

    return data


# ============================================================
# COMMAND EXECUTION
# ============================================================

def execute_command(command):
    print()
    print("=" * 60)
    print("COMMAND")
    print("=" * 60)

    print(
        json.dumps(
            command,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    command_id = command.get("command_id")

    command_type = command.get(
        "type",
        "shell",
    )

    if command_type == "shell":

        cmd = command.get("command")

        if not cmd:
            return {
                "success": False,
                "error": "command is missing",
            }

        print(
            f"Executing: {cmd}",
            flush=True,
        )

        try:

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=900,
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[-20000:],
                "stderr": result.stderr[-20000:],
            }

        except subprocess.TimeoutExpired:

            return {
                "success": False,
                "error": "command timeout",
            }

        except Exception as e:

            return {
                "success": False,
                "error": repr(e),
            }

    return {
        "success": False,
        "error": f"unknown command type: {command_type}",
    }


# ============================================================
# SEND RESULT
# ============================================================

def send_result(command, result):
    command_id = command.get("command_id")

    path = (
        f"/internal/session/"
        f"{SESSION_ID}/result"
    )

    payload = {
        "command_id": command_id,
        "result": result,
    }

    post(
        path,
        payload,
        retries=3,
    )


# ============================================================
# KEEP ALIVE LOOP
# ============================================================

def command_loop():
    print()
    print("=" * 60)
    print("COMMAND LOOP")
    print("=" * 60)

    last_heartbeat = 0

    while True:

        now = time.time()

        # ----------------------------------------------------
        # HEARTBEAT
        # ----------------------------------------------------

        if now - last_heartbeat >= HEARTBEAT_INTERVAL:

            ok = heartbeat()

            if ok:
                print(
                    "[KEEP-ALIVE] heartbeat OK",
                    flush=True,
                )
            else:
                print(
                    "[KEEP-ALIVE] heartbeat failed",
                    flush=True,
                )

            last_heartbeat = now

        # ----------------------------------------------------
        # COMMAND
        # ----------------------------------------------------

        command = get_command()

        if command:

            print(
                "[COMMAND] received",
                flush=True,
            )

            try:

                result = execute_command(
                    command
                )

                send_result(
                    command,
                    result,
                )

            except Exception as e:

                print(
                    "Command execution error:",
                    repr(e),
                    flush=True,
                )

                send_result(
                    command,
                    {
                        "success": False,
                        "error": repr(e),
                    },
                )

        time.sleep(COMMAND_INTERVAL)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("KAGGLE GPU WORKER")
    print("=" * 60)

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

    print("=" * 60)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    nvidia_smi()

    gpu_info = get_gpu_info()

    print(
        f"GPU: {gpu_info}",
        flush=True,
    )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    cuda_ok, capability = cuda_test()

    if not cuda_ok:

        raise RuntimeError(
            "CUDA/GPU detection failed"
        )

    # --------------------------------------------------------
    # READY
    # --------------------------------------------------------

    ready = notify_worker_ready(
        gpu_info,
        capability,
    )

    if not ready:

        raise RuntimeError(
            "Failed to notify Railway that worker is ready"
        )

    # --------------------------------------------------------
    # KEEP ALIVE + COMMAND LOOP
    # --------------------------------------------------------

    command_loop()


if __name__ == "__main__":
    main()
