import os
import time
import signal
import sys

SESSION_ID = os.environ.get("SESSION_ID", "unknown")
MAX_SESSION_MINUTES = int(os.environ.get("MAX_SESSION_MINUTES", "20"))

running = True


def stop_handler(signum, frame):
    global running
    print("\nStopping GPU session...")
    running = False


signal.signal(signal.SIGTERM, stop_handler)
signal.signal(signal.SIGINT, stop_handler)


print("======================================")
print("       KAGGLE GPU SESSION")
print("======================================")
print(f"Session ID: {SESSION_ID}")
print(f"Maximum time: {MAX_SESSION_MINUTES} minutes")
print()

try:
    import torch

    if not torch.cuda.is_available():
        print("GPU unavailable")
        print("Reason: CUDA is not available")
        sys.exit(2)

    gpu = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu}")
    print("CUDA: available")
    print()
    print("GPU SESSION STARTED")

except Exception as e:
    print("GPU initialization failed")
    print(f"Reason: {e}")
    sys.exit(3)


start = time.time()
limit = MAX_SESSION_MINUTES * 60

while running:
    elapsed = time.time() - start

    if elapsed >= limit:
        print()
        print("Session time limit reached.")
        break

    remaining = int(limit - elapsed)

    print(
        f"SESSION ACTIVE | "
        f"GPU={gpu} | "
        f"remaining={remaining}s"
    )

    time.sleep(10)


print()
print("======================================")
print("       GPU SESSION STOPPED")
print("======================================")
