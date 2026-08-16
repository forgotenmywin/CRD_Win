import os
import time
import subprocess


def main():

    session_id = os.environ.get("SESSION_ID", "unknown")

    print("=" * 60)
    print("GPU WORKER KEEP-ALIVE")
    print("=" * 60)
    print("SESSION:", session_id)

    while True:

        print("")
        print("=" * 60)
        print("WORKER ALIVE")
        print("=" * 60)

        try:
            subprocess.run(
                ["nvidia-smi"],
                check=False,
            )
        except Exception as e:
            print("nvidia-smi:", e)

        print("Waiting for commands...")

        time.sleep(60)


if __name__ == "__main__":
    main()
