import json
import os
import subprocess
import sys
import time
from pathlib import Path

KERNEL = "darytredf/notebookd9d7092a0a"
WORK_DIR = Path("kaggle_kernel")


def run(command, check=True):
    print(f"\n>>> {command}")
    result = subprocess.run(
        command,
        shell=True,
        text=True
    )

    if check and result.returncode != 0:
        sys.exit(result.returncode)

    return result.returncode


print("======================================")
print("      Kaggle GPU Worker Controller")
print("======================================")

# --------------------------------------------------
# 1. Check credentials
# --------------------------------------------------

if not os.getenv("KAGGLE_USERNAME"):
    sys.exit("KAGGLE_USERNAME is missing")

if not os.getenv("KAGGLE_API_TOKEN"):
    sys.exit("KAGGLE_API_TOKEN is missing")

print("\n[OK] Kaggle credentials found")


# --------------------------------------------------
# 2. Clean old files
# --------------------------------------------------

if WORK_DIR.exists():
    import shutil
    shutil.rmtree(WORK_DIR)

WORK_DIR.mkdir(parents=True)


# --------------------------------------------------
# 3. Download existing Kaggle notebook
# --------------------------------------------------

print("\n[1/5] Pulling Kaggle notebook...")

run(
    f"kaggle kernels pull {KERNEL} "
    f"-p {WORK_DIR} -m"
)


# --------------------------------------------------
# 4. Enable GPU in metadata
# --------------------------------------------------

print("\n[2/5] Preparing GPU configuration...")

metadata_file = WORK_DIR / "kernel-metadata.json"

if not metadata_file.exists():
    sys.exit("kernel-metadata.json was not found")

with open(metadata_file, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata["enable_gpu"] = True

with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("[OK] GPU enabled")


# --------------------------------------------------
# 5. Push notebook back to Kaggle and run it
# --------------------------------------------------

print("\n[3/5] Starting Kaggle GPU job...")

run(
    f"kaggle kernels push "
    f"-p {WORK_DIR} "
    f"--accelerator NvidiaTeslaP100"
)

print("\n[OK] Kaggle job submitted")


# --------------------------------------------------
# 6. Wait for completion
# --------------------------------------------------

print("\n[4/5] Waiting for Kaggle...")

for i in range(60):

    print(f"\n--- Status check {i + 1}/60 ---")

    result = subprocess.run(
        f"kaggle kernels status {KERNEL}",
        shell=True,
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    print(output)

    lower = output.lower()

    if "complete" in lower:
        print("[OK] Kaggle job completed")
        break

    if "error" in lower or "failed" in lower:
        print("[ERROR] Kaggle job failed")
        sys.exit(1)

    time.sleep(10)

else:
    print("[ERROR] Timeout waiting for Kaggle")
    sys.exit(1)


# --------------------------------------------------
# 7. Download output
# --------------------------------------------------

print("\n[5/5] Downloading Kaggle output...")

run(
    f"kaggle kernels output {KERNEL} "
    f"-p kaggle_output"
)

print("\n======================================")
print("          GPU JOB FINISHED")
print("======================================")

print("\nOutput files:")

if Path("kaggle_output").exists():
    for file in Path("kaggle_output").rglob("*"):
        if file.is_file():
            print(" -", file)
