import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


USERNAME = os.environ["KAGGLE_USERNAME"]
KERNEL = f"{USERNAME}/notebookd9d7092a0a"

WORK_DIR = Path("kaggle_kernel")
OUTPUT_DIR = Path("kaggle_output")


def run(command):
    print(f"\n>>> {command}")

    result = subprocess.run(
        command,
        shell=True,
        text=True
    )

    if result.returncode != 0:
        sys.exit(result.returncode)


print("======================================")
print("       KAGGLE GPU GENERIC WORKER")
print("======================================")


# --------------------------------------------------
# 1. Clean
# --------------------------------------------------

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

WORK_DIR.mkdir()
OUTPUT_DIR.mkdir()


# --------------------------------------------------
# 2. Pull existing Kaggle Notebook
# --------------------------------------------------

print("\n[1] Pulling Kaggle Notebook...")

run(
    f"python -m kaggle kernels pull "
    f"{KERNEL} "
    f"-p {WORK_DIR} "
    f"-m"
)


# --------------------------------------------------
# 3. Read job.py
# --------------------------------------------------

print("\n[2] Reading GPU job...")

job_code = Path("job.py").read_text(
    encoding="utf-8"
)

print("Job loaded.")


# --------------------------------------------------
# 4. Find notebook
# --------------------------------------------------

notebook_file = next(
    WORK_DIR.glob("*.ipynb"),
    None
)

if notebook_file is None:
    sys.exit("Notebook file not found")


print("Notebook:", notebook_file)


# --------------------------------------------------
# 5. Replace notebook cells
# --------------------------------------------------

print("\n[3] Preparing GPU Worker...")


notebook = json.loads(
    notebook_file.read_text(
        encoding="utf-8"
    )
)


runner_code = f'''
# ==========================================
# GENERIC KAGGLE GPU WORKER
# ==========================================

import os
import sys
import time

print("======================================")
print("       KAGGLE GPU WORKER")
print("======================================")

print("Python:", sys.version)

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
    else:
        print("WARNING: CUDA is not available")

except Exception as e:
    print("Torch check error:", e)


print("\\n======================================")
print("           RUNNING JOB")
print("======================================")

{job_code}

print("\\n======================================")
print("          JOB FINISHED")
print("======================================")
'''


notebook["cells"] = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": runner_code.splitlines(True)
    }
]


# --------------------------------------------------
# 6. Force GPU
# --------------------------------------------------

metadata_file = WORK_DIR / "kernel-metadata.json"

metadata = json.loads(
    metadata_file.read_text(
        encoding="utf-8"
    )
)

metadata["enable_gpu"] = True
metadata["machine_shape"] = "Gpu"


metadata_file.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


notebook_file.write_text(
    json.dumps(
        notebook,
        indent=2
    ),
    encoding="utf-8"
)


print("GPU Worker prepared.")


# --------------------------------------------------
# 7. Push to Kaggle
# --------------------------------------------------

print("\n[4] Starting Kaggle GPU Job...")

run(
    f"python -m kaggle kernels push "
    f"-p {WORK_DIR}"
)


# --------------------------------------------------
# 8. Wait
# --------------------------------------------------

print("\n[5] Waiting for GPU...")

for i in range(120):

    print()
    print("======================================")
    print(f"Status check {i + 1}/120")
    print("======================================")


    result = subprocess.run(
        f"python -m kaggle kernels status {KERNEL}",
        shell=True,
        capture_output=True,
        text=True
    )

    status = result.stdout + result.stderr

    print(status)


    if "COMPLETE" in status.upper():
        print("\nGPU job completed.")
        break


    if any(
        x in status.upper()
        for x in [
            "ERROR",
            "FAILED",
            "CANCELLED"
        ]
    ):
        print("\nGPU job failed.")
        sys.exit(1)


    time.sleep(10)


else:
    print("\nGPU job timeout.")
    sys.exit(1)


# --------------------------------------------------
# 9. Download output
# --------------------------------------------------

print("\n[6] Downloading output...")

run(
    f"python -m kaggle kernels output "
    f"{KERNEL} "
    f"-p {OUTPUT_DIR}"
)


print("\n======================================")
print("           WORKER COMPLETE")
print("======================================")

for file in OUTPUT_DIR.rglob("*"):
    if file.is_file():
        print("OUTPUT:", file)
