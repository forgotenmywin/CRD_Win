import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ==================================================
# CONFIG
# ==================================================

USERNAME = os.environ["KAGGLE_USERNAME"]
KERNEL = f"{USERNAME}/notebookd9d7092a0a"

WORK_DIR = Path("kaggle_kernel")
OUTPUT_DIR = Path("kaggle_output")


# ==================================================
# COMMAND
# ==================================================

def run(command, check=True):

    print(f"\n>>> {command}", flush=True)

    result = subprocess.run(
        command,
        shell=True,
        text=True
    )

    if check and result.returncode != 0:
        print(
            f"Command failed: {result.returncode}",
            flush=True
        )
        sys.exit(result.returncode)

    return result.returncode


# ==================================================
# START
# ==================================================

print("======================================")
print("       KAGGLE GPU GENERIC WORKER")
print("======================================")

if not os.getenv("KAGGLE_USERNAME"):
    sys.exit("KAGGLE_USERNAME is missing")

if not os.getenv("KAGGLE_API_TOKEN"):
    sys.exit("KAGGLE_API_TOKEN is missing")

print("[OK] Kaggle credentials found")


# ==================================================
# CLEAN
# ==================================================

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

WORK_DIR.mkdir(parents=True)
OUTPUT_DIR.mkdir(parents=True)


# ==================================================
# PULL NOTEBOOK
# ==================================================

print("\n[1] Pulling Kaggle Notebook...")

run(
    f"python -m kaggle kernels pull "
    f"{KERNEL} "
    f"-p {WORK_DIR} "
    f"-m"
)


# ==================================================
# FIND NOTEBOOK
# ==================================================

notebooks = list(
    WORK_DIR.glob("*.ipynb")
)

if not notebooks:
    sys.exit("Notebook .ipynb not found")

notebook_file = notebooks[0]

print(
    "[OK] Notebook:",
    notebook_file,
    flush=True
)


# ==================================================
# READ JOB
# ==================================================

job_file = Path("job.py")

if not job_file.exists():
    sys.exit("job.py not found")

job_code = job_file.read_text(
    encoding="utf-8"
)

print("[OK] job.py loaded")


# ==================================================
# LOAD NOTEBOOK
# ==================================================

notebook = json.loads(
    notebook_file.read_text(
        encoding="utf-8"
    )
)


# ==================================================
# KAGGLE GPU WORKER CODE
# ==================================================

worker_code = r'''
print("======================================")
print("       KAGGLE GPU WORKER START")
print("======================================")

import sys
import subprocess

print("Python:", sys.version)


# --------------------------------------------------
# Install P100-compatible PyTorch
# --------------------------------------------------

print("")
print("Installing PyTorch 2.3.1 + CUDA 11.8...")
print("")


subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    "--force-reinstall",
    "torch==2.3.1",
    "torchvision==0.18.1",
    "torchaudio==2.3.1",
    "--index-url",
    "https://download.pytorch.org/whl/cu118"
])


# --------------------------------------------------
# IMPORTANT:
# Reload torch after installation
# --------------------------------------------------

print("")
print("PyTorch installation completed.")
print("")


import torch


# --------------------------------------------------
# GPU information
# --------------------------------------------------

print("======================================")
print("             GPU CHECK")
print("======================================")

print("PyTorch:", torch.__version__)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "CUDA version:",
    torch.version.cuda
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is not available"
    )


gpu_name = torch.cuda.get_device_name(0)

print(
    "GPU:",
    gpu_name
)


print(
    "Compute capability:",
    torch.cuda.get_device_capability(0)
)


print(
    "Supported architectures:",
    torch.cuda.get_arch_list()
)


# --------------------------------------------------
# Real GPU test
# --------------------------------------------------

print("")
print("======================================")
print("          REAL GPU TEST")
print("======================================")

size = 2048

print(
    f"Creating {size}x{size} tensors..."
)


x = torch.randn(
    size,
    size,
    device="cuda"
)


y = torch.randn(
    size,
    size,
    device="cuda"
)


torch.cuda.synchronize()


print("Running matrix multiplication...")


import time

start = time.perf_counter()


z = torch.matmul(
    x,
    y
)


torch.cuda.synchronize()


elapsed = (
    time.perf_counter()
    - start
)


print("")
print("======================================")
print("         GPU TEST SUCCESS")
print("======================================")

print(
    "GPU:",
    gpu_name
)

print(
    "Result shape:",
    z.shape
)

print(
    "Time:",
    elapsed,
    "seconds"
)


# --------------------------------------------------
# USER JOB
# --------------------------------------------------

print("")
print("======================================")
print("          RUNNING USER JOB")
print("======================================")

''' + job_code + r'''

print("")
print("======================================")
print("          USER JOB FINISHED")
print("======================================")
'''


# ==================================================
# REPLACE NOTEBOOK CELLS
# ==================================================

notebook["cells"] = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": worker_code.splitlines(
            keepends=True
        )
    }
]


# ==================================================
# GPU METADATA
# ==================================================

metadata_file = (
    WORK_DIR /
    "kernel-metadata.json"
)

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


# ==================================================
# SAVE NOTEBOOK
# ==================================================

notebook_file.write_text(
    json.dumps(
        notebook,
        indent=2
    ),
    encoding="utf-8"
)


print("")
print("[OK] Notebook prepared")


# ==================================================
# PUSH
# ==================================================

print("")
print("[2] Pushing GPU Worker to Kaggle...")


push_result = run(
    f"python -m kaggle kernels push "
    f"-p {WORK_DIR}",
    check=False
)


if push_result != 0:
    sys.exit(push_result)


print("")
print("[OK] Kaggle GPU job submitted")


# ==================================================
# WAIT
# ==================================================

print("")
print("[3] Waiting for Kaggle...")


final_status = None


for i in range(120):

    print("")
    print(
        "======================================"
    )

    print(
        f"STATUS {i + 1}/120"
    )

    print(
        "======================================"
    )


    result = subprocess.run(
        f"python -m kaggle kernels status {KERNEL}",
        shell=True,
        capture_output=True,
        text=True
    )


    status = (
        result.stdout +
        result.stderr
    )


    print(
        status,
        flush=True
    )


    upper = status.upper()


    if "COMPLETE" in upper:

        final_status = "COMPLETE"

        break


    if "ERROR" in upper:

        final_status = "ERROR"

        break


    if "FAILED" in upper:

        final_status = "FAILED"

        break


    if "CANCELLED" in upper:

        final_status = "CANCELLED"

        break


    time.sleep(10)


# ==================================================
# DOWNLOAD OUTPUT
# ==================================================

print("")
print("[4] Downloading Kaggle output...")


subprocess.run(
    f"python -m kaggle kernels output "
    f"{KERNEL} "
    f"-p {OUTPUT_DIR}",
    shell=True,
    text=True
)


# ==================================================
# LIST FILES
# ==================================================

print("")
print("======================================")
print("           KAGGLE OUTPUT")
print("======================================")


for file in OUTPUT_DIR.rglob("*"):

    if file.is_file():

        print(
            "OUTPUT:",
            file,
            flush=True
        )


# ==================================================
# PRINT LOG
# ==================================================

print("")
print("======================================")
print("           KAGGLE LOG")
print("======================================")


log_files = list(
    OUTPUT_DIR.rglob("*.log")
)


if not log_files:

    print("No log file found.")

else:

    for log_file in log_files:

        print("")
        print(
            f"========== {log_file} =========="
        )

        try:

            content = log_file.read_text(
                encoding="utf-8",
                errors="replace"
            )

            print(
                content,
                flush=True
            )

        except Exception as e:

            print(
                "Could not read log:",
                e,
                flush=True
            )


# ==================================================
# FINAL
# ==================================================

print("")
print("======================================")
print("           FINAL STATUS")
print("======================================")

print(
    "Kaggle:",
    final_status
)


if final_status != "COMPLETE":

    print(
        "\nGPU Worker failed."
    )

    sys.exit(1)


print(
    "\nGPU Worker completed successfully."
)
