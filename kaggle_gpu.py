import subprocess
import sys

# نصب Numba (Kaggle داره ولی مطمئن شو)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "numba"])

import numpy as np
from numba import cuda

# لاگ
f = open('/kaggle/working/output.txt', 'w')

try:
    f.write("STEP 1: GPU Info\n")
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    f.write(f"GPU: {smi.stdout}\n")
    
    f.write("STEP 2: Numba CUDA\n")
    f.write(f"CUDA available: {cuda.is_available()}\n")
    f.write(f"Device: {cuda.get_current_device().name}\n")
    
    f.write("STEP 3: Running kernel\n")
    @cuda.jit
    def add_kernel(a, b, c):
        i = cuda.grid(1)
        if i < c.size:
            c[i] = a[i] + b[i]
    
    n = 10000000
    a = cuda.to_device(np.ones(n, dtype=np.float32))
    b = cuda.to_device(np.ones(n, dtype=np.float32))
    c = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    
    add_kernel[blocks, threads](a, b, c)
    cuda.synchronize()
    
    # Checksum
    result = c.copy_to_host()
    f.write(f"Result sum: {result.sum()}\n")
    f.write("SUCCESS! Numba CUDA works!\n")
    
except Exception as e:
    f.write(f"ERROR: {e}\n")
    import traceback
    f.write(traceback.format_exc())

f.close()
