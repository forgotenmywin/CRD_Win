import sys
import traceback

# همه چی رو توی فایل بنویس که حتی اگه crash کنه، لاگ داشته باشیم
f = open('/kaggle/working/debug_log.txt', 'w')
sys.stdout = f
sys.stderr = f

print("=== START ===")

try:
    import subprocess
    print("Step 1: Checking GPU...")
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    print("GPU:", smi.stdout.strip())
    print("GPU stderr:", smi.stderr.strip() if smi.stderr else "none")
except Exception as e:
    print("nvidia-smi failed:", e)

try:
    print("\nStep 2: Trying Numba CUDA...")
    import numpy as np
    from numba import cuda
    
    print("Numba imported OK")
    print("CUDA available:", cuda.is_available())
    
    @cuda.jit
    def add_kernel(a, b, c):
        i = cuda.grid(1)
        if i < c.size:
            c[i] = a[i] + b[i]
    
    n = 10000000
    print(f"Allocating {n} elements...")
    a = cuda.to_device(np.ones(n, dtype=np.float32))
    b = cuda.to_device(np.ones(n, dtype=np.float32))
    c = cuda.device_array(n, dtype=np.float32)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    
    print("Running kernel...")
    add_kernel[blocks, threads](a, b, c)
    cuda.synchronize()
    
    print("SUCCESS! Numba CUDA worked.")
    
except Exception as e:
    print("Numba failed:", e)
    traceback.print_exc()
    
    print("\nStep 3: Trying CPU fallback...")
    try:
        import numpy as np
        a = np.ones(1000000)
        b = np.ones(1000000)
        c = a + b
        print("CPU fallback OK, sum:", c.sum())
    except Exception as e2:
        print("CPU also failed:", e2)
        traceback.print_exc()

print("\n=== END ===")
f.close()
