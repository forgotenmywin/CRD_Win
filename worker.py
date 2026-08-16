#!/usr/bin/env python3
"""
Kaggle GPU Worker
Connects to external API, reports GPU status, and executes commands.
"""

import os
import sys
import json
import time
import signal
import logging
import traceback
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("KaggleWorker")

running = True


def load_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def get_gpu_info() -> Dict[str, Any]:
    """Detect GPU using multiple methods"""
    info = {"available": False, "name": None, "memory_mb": None}
    
    # Try PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            info["memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            return info
    except ImportError:
        pass
    
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                info["available"] = True
                info["name"] = parts[0].strip()
                mem_str = parts[1].strip().replace(" MiB", "").replace(" MB", "")
                info["memory_mb"] = float(mem_str)
                return info
    except Exception:
        pass
    
    return info


class KaggleWorker:
    def __init__(self, config: Dict[str, Any]):
        self.api_url = config["api_url"].rstrip("/")
        self.session_id = config["session_id"]
        self.token = config["worker_token"]
        self.kernel_name = config.get("kernel_name", "unknown")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-Session-ID": self.session_id
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.poll_interval = 3
        self.heartbeat_interval = 15

    def _post(self, endpoint: str, payload: Dict = None, timeout: int = 10) -> Optional[requests.Response]:
        try:
            return self.session.post(f"{self.api_url}{endpoint}", json=payload or {}, timeout=timeout)
        except Exception as e:
            logger.debug(f"POST {endpoint} failed: {e}")
            return None

    def _get(self, endpoint: str, timeout: int = 10) -> Optional[requests.Response]:
        try:
            return self.session.get(f"{self.api_url}{endpoint}", timeout=timeout)
        except Exception as e:
            logger.debug(f"GET {endpoint} failed: {e}")
            return None

    def register(self, gpu_info: Dict[str, Any]) -> bool:
        """Tell API that worker is ready"""
        endpoints = [
            f"/gpu/session/{self.session_id}/worker_ready",
            f"/gpu/session/{self.session_id}/ready",
            f"/gpu/worker/{self.session_id}/register",
        ]
        
        for url in endpoints:
            r = self._post(url, {"gpu": gpu_info, "kernel": self.kernel_name, "status": "ready"})
            if r and r.status_code in (200, 201, 202):
                logger.info(f"✅ Registered at {url}")
                return True
        
        logger.warning("Could not register worker (API may not need it)")
        return False

    def heartbeat(self):
        """Send periodic heartbeat"""
        endpoints = [
            f"/gpu/session/{self.session_id}/heartbeat",
            f"/gpu/session/{self.session_id}/ping",
            f"/gpu/worker/{self.session_id}/heartbeat",
        ]
        
        while running:
            for url in endpoints:
                r = self._post(url, {"status": "alive", "timestamp": time.time()}, timeout=5)
                if r and r.status_code in (200, 202, 204):
                    break
            time.sleep(self.heartbeat_interval)

    def fetch_task(self) -> Optional[Dict[str, Any]]:
        """Poll API for pending tasks/commands"""
        endpoints = [
            f"/gpu/session/{self.session_id}/poll",
            f"/gpu/session/{self.session_id}/next_command",
            f"/gpu/worker/{self.session_id}/task",
        ]
        
        for url in endpoints:
            r = self._get(url, timeout=10)
            if r and r.status_code == 200:
                data = r.json()
                if data and not (isinstance(data, dict) and data.get("status") == "no_task"):
                    return data
        return None

    def submit_result(self, task_id: str, result: Any, error: str = None):
        """Send task result back to API"""
        endpoints = [
            f"/gpu/session/{self.session_id}/command/{task_id}/result",
            f"/gpu/session/{self.session_id}/result",
            f"/gpu/worker/{self.session_id}/result",
        ]
        
        payload = {
            "task_id": task_id,
            "result": result,
            "error": error,
            "timestamp": time.time()
        }
        
        for url in endpoints:
            r = self._post(url, payload, timeout=30)
            if r and r.status_code in (200, 202):
                logger.info(f"✅ Result submitted for task {task_id}")
                return True
        
        logger.error(f"Failed to submit result for task {task_id}")
        return False

    def execute_command(self, task: Dict[str, Any]) -> Any:
        """Execute a task/command"""
        op = task.get("operation") or task.get("type") or "unknown"
        task_id = task.get("id") or task.get("command_id") or "unknown"
        
        logger.info(f"Executing task {task_id}: {op}")
        
        try:
            if op in ("info", "system_info"):
                return {
                    "gpu": get_gpu_info(),
                    "python_version": sys.version,
                    "platform": sys.platform
                }
            
            elif op in ("execute", "run", "exec"):
                code = task.get("code", task.get("script", ""))
                if not code:
                    return {"error": "No code provided"}
                
                local_vars = {}
                exec(code, {"__builtins__": __builtins__}, local_vars)
                return {"status": "completed", "locals": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
            
            elif op in ("benchmark", "gpu_benchmark"):
                return self._gpu_benchmark()
            
            elif op in ("health", "ping"):
                return {"status": "healthy", "gpu": get_gpu_info()}
            
            else:
                return {"error": f"Unknown operation: {op}"}
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _gpu_benchmark(self) -> Dict[str, Any]:
        try:
            import torch
            if not torch.cuda.is_available():
                return {"error": "GPU not available"}
            
            size = 4096
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            
            torch.cuda.synchronize()
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            return {
                "benchmark": "matmul",
                "size": size,
                "time_seconds": round(elapsed, 4),
                "device": torch.cuda.get_device_name(0),
                "flops_approx": round(2 * (size ** 3) / elapsed / 1e12, 2)  # TFLOPS
            }
        except ImportError:
            return {"error": "PyTorch not installed"}
        except Exception as e:
            return {"error": str(e)}

    def run(self):
        logger.info("=" * 50)
        logger.info("KAGGLE GPU WORKER STARTING")
        logger.info(f"Session: {self.session_id}")
        logger.info(f"API: {self.api_url}")
        logger.info("=" * 50)
        
        gpu_info = get_gpu_info()
        logger.info(f"GPU Info: {gpu_info}")
        
        # Register with API
        self.register(gpu_info)
        
        # Start heartbeat in background
        hb_thread = threading.Thread(target=self.heartbeat, daemon=True)
        hb_thread.start()
        
        # Main loop
        consecutive_errors = 0
        
        while running:
            try:
                task = self.fetch_task()
                
                if task:
                    task_id = task.get("id") or task.get("command_id") or "unknown"
                    result = self.execute_command(task)
                    self.submit_result(task_id, result)
                    consecutive_errors = 0
                else:
                    time.sleep(self.poll_interval)
                    consecutive_errors = 0
                    
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Main loop error ({consecutive_errors}): {e}")
                if consecutive_errors >= 20:
                    logger.error("Too many errors, exiting")
                    break
                time.sleep(5)
        
        logger.info("Worker stopped")


def signal_handler(signum, frame):
    global running
    logger.info(f"Received signal {signum}, shutting down...")
    running = False


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    config = load_config()
    worker = KaggleWorker(config)
    worker.run()
