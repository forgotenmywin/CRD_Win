#!/usr/bin/env python3
"""
Kaggle GPU Worker
Connects to an external API and executes tasks on Kaggle's free GPU.
"""

import os
import sys
import json
import time
import signal
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("KaggleWorker")

# Global flag for graceful shutdown
running = True


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Config loaded: {config.get('session_id', 'unknown')}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)


def verify_gpu() -> bool:
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✅ GPU Available: {device_name} ({memory:.1f} GB)")
            return True
        else:
            logger.warning("⚠️ GPU not available, using CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not installed, cannot verify GPU")
        return False


class KaggleWorker:
    def __init__(self, config: Dict[str, Any]):
        self.api_url = config["api_url"].rstrip("/")
        self.session_id = config["session_id"]
        self.worker_token = config["worker_token"]
        self.kernel_name = config.get("kernel_name", "unknown")
        
        self.headers = {
            "Authorization": f"Bearer {self.worker_token}",
            "Content-Type": "application/json",
            "X-Session-ID": self.session_id,
            "X-Worker-Type": "kaggle-gpu"
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        self.heartbeat_interval = 30  # seconds
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
        
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()

    def _get_url(self, endpoint: str) -> str:
        return f"{self.api_url}{endpoint}"

    def register(self) -> bool:
        """Register worker with the API"""
        try:
            payload = {
                "session_id": self.session_id,
                "kernel_name": self.kernel_name,
                "status": "ready",
                "gpu_available": verify_gpu()
            }
            
            response = self.session.post(
                self._get_url(f"/gpu/worker/{self.session_id}/register"),
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ Worker registered successfully")
                return True
            else:
                logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    def heartbeat(self):
        """Send periodic heartbeats to keep session alive"""
        while not self._stop_heartbeat.is_set() and running:
            try:
                payload = {
                    "session_id": self.session_id,
                    "timestamp": time.time(),
                    "status": "alive"
                }
                
                response = self.session.post(
                    self._get_url(f"/gpu/worker/{self.session_id}/heartbeat"),
                    json=payload,
                    timeout=10
                )
                
                if response.status_code != 200:
                    logger.warning(f"Heartbeat failed: {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
            
            # Wait with interrupt support
            self._stop_heartbeat.wait(self.heartbeat_interval)

    def start_heartbeat(self):
        """Start heartbeat in background thread"""
        self._heartbeat_thread = threading.Thread(target=self.heartbeat, daemon=True)
        self._heartbeat_thread.start()
        logger.info("Heartbeat thread started")

    def stop_heartbeat(self):
        """Stop heartbeat thread"""
        self._stop_heartbeat.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)

    def fetch_task(self) -> Optional[Dict[str, Any]]:
        """Fetch next task from API"""
        try:
            response = self.session.get(
                self._get_url(f"/gpu/worker/{self.session_id}/task"),
                timeout=30
            )
            
            if response.status_code == 204:
                return None  # No task available
            
            if response.status_code == 200:
                return response.json()
                
            logger.warning(f"Unexpected status fetching task: {response.status_code}")
            return None
            
        except requests.exceptions.Timeout:
            logger.warning("Task fetch timeout")
            return None
        except Exception as e:
            logger.error(f"Error fetching task: {e}")
            return None

    def submit_result(self, task_id: str, result: Dict[str, Any], error: Optional[str] = None):
        """Submit task result back to API"""
        try:
            payload = {
                "task_id": task_id,
                "session_id": self.session_id,
                "result": result,
                "error": error,
                "timestamp": time.time()
            }
            
            response = self.session.post(
                self._get_url(f"/gpu/worker/{self.session_id}/result"),
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Result submitted for task {task_id}")
            else:
                logger.error(f"Failed to submit result: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error submitting result: {e}")

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task (override this for custom logic)"""
        task_type = task.get("type", "unknown")
        task_id = task.get("id", "unknown")
        
        logger.info(f"Executing task {task_id} of type '{task_type}'")
        
        try:
            # Example: Run Python code
            if task_type == "execute":
                code = task.get("code", "")
                # WARNING: exec is dangerous - use only with trusted API
                local_vars = {}
                exec(code, {"__builtins__": __builtins__}, local_vars)
                return {"status": "completed", "output": local_vars}
            
            # Example: GPU benchmark
            elif task_type == "benchmark":
                return self._run_benchmark()
            
            # Example: Health check
            elif task_type == "health":
                return {
                    "status": "healthy",
                    "gpu": verify_gpu(),
                    "uptime": time.time() - start_time
                }
            
            else:
                return {"status": "unknown_task_type", "type": task_type}
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise

    def _run_benchmark(self) -> Dict[str, Any]:
        """Run a simple GPU benchmark"""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"error": "GPU not available"}
            
            # Simple matrix multiplication benchmark
            size = 5000
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
                "time_seconds": round(elapsed, 3),
                "device": torch.cuda.get_device_name(0)
            }
        except Exception as e:
            return {"error": str(e)}

    def run(self):
        """Main worker loop"""
        logger.info("=" * 50)
        logger.info("KAGGLE GPU WORKER STARTING")
        logger.info(f"Session: {self.session_id}")
        logger.info(f"API: {self.api_url}")
        logger.info("=" * 50)
        
        # Register with API
        if not self.register():
            logger.error("Failed to register worker. Exiting.")
            return
        
        # Start heartbeat
        self.start_heartbeat()
        
        # Main loop
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while running:
            try:
                task = self.fetch_task()
                
                if task is None:
                    time.sleep(2)
                    consecutive_errors = 0
                    continue
                
                # Execute task
                task_id = task.get("id", "unknown")
                result = self.execute_task(task)
                self.submit_result(task_id, result)
                consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Main loop error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many consecutive errors. Exiting.")
                    break
                    
                time.sleep(self.reconnect_delay)
        
        # Cleanup
        self.stop_heartbeat()
        logger.info("Worker stopped gracefully")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global running
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    running = False


# Track start time
start_time = time.time()

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load config
    config = load_config()
    
    # Create and run worker
    worker = KaggleWorker(config)
    worker.run()
    
    logger.info("Worker process exiting")
