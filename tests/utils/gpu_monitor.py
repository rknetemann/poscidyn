# gpu_monitor.py
import threading
import time
from typing import List, Dict, Optional

class GpuMonitor:
    """
    Background GPU monitor. Prefers NVML (pynvml) for low overhead, falls back to nvidia-smi.
    Use as a context manager:
        with GpuMonitor(interval=1.0) as gm:
            ...
            pbar.set_postfix_str(gm.summary())
            # or for max values:
            pbar.set_postfix_str(gm.max_summary())
    """
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._summary: str = ""
        self._stats: List[Dict] = []
        self._max_stats: List[Dict] = []  # Store maximum values
        self._max_summary: str = ""
        self._thread: Optional[threading.Thread] = None
        self._use_nvml = False
        self._nvml = None
        self._init_backend()

    def _init_backend(self):
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            # Make sure at least one device is present
            if pynvml.nvmlDeviceGetCount() > 0:
                self._use_nvml = True
                self._nvml = pynvml
            else:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            self._use_nvml = False
            self._nvml = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._use_nvml and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def stats(self) -> List[Dict]:
        with self._lock:
            return list(self._stats)
            
    def max_stats(self) -> List[Dict]:
        """Return the maximum GPU stats observed since monitoring began."""
        with self._lock:
            return list(self._max_stats)

    def summary(self) -> str:
        with self._lock:
            return self._summary
            
    def max_summary(self) -> str:
        """Return a summary string of the maximum GPU stats observed."""
        with self._lock:
            return self._max_summary

    # --- internals ---

    def _update_max_stats(self, current_stats: List[Dict]):
        """Update maximum stats based on current values."""
        if not self._max_stats:
            # Initialize max_stats with the first reading
            self._max_stats = [stat.copy() for stat in current_stats]
            return
            
        # Ensure max_stats has entries for all GPUs
        gpu_indices = {stat["index"] for stat in current_stats}
        max_indices = {stat["index"] for stat in self._max_stats}
        
        # Add any missing GPUs to max_stats
        for idx in gpu_indices - max_indices:
            for stat in current_stats:
                if stat["index"] == idx:
                    self._max_stats.append(stat.copy())
                    break
        
        # Update max values
        for current in current_stats:
            for i, max_stat in enumerate(self._max_stats):
                if max_stat["index"] == current["index"]:
                    max_stat["util"] = max(max_stat["util"], current["util"])
                    max_stat["used_gb"] = max(max_stat["used_gb"], current["used_gb"])
        
        # Generate max summary string
        parts = []
        for stat in sorted(self._max_stats, key=lambda x: x["index"]):
            i = stat["index"]
            util = stat["util"]
            used_gb = stat["used_gb"]
            total_gb = stat["total_gb"]
            parts.append(f"GPU{i}: {util}% {used_gb:.1f}GB/{total_gb:.1f}GB")
        self._max_summary = " | ".join(parts)

    def _run(self):
        while not self._stop.is_set():
            try:
                if self._use_nvml:
                    self._poll_nvml()
                else:
                    self._poll_nvidia_smi()
            except Exception:
                # keep the monitor resilient
                pass
            finally:
                self._stop.wait(self.interval)

    def _poll_nvml(self):
        nvml = self._nvml
        assert nvml is not None
        n = nvml.nvmlDeviceGetCount()
        stats = []
        parts = []
        for i in range(n):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            util = nvml.nvmlDeviceGetUtilizationRates(h).gpu  # percent
            mem = nvml.nvmlDeviceGetMemoryInfo(h)
            used_gb = mem.used / (1024**3)
            total_gb = mem.total / (1024**3)
            stats.append({"index": i, "util": util, "used_gb": used_gb, "total_gb": total_gb})
            parts.append(f"GPU{i} {util}% {used_gb:.1f}GB/{total_gb:.1f}GB")
        summary = " | ".join(parts)
        with self._lock:
            self._stats = stats
            self._summary = summary
            self._update_max_stats(stats)

    def _poll_nvidia_smi(self):
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8"
        )
        stats = []
        parts = []
        for line in out.strip().splitlines():
            idx_s, util_s, used_mb_s, total_mb_s = [x.strip() for x in line.split(",")]
            idx = int(idx_s)
            util = int(util_s)
            used_gb = float(used_mb_s) / 1024.0
            total_gb = float(total_mb_s) / 1024.0
            stats.append({"index": idx, "util": util, "used_gb": used_gb, "total_gb": total_gb})
            parts.append(f"GPU{idx} {util}% {used_gb:.1f}GB/{total_gb:.1f}GB")
        summary = " | ".join(parts)
        with self._lock:
            self._stats = stats
            self._summary = summary
            self._update_max_stats(stats)
