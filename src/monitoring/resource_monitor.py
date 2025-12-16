"""System resource monitoring for benchmarks."""
import time
import psutil
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ResourceSnapshot:
    """Single snapshot of system resources."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


@dataclass
class ResourceMetrics:
    """Aggregated resource metrics."""
    duration: float
    cpu_avg: float
    cpu_max: float
    cpu_min: float
    memory_avg_mb: float
    memory_max_mb: float
    memory_min_mb: float
    disk_read_total_mb: float
    disk_write_total_mb: float
    network_sent_total_mb: float
    network_recv_total_mb: float
    snapshots: List[ResourceSnapshot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'duration': self.duration,
            'cpu': {
                'avg': self.cpu_avg,
                'max': self.cpu_max,
                'min': self.cpu_min
            },
            'memory': {
                'avg_mb': self.memory_avg_mb,
                'max_mb': self.memory_max_mb,
                'min_mb': self.memory_min_mb
            },
            'disk': {
                'read_total_mb': self.disk_read_total_mb,
                'write_total_mb': self.disk_write_total_mb
            },
            'network': {
                'sent_total_mb': self.network_sent_total_mb,
                'recv_total_mb': self.network_recv_total_mb
            }
        }


class ResourceMonitor:
    """Monitor system resource usage during benchmarks."""

    def __init__(self, interval: float = 0.5):
        """
        Initialize resource monitor.

        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None

        # Initial values for delta calculations
        self.initial_disk_io: Optional[psutil._common.sdiskio] = None
        self.initial_network_io: Optional[psutil._common.snetio] = None

    def start(self):
        """Start monitoring resources."""
        if self.monitoring:
            return

        self.monitoring = True
        self.start_time = time.time()
        self.snapshots = []

        # Get initial I/O counters
        self.initial_disk_io = psutil.disk_io_counters()
        self.initial_network_io = psutil.net_io_counters()

        # Start monitoring thread
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> ResourceMetrics:
        """
        Stop monitoring and return metrics.

        Returns:
            Aggregated resource metrics
        """
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)

        return self._calculate_metrics()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            snapshot = self._take_snapshot()
            if snapshot:
                self.snapshots.append(snapshot)
            time.sleep(self.interval)

    def _take_snapshot(self) -> Optional[ResourceSnapshot]:
        """Take a single resource snapshot."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent

            # Disk I/O (cumulative)
            disk_io = psutil.disk_io_counters()
            if disk_io and self.initial_disk_io:
                disk_read_mb = (disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024 * 1024)
            else:
                disk_read_mb = 0
                disk_write_mb = 0

            # Network I/O (cumulative)
            network_io = psutil.net_io_counters()
            if network_io and self.initial_network_io:
                network_sent_mb = (network_io.bytes_sent - self.initial_network_io.bytes_sent) / (1024 * 1024)
                network_recv_mb = (network_io.bytes_recv - self.initial_network_io.bytes_recv) / (1024 * 1024)
            else:
                network_sent_mb = 0
                network_recv_mb = 0

            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )

        except Exception as e:
            print(f"Error taking resource snapshot: {e}")
            return None

    def _calculate_metrics(self) -> ResourceMetrics:
        """Calculate aggregated metrics from snapshots."""
        if not self.snapshots:
            # Return empty metrics
            return ResourceMetrics(
                duration=0,
                cpu_avg=0,
                cpu_max=0,
                cpu_min=0,
                memory_avg_mb=0,
                memory_max_mb=0,
                memory_min_mb=0,
                disk_read_total_mb=0,
                disk_write_total_mb=0,
                network_sent_total_mb=0,
                network_recv_total_mb=0,
                snapshots=[]
            )

        # Calculate duration
        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp

        # Extract metrics
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_mb for s in self.snapshots]

        # Get final I/O values (cumulative)
        final_snapshot = self.snapshots[-1]

        return ResourceMetrics(
            duration=duration,
            cpu_avg=np.mean(cpu_values),
            cpu_max=np.max(cpu_values),
            cpu_min=np.min(cpu_values),
            memory_avg_mb=np.mean(memory_values),
            memory_max_mb=np.max(memory_values),
            memory_min_mb=np.min(memory_values),
            disk_read_total_mb=final_snapshot.disk_read_mb,
            disk_write_total_mb=final_snapshot.disk_write_mb,
            network_sent_total_mb=final_snapshot.network_sent_mb,
            network_recv_total_mb=final_snapshot.network_recv_mb,
            snapshots=self.snapshots
        )


class ProcessResourceMonitor(ResourceMonitor):
    """Monitor resources for a specific process."""

    def __init__(self, pid: Optional[int] = None, interval: float = 0.5):
        """
        Initialize process resource monitor.

        Args:
            pid: Process ID (if None, monitors current process)
            interval: Sampling interval in seconds
        """
        super().__init__(interval)
        self.pid = pid or psutil.Process().pid
        self.process = psutil.Process(self.pid)

    def _take_snapshot(self) -> Optional[ResourceSnapshot]:
        """Take a snapshot of process resources."""
        try:
            # CPU usage (process-specific)
            cpu_percent = self.process.cpu_percent(interval=None)

            # Memory usage (process-specific)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self.process.memory_percent()

            # I/O (process-specific)
            try:
                io_counters = self.process.io_counters()
                if io_counters and self.initial_disk_io:
                    disk_read_mb = (io_counters.read_bytes - self.initial_disk_io.read_bytes) / (1024 * 1024)
                    disk_write_mb = (io_counters.write_bytes - self.initial_disk_io.write_bytes) / (1024 * 1024)
                else:
                    disk_read_mb = 0
                    disk_write_mb = 0
            except (AttributeError, psutil.AccessDenied):
                disk_read_mb = 0
                disk_write_mb = 0

            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                network_sent_mb=0,  # Not available per-process
                network_recv_mb=0
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
            print(f"Error monitoring process {self.pid}: {e}")
            return None

    def start(self):
        """Start monitoring process resources."""
        # Get initial I/O counters for process
        try:
            self.initial_disk_io = self.process.io_counters()
        except (AttributeError, psutil.AccessDenied):
            self.initial_disk_io = None

        super().start()
