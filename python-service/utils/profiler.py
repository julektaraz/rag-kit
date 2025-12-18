"""
Performance Profiling Module

Tracks timing and memory usage for operations.
"""

import time
import logging
from typing import Dict, Optional
from contextlib import contextmanager
import torch

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Track performance metrics."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics: Dict[str, list] = {}
        self.current_operation: Optional[str] = None
        self.start_time: Optional[float] = None
    
    @contextmanager
    def profile(self, operation: str):
        """Context manager for profiling operations."""
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_mb()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_mb()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append({
                "duration": duration,
                "memory_delta_mb": memory_delta,
            })
            
            logger.debug(
                f"Profiled {operation}: {duration:.3f}s, "
                f"memory delta: {memory_delta:+.1f}MB"
            )
    
    def _get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get aggregated statistics."""
        stats = {}
        for operation, measurements in self.metrics.items():
            if not measurements:
                continue
                
            durations = [m["duration"] for m in measurements]
            memory_deltas = [m["memory_delta_mb"] for m in measurements]
            
            stats[operation] = {
                "count": len(measurements),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "total_duration": sum(durations),
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "avg_memory_delta": sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
            }
        
        return stats
    
    def log_summary(self) -> None:
        """Log performance summary."""
        stats = self.get_stats()
        if not stats:
            return
        
        logger.info("=== Performance Summary ===")
        for operation, stat in stats.items():
            logger.info(
                f"{operation}: {stat['count']} calls, "
                f"avg {stat['avg_duration']:.3f}s, "
                f"total {stat['total_duration']:.2f}s"
            )
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        logger.info("Performance profiler reset")


# Global profiler instance
_profiler: Optional[PerformanceProfiler] = None


def get_profiler(enabled: bool = True) -> PerformanceProfiler:
    """Get or create global profiler."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler(enabled=enabled)
    return _profiler


__all__ = ["PerformanceProfiler", "get_profiler"]

