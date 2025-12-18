"""
GPU Memory Monitoring and Management

Provides real-time GPU memory tracking, capacity monitoring, and memory optimization.
"""

import torch
import logging
from typing import Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor and manage GPU memory usage."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.startswith("cuda")
        
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dict with:
                - allocated_mb: Currently allocated memory (MB)
                - reserved_mb: Reserved memory (MB)
                - total_mb: Total GPU memory (MB)
                - free_mb: Free memory (MB)
                - utilization_pct: Utilization percentage
        """
        if not self.is_cuda:
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_mb": 0.0,
                "free_mb": 0.0,
                "utilization_pct": 0.0,
            }
        
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        free = total - reserved
        utilization = (reserved / total * 100) if total > 0 else 0.0
        
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "total_mb": total,
            "free_mb": free,
            "utilization_pct": utilization,
        }
    
    def get_optimal_batch_size(
        self,
        base_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 256,
    ) -> int:
        """
        Dynamically determine optimal batch size based on available GPU memory.
        
        Args:
            base_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            
        Returns:
            Optimal batch size
        """
        if not self.is_cuda:
            return base_batch_size
        
        info = self.get_memory_info()
        utilization = info["utilization_pct"]
        
        # Adjust batch size based on memory pressure
        if utilization < 40:
            # Plenty of memory - increase batch size
            optimal = min(base_batch_size * 2, max_batch_size)
        elif utilization < 70:
            # Moderate usage - use base
            optimal = base_batch_size
        elif utilization < 85:
            # High usage - reduce batch size
            optimal = max(base_batch_size // 2, min_batch_size)
        else:
            # Very high usage - minimal batch size
            optimal = min_batch_size
        
        logger.debug(f"GPU utilization: {utilization:.1f}%, optimal batch size: {optimal}")
        return optimal
    
    def clear_cache(self) -> None:
        """Clear CUDA cache to free fragmented memory."""
        if self.is_cuda:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    
    @contextmanager
    def memory_context(self, clear_after: bool = True):
        """Context manager for memory operations."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        try:
            yield
        finally:
            if clear_after and self.is_cuda:
                torch.cuda.empty_cache()
    
    def log_memory_stats(self, operation: str = "") -> None:
        """Log current memory statistics."""
        if not self.is_cuda:
            return
        
        info = self.get_memory_info()
        logger.info(
            f"GPU Memory [{operation}]: "
            f"{info['allocated_mb']:.1f}MB allocated, "
            f"{info['reserved_mb']:.1f}MB reserved, "
            f"{info['utilization_pct']:.1f}% utilized"
        )


# Global instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor(device: Optional[str] = None) -> GPUMonitor:
    """Get or create global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(device)
    return _gpu_monitor


__all__ = ["GPUMonitor", "get_gpu_monitor"]

