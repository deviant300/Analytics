"""To check memory"""

import cupy as cp


def memory_pool_func():
    """Flushes memory cache and checks memory pool information"""
    # Flush the CuPy memory cache
    cp.get_default_memory_pool().free_all_blocks()

    # Check memory pool information
    memory_pool_info = cp.get_default_memory_pool()
    print("Allocated memory:", memory_pool_info.total_bytes() / (1024**3), "GB")


# End-of-file (EOF)
