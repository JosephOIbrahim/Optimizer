"""Post-optimization benchmark script"""
import torch
import time
import os

print("=" * 70)
print("POST-OPTIMIZATION BENCHMARK")
print("=" * 70)
print()

# Current env state
print("Active Environment Variables:")
print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
print(f"  CUDA_DEVICE_MAX_L2_PERSISTING_CACHE_SIZE: {os.environ.get('CUDA_DEVICE_MAX_L2_PERSISTING_CACHE_SIZE', 'NOT SET')}")
print(f"  CUDNN_BENCHMARK: {os.environ.get('CUDNN_BENCHMARK', 'NOT SET')}")
print(f"  HOUDINI_OCL_DEVICETYPE: {os.environ.get('HOUDINI_OCL_DEVICETYPE', 'NOT SET')}")
print()

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Benchmark 1: Allocation speed
    print("Benchmark 1: CUDA Allocation Speed")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        tensors = [torch.empty(2048, 2048, device='cuda') for _ in range(20)]
        del tensors
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    alloc_time = (time.perf_counter() - start) * 1000
    print(f"  50 rounds of 20x allocations: {alloc_time:.2f} ms")

    # Benchmark 2: Memory bandwidth (L2 cache relevant)
    print()
    print("Benchmark 2: Memory-Bound Operation (matmul)")
    a = torch.randn(4096, 4096, device='cuda')
    b = torch.randn(4096, 4096, device='cuda')

    # Warmup
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(20):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    matmul_time = (time.perf_counter() - start) * 1000
    print(f"  20x [4096x4096] matmul: {matmul_time:.2f} ms")
    print(f"  Per-op: {matmul_time/20:.2f} ms")

    # Calculate TFLOPS
    flops = 2 * 4096 * 4096 * 4096 * 20
    tflops = flops / (matmul_time / 1000) / 1e12
    print(f"  Throughput: {tflops:.1f} TFLOPS")

    del a, b, c
    torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print(f"POST-OPT: Alloc={alloc_time:.2f}ms, Matmul={matmul_time:.2f}ms")
    print("=" * 70)
