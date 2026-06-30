"""Health check & device diagnostic script for PyTorch (CPU/AMD ROCm) with Benchmark."""
import os
import sys
import time
import torch


def run_benchmark(device: str, runs: int = 20) -> float:
    """Runs a heavy matrix multiplication loop and returns the total time in seconds."""
    # Matrix size chosen to balance CPU load and GPU parallelization
    size = 2048
    
    # Warm-up run to initialize the device and caches properly
    x = torch.randn(size, size, device=device, dtype=torch.float32)
    y = torch.randn(size, size, device=device, dtype=torch.float32)
    _ = x @ y
    if device == "cuda":
        torch.cuda.synchronize()

    # Real timed loop
    start_time = time.perf_counter()
    for _ in range(runs):
        _ = x @ y
        
    if device == "cuda":
        torch.cuda.synchronize()  # Force GPU to finish before stopping clock
        
    end_time = time.perf_counter()
    return end_time - start_time


def main() -> None:
    print("-" * 60)
    print("🚀 STARTING PYTORCH DEVICE DIAGNOSTIC & BENCHMARK")
    print("-" * 60)

    # 1. Environment & Driver Check
    gfx_env = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    hip_env = os.environ.get("HIP_VISIBLE_DEVICES")
    
    print(f"📌 Detected HSA_OVERRIDE_GFX_VERSION: {gfx_env if gfx_env else '❌ NOT SET'}")
    print(f"📌 Detected HIP_VISIBLE_DEVICES:     {hip_env if hip_env else '❌ NOT SET'}")
    print(f"📌 PyTorch version:                   {torch.__version__}")

    # 2. Smart GPU Safety Validation (Prevents SegFault / Error 139)
    rocm_available = torch.cuda.is_available()
    gpu_is_safe = False
    device_name = "Unknown"

    if rocm_available:
        try:
            device_name = torch.cuda.get_device_name(0).upper()
            print(f"📌 Visible GPU count:                 {torch.cuda.device_count()}")
            print(f"📌 Device 0 Name:                     {device_name}")
            
            # CRITICAL CHECK: Mismatch detection
            is_midrange_amd = any(card in device_name for card in ["RX 6600", "RX 6700", "RX 7600", "RX 7700"])
            
            if is_midrange_amd and not gfx_env:
                print("\n⚠️  WARNING: Mismatch detected!")
                print(f"👉 Your '{device_name}' requires 'HSA_OVERRIDE_GFX_VERSION' to compute safely.")
                print("❌ Running GPU benchmarks now would cause a system Segmentation Fault (Error 139).")
                print("🔒 Forcing fallback to CPU-only mode to maintain stability.")
                gpu_is_safe = False
            else:
                gpu_is_safe = True
                
        except Exception as e:
            print(f"⚠️ Could not read GPU metadata: {e}")
            gpu_is_safe = False
    else:
        print("ℹ️ No AMD GPU detected via ROCm. Defaulting to CPU mode.")

    # 3. BENCHMARK SECTION
    print("-" * 60)
    print("📊 RUNNING HEAVY COMPUTATION BENCHMARK (2048x2048 MatMul Loop)")
    print("-" * 60)

    # Always test CPU as baseline
    print("⏳ Benchmarking CPU... (this might take a few seconds)")
    cpu_time = run_benchmark("cpu")
    print(f"⏱️  CPU Total Time: {cpu_time:.4f} seconds")

    if rocm_available and gpu_is_safe:
        print("\n⏳ Benchmarking AMD GPU via ROCm...")
        try:
            gpu_time = run_benchmark("cuda")
            print(f"⏱️  GPU Total Time: {gpu_time:.4f} seconds")
            
            # Calculate speedup factor
            speedup = cpu_time / gpu_time
            print(f"\n🚀 RESULT: Your GPU is {speedup:.1f}x FASTER than your CPU!")
            print("-" * 60)
            print("🎉 System fully optimized! Ready for high-speed AMD GPU training. 🎉")
            print("-" * 60)
        except Exception as e:
            print(f"❌ Unexpected error during GPU benchmark: {e}")
    else:
        print("\nℹ️  GPU Benchmark skipped (Unsafe or Not Available).")
        print("💡 Did you know? A properly configured AMD GPU is usually 20x to 50x faster")
        print("   than a CPU for these types of neural network workloads.")
        print("-" * 60)
        print("🚀 Ready for CPU training.")
        print("📢 Note: Run 'make setup' if you want to unlock GPU acceleration.")
        print("-" * 60)


if __name__ == "__main__":
    main()
