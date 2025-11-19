#!/usr/bin/env python3
"""
GPU Diagnostic Tool for RTX 5090 and PyTorch Setup
Helps identify CUDA compatibility issues

Usage:
    python diagnose_gpu.py
"""

import sys
import subprocess
from pathlib import Path


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def run_command(cmd: str) -> str:
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error running command: {e}"


def check_pytorch():
    """Check PyTorch installation"""
    print_section("PyTorch Installation")

    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")

        # Parse version
        version_str = torch.__version__.split('+')[0]
        version_parts = version_str.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])

        print(f"  - Major version: {major}")
        print(f"  - Minor version: {minor}")

        if major < 2 or (major == 2 and minor < 8):
            print(f"\n✗ WARNING: PyTorch {torch.__version__} is too old for RTX 5090!")
            print(f"  RTX 5090 requires PyTorch 2.8.0 or newer")
            print(f"\n  To fix:")
            print(f"    pip uninstall torch torchvision torchaudio -y")
            print(f"    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            return False
        else:
            print(f"✓ PyTorch version is compatible with RTX 5090")

        # Check CUDA build
        if '+cu' in torch.__version__:
            cuda_build = torch.__version__.split('+')[1]
            print(f"✓ CUDA build: {cuda_build}")
        else:
            print(f"✗ WARNING: PyTorch appears to be CPU-only build!")
            return False

        return True

    except ImportError:
        print("✗ PyTorch not installed!")
        print("\n  To install:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return False


def check_cuda():
    """Check CUDA availability"""
    print_section("CUDA Availability")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if not cuda_available:
            print("\n✗ CUDA is not available!")
            print("  Possible reasons:")
            print("    1. PyTorch CPU-only build")
            print("    2. CUDA drivers not installed")
            print("    3. GPU not accessible")
            return False

        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")

        return True

    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_gpu():
    """Check GPU details"""
    print_section("GPU Information")

    try:
        import torch

        if not torch.cuda.is_available():
            print("✗ No GPU available to check")
            return False

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")

            # Get compute capability
            capability = torch.cuda.get_device_capability(i)
            capability_str = f"sm_{capability[0]}{capability[1]}"
            print(f"  Compute capability: {capability[0]}.{capability[1]} ({capability_str})")

            # Check for RTX 5090
            if capability == (12, 0):
                print(f"  ✓ RTX 5090 detected (sm_120)")
            elif "5090" in torch.cuda.get_device_name(i):
                print(f"  ⚠️  RTX 5090 name detected but capability is {capability_str}")

            # Memory
            props = torch.cuda.get_device_properties(i)
            total_memory_gb = props.total_memory / (1024**3)
            print(f"  Total memory: {total_memory_gb:.2f} GB")
            print(f"  Multi-processor count: {props.multi_processor_count}")

        return True

    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def check_cuda_kernel():
    """Test CUDA kernel execution"""
    print_section("CUDA Kernel Test")

    try:
        import torch

        if not torch.cuda.is_available():
            print("✗ CUDA not available, skipping kernel test")
            return False

        print("Testing simple tensor operations on GPU...")

        # Test 1: Create tensor
        try:
            test_tensor = torch.randn(10, 10, device='cuda')
            print("✓ Tensor creation on GPU successful")
        except RuntimeError as e:
            if "no kernel image is available" in str(e):
                print(f"\n✗ KERNEL ERROR: {e}")
                print("\nThis means PyTorch doesn't have kernels compiled for your GPU!")
                print("For RTX 5090 (sm_120), you MUST use PyTorch 2.8.0+")
                return False
            else:
                raise

        # Test 2: Matrix multiplication
        try:
            result = test_tensor @ test_tensor.T
            print("✓ Matrix multiplication successful")
        except RuntimeError as e:
            if "no kernel image is available" in str(e):
                print(f"\n✗ KERNEL ERROR during matmul: {e}")
                return False
            else:
                raise

        # Test 3: Move to CPU and back
        try:
            cpu_tensor = test_tensor.cpu()
            gpu_tensor = cpu_tensor.cuda()
            print("✓ CPU ↔ GPU transfer successful")
        except RuntimeError as e:
            print(f"✗ Transfer error: {e}")
            return False

        # Test 4: Check supported capabilities
        print("\nSupported CUDA capabilities in this PyTorch build:")
        capability = torch.cuda.get_device_capability(0)
        current_cap = f"sm_{capability[0]}{capability[1]}"

        # Try to infer from error messages or version
        print(f"  Current GPU: {current_cap}")

        if capability[0] >= 12:
            print("  ✓ sm_120 appears to be supported")

        print("\n✓ All CUDA kernel tests passed!")
        return True

    except Exception as e:
        print(f"✗ Unexpected error during kernel test: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check other critical dependencies"""
    print_section("Critical Dependencies")

    checks = {
        "FlagEmbedding": "FlagEmbedding",
        "ChromaDB": "chromadb",
        "PEFT": "peft",
        "ONNX Runtime": "onnxruntime_gpu",
    }

    all_ok = True

    for name, module in checks.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
            all_ok = False

    return all_ok


def check_flagembedding_gpu():
    """Test FlagEmbedding with GPU"""
    print_section("FlagEmbedding GPU Test")

    try:
        import torch

        if not torch.cuda.is_available():
            print("✗ Skipping FlagEmbedding test (no CUDA)")
            return False

        from FlagEmbedding import BGEM3FlagModel

        print("Loading BGEM3FlagModel on GPU...")
        print("(This may take a while on first run)")

        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
        print("✓ Model loaded successfully")

        # Test encoding
        print("Testing encoding...")
        embeddings = model.encode(
            ["Test sentence for GPU encoding"],
            return_dense=True
        )

        shape = embeddings['dense_vecs'].shape
        print(f"✓ Encoding successful - shape: {shape}")

        # Clean up
        del model
        torch.cuda.empty_cache()

        print("✓ FlagEmbedding GPU test passed!")
        return True

    except Exception as e:
        print(f"✗ FlagEmbedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_environment():
    """Check environment variables and files"""
    print_section("Environment Check")

    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print(f"✓ .env file exists")
        # Check if it has OPENAI_API_KEY
        content = env_file.read_text()
        if "OPENAI_API_KEY" in content:
            print(f"✓ OPENAI_API_KEY found in .env")
        else:
            print(f"⚠️  OPENAI_API_KEY not found in .env")
    else:
        print(f"✗ .env file not found")
        print("  Create it with: echo 'OPENAI_API_KEY=your-key' > .env")

    # Check config
    config_file = Path("config/base.yaml")
    if config_file.exists():
        print(f"✓ config/base.yaml exists")
    else:
        print(f"✗ config/base.yaml not found")

    # Check data directory
    data_dir = Path("../../data/raw")
    if data_dir.exists():
        print(f"✓ data/raw directory exists")
    else:
        print(f"⚠️  data/raw directory not found")


def check_nvidia_smi():
    """Check nvidia-smi output"""
    print_section("nvidia-smi Output")

    output = run_command("nvidia-smi")
    if output:
        print(output)
    else:
        print("✗ Could not run nvidia-smi")
        print("  Make sure NVIDIA drivers are installed")


def main():
    """Run all diagnostics"""
    print("\n" + "="*60)
    print("  GPU DIAGNOSTICS FOR RTX 5090")
    print("  PyTorch CUDA Compatibility Check")
    print("="*60)

    results = {
        "PyTorch": check_pytorch(),
        "CUDA": check_cuda(),
        "GPU": check_gpu(),
        "CUDA Kernels": check_cuda_kernel(),
        "Dependencies": check_dependencies(),
        "FlagEmbedding": check_flagembedding_gpu(),
    }

    check_environment()
    check_nvidia_smi()

    # Summary
    print_section("DIAGNOSTIC SUMMARY")

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {check}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("="*60)
        print("  ✓ ALL CHECKS PASSED!")
        print("  Your environment is ready for RTX 5090")
        print("="*60)
        print("\nYou can now run:")
        print("  python smoke_test.py --skip-prep")
        return 0
    else:
        print("="*60)
        print("  ✗ SOME CHECKS FAILED")
        print("  Review the errors above")
        print("="*60)
        print("\nCommon fixes:")
        print("  1. For PyTorch version issues:")
        print("     pip uninstall torch torchvision torchaudio -y")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("  2. For 'no kernel image' errors:")
        print("     Make sure you have PyTorch 2.8.0+ installed")
        print()
        print("  3. For missing dependencies:")
        print("     pip install -r requirements_runpod.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
