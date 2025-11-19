# RTX 5090 Troubleshooting Guide

This guide helps you resolve the most common RTX 5090 setup issues.

## Problem: "no kernel image is available for execution on the device"

### Symptoms
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.

UserWarning: NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_37 sm_90.
```

### Root Cause
Your PyTorch version doesn't have CUDA kernels compiled for sm_120 (RTX 5090's compute capability).

RTX 5090 requires **PyTorch 2.8.0 or newer** with CUDA 12.4 support.

---

## Solution: Complete PyTorch Reinstall

### Step 1: Run Diagnostics

First, identify the exact problem:

```bash
cd /path/to/testrag/experiments/rag_pipeline
python diagnose_gpu.py
```

Look for these key indicators:
- ✗ PyTorch version < 2.8.0
- ✗ CUDA Kernel Test fails
- ⚠️ Warning about sm_120 compatibility

### Step 2: Use Automated Fix

The easiest solution is to run the automated setup script:

```bash
bash install_runpod.sh
```

This script will:
1. **Aggressively remove** all PyTorch installations (including hidden files)
2. **Verify** complete removal
3. **Install** PyTorch 2.8.0+ from CUDA 12.4 index
4. **Test** sm_120 kernel support
5. **Install** all dependencies
6. **Verify** FlagEmbedding GPU compatibility

### Step 3: Manual Fix (if script fails)

If the automated script doesn't work, follow these manual steps:

#### 3.1 Remove ALL PyTorch Installations

```bash
# Find all torch packages
pip list | grep torch

# Uninstall ALL of them
pip list | grep torch | awk '{print $1}' | xargs pip uninstall -y

# Explicitly remove main packages (run twice!)
pip uninstall torch torchvision torchaudio -y
pip uninstall torch torchvision torchaudio -y

# Purge pip cache
pip cache purge
```

#### 3.2 Clean Site-Packages

Remove residual torch directories:

```python
python -c "
import site
import shutil
from pathlib import Path

for site_dir in site.getsitepackages():
    site_path = Path(site_dir)
    if site_path.exists():
        for path in site_path.glob('*torch*'):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                print(f'Removed: {path}')
"
```

#### 3.3 Verify Removal

```bash
# This should FAIL with ImportError
python -c "import torch"
```

If it succeeds, PyTorch is still installed somewhere. Check:
- System-wide installation
- Conda environments
- Other virtual environments

#### 3.4 Install PyTorch 2.8.0+

```bash
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 3.5 Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

# Critical: Test sm_120 kernel
test = torch.randn(10, 10, device='cuda')
result = test @ test.T
print('✓ sm_120 kernels working!')
"
```

Expected output:
```
PyTorch: 2.8.0+cu124
CUDA: True
GPU: NVIDIA GeForce RTX 5090
✓ sm_120 kernels working!
```

---

## Solution: Fresh Virtual Environment

If the above doesn't work, old PyTorch might be cached in your environment.

### Create New Environment

```bash
# Exit current environment
deactivate  # or conda deactivate

# Remove old environment
rm -rf venv/  # or conda env remove -n myenv

# Create fresh environment
python3.11 -m venv venv_rtx5090
source venv_rtx5090/bin/activate

# Verify clean state
python -c "import torch"  # Should fail

# Install PyTorch 2.8.0+
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install --no-cache-dir -r requirements_runpod.txt

# Test
python diagnose_gpu.py
```

---

## Verification Checklist

After installation, verify everything works:

### ✅ PyTorch Version
```bash
python -c "import torch; print(torch.__version__)"
```
**Must be**: `2.8.0+cu124` or newer

### ✅ CUDA Available
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
**Must be**: `True`

### ✅ GPU Detection
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
**Should be**: `NVIDIA GeForce RTX 5090`

### ✅ Compute Capability
```bash
python -c "import torch; cap = torch.cuda.get_device_capability(0); print(f'sm_{cap[0]}{cap[1]}')"
```
**Must be**: `sm_120`

### ✅ Kernel Test
```bash
python -c "import torch; t = torch.randn(100, 100, device='cuda'); r = t @ t.T; print('✓ Kernels work!')"
```
**Must NOT** error with "no kernel image available"

### ✅ FlagEmbedding GPU
```bash
python -c "
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
emb = model.encode(['test'], return_dense=True)
print('✓ FlagEmbedding works!')
"
```
**Must NOT** error

### ✅ Full Diagnostic
```bash
python diagnose_gpu.py
```
**All checks** should pass

---

## Common Mistakes

### ❌ Not Removing Old PyTorch Completely
Running `pip install torch --upgrade` doesn't fully remove old versions. You **must** uninstall first.

### ❌ Installing Wrong PyTorch Index
```bash
# WRONG - defaults to CPU or old CUDA
pip install torch

# WRONG - CUDA 11.8 (too old)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CORRECT - CUDA 12.4 (supports sm_120)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### ❌ System-Wide vs Virtual Environment
If you have PyTorch installed system-wide, your virtual environment might still import it. Always verify:
```bash
python -c "import torch; print(torch.__file__)"
```
Should point to your venv, not `/usr/lib` or system Python.

### ❌ Cached Files
Pip cache can prevent clean installation:
```bash
pip cache purge  # Always run before reinstalling
```

### ❌ Multiple CUDA Versions
If you have multiple CUDA toolkits installed, PyTorch might link to the wrong one. Set:
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## Still Not Working?

### Check nvidia-smi
```bash
nvidia-smi
```
Verify:
- Driver version (should be 550+ for RTX 5090)
- CUDA version (should be 12.4+)
- GPU is listed

### Check CUDA Toolkit
```bash
nvcc --version
```
Should show CUDA 12.4 or newer.

### Check PyTorch Build Info
```python
import torch
print(torch.__config__.show())
```
Look for `CUDA_VERSION=12.4` and `sm_120` in CUDA architectures.

### Contact Support

If all else fails:
1. Run `python diagnose_gpu.py` and save output
2. Run `nvidia-smi` and save output
3. Run `pip list | grep torch` and save output
4. Open an issue with all diagnostics

---

## Quick Commands Reference

```bash
# Diagnose issue
python diagnose_gpu.py

# Automated fix
bash install_runpod.sh

# Manual PyTorch reinstall
pip uninstall torch torchvision torchaudio -y && \
pip cache purge && \
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify installation
python -c "import torch; print(torch.__version__); t=torch.randn(10,10,device='cuda'); print('✓ Works!')"

# Fresh environment
deactivate && rm -rf venv && python3.11 -m venv venv && source venv/bin/activate && \
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## Success Criteria

You're ready to run the RAG pipeline when:
- ✅ `python diagnose_gpu.py` shows all checks passing
- ✅ PyTorch version is 2.8.0+cu124 or newer
- ✅ CUDA kernel test creates tensors without errors
- ✅ FlagEmbedding loads on GPU without warnings

Then run:
```bash
python smoke_test.py --skip-prep
```
