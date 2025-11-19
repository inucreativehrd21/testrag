#!/bin/bash
# Automated setup script for Runpod with RTX 5090
# Usage: bash install_runpod.sh
# This script AGGRESSIVELY removes old PyTorch and installs 2.8.0+ for sm_120 support

set -e  # Exit on error

echo "=================================="
echo "RAG Pipeline Runpod Setup Script"
echo "RTX 5090 (sm_120) Compatible"
echo "=================================="
echo ""

# Check Python version
echo "[0/7] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if in virtual environment (recommended)
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  WARNING: Not in a virtual environment!"
    echo "   It's recommended to use a venv to avoid conflicts"
    echo "   Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5
else
    echo "✓ Virtual environment: $VIRTUAL_ENV"
fi
echo ""

# Step 1: AGGRESSIVE PyTorch removal
echo "[1/7] AGGRESSIVELY removing ALL PyTorch installations..."
echo "This will remove torch, torchvision, torchaudio, and all related packages"

# Find and uninstall ALL torch-related packages
pip list | grep torch | awk '{print $1}' | xargs -r pip uninstall -y 2>/dev/null || true

# Explicitly uninstall main packages
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true  # Run twice to be sure

# Purge pip cache
pip cache purge

# Remove any torch-related directories from site-packages
echo "Removing residual torch directories..."
python -c "
import site
import shutil
import os
from pathlib import Path

for site_dir in site.getsitepackages():
    site_path = Path(site_dir)
    if site_path.exists():
        # Remove torch directories
        for pattern in ['torch*', '*torch*']:
            for path in site_path.glob(pattern):
                if 'torch' in path.name.lower():
                    try:
                        if path.is_dir():
                            shutil.rmtree(path, ignore_errors=True)
                            print(f'Removed: {path}')
                        elif path.is_file():
                            path.unlink()
                            print(f'Removed: {path}')
                    except Exception as e:
                        print(f'Could not remove {path}: {e}')
" || echo "Could not clean torch directories (not critical)"

echo "✓ PyTorch removal complete"
echo ""

# Step 2: Verify removal
echo "[2/7] Verifying PyTorch is completely removed..."
if python -c "import torch" 2>/dev/null; then
    echo "✗ ERROR: PyTorch is still importable!"
    echo "  This means old PyTorch wasn't fully removed."
    echo "  Try manually removing it:"
    echo "    pip list | grep torch"
    echo "    pip uninstall <package-name> -y"
    exit 1
else
    echo "✓ PyTorch successfully removed (import fails as expected)"
fi
echo ""

# Step 3: Install PyTorch 2.8.0+ with CUDA 12.4
echo "[3/7] Installing PyTorch 2.8.0+ with CUDA 12.4 support..."
echo "This version supports RTX 5090 (sm_120)"
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo ""

# Step 4: Verify PyTorch installation and RTX 5090 support
echo "[4/7] Verifying PyTorch installation..."
python -c "
import torch
import sys

print(f'✓ PyTorch version: {torch.__version__}')

# Don't check specific version number, just verify it works with sm_120
# (nightly builds may have different version schemes)

print(f'✓ CUDA available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('✗ ERROR: CUDA not available!')
    sys.exit(1)

print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ CUDA version: {torch.version.cuda}')

# Check CUDA capability
capability = torch.cuda.get_device_capability(0)
capability_str = f'sm_{capability[0]}{capability[1]}'
print(f'✓ CUDA capability: {capability_str}')

# Verify sm_120 support for RTX 5090
if 'RTX 5090' in torch.cuda.get_device_name(0) or capability == (12, 0):
    print('✓ RTX 5090 detected - checking sm_120 support...')

    # Try to create a simple tensor on GPU
    try:
        test_tensor = torch.randn(10, 10, device='cuda')
        result = test_tensor @ test_tensor.T
        print('✓ sm_120 kernel test PASSED - RTX 5090 fully supported!')
    except RuntimeError as e:
        if 'no kernel image is available' in str(e):
            print(f'✗ ERROR: sm_120 NOT supported! {e}')
            print('  Your PyTorch installation does not support RTX 5090')
            sys.exit(1)
        else:
            raise

print('')
print('========================================')
print('✓ PyTorch verification SUCCESSFUL')
print('========================================')
"

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ PyTorch verification FAILED!"
    echo "  Please check the error messages above"
    exit 1
fi
echo ""

# Step 5: Install dependencies
echo "[5/7] Installing RAG pipeline dependencies..."
pip install --no-cache-dir -r requirements_runpod.txt
echo ""

# Step 6: Verify FlagEmbedding with GPU
echo "[6/7] Verifying FlagEmbedding can use GPU..."
python -c "
import torch
from FlagEmbedding import BGEM3FlagModel

print('Testing BGEM3FlagModel on GPU...')
try:
    # This will download the model if not cached
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
    print('✓ Model loaded successfully on GPU')

    # Test encoding
    embeddings = model.encode(['Test sentence'], return_dense=True)
    print(f'✓ Encoding test passed - shape: {embeddings[\"dense_vecs\"].shape}')

    # Clean up
    del model
    torch.cuda.empty_cache()
    print('✓ FlagEmbedding verification PASSED')
except Exception as e:
    print(f'✗ ERROR: FlagEmbedding test failed: {e}')
    print('  This might cause issues during indexing/retrieval')
    exit(1)
" || {
    echo "✗ FlagEmbedding verification failed!"
    echo "  Check if models can be downloaded from HuggingFace"
    exit 1
}
echo ""

# Step 7: Check environment
echo "[7/7] Checking environment setup..."

if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "   Create it with: echo 'OPENAI_API_KEY=your-key-here' > .env"
else
    echo "✓ .env file exists"
fi

if [ ! -d "artifacts" ]; then
    echo "ℹ️  INFO: artifacts/ directory will be created on first run"
else
    echo "✓ artifacts/ directory exists"
fi

if [ ! -d "../../data/raw" ]; then
    echo "⚠️  WARNING: ../../data/raw not found!"
    echo "   Make sure your data files are in the correct location"
else
    echo "✓ data/raw directory exists"
fi

echo ""
echo "========================================"
echo "✓ SETUP COMPLETE!"
echo "========================================"
echo ""
echo "Your Runpod environment is ready for RTX 5090"
echo ""
echo "Next steps:"
echo "  1. Ensure .env file has your OpenAI API key:"
echo "     echo 'OPENAI_API_KEY=your-key-here' > .env"
echo ""
echo "  2. Run diagnostic check (optional):"
echo "     python diagnose_gpu.py"
echo ""
echo "  3. Run smoke test (skip prep if data exists):"
echo "     python smoke_test.py --skip-prep"
echo ""
echo "  4. Or start API server directly:"
echo "     python serve.py"
echo ""
echo "If you encounter 'no kernel image' errors, run:"
echo "  python diagnose_gpu.py"
echo ""
