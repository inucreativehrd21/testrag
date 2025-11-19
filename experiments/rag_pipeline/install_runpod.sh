#!/bin/bash
# Automated setup script for Runpod with RTX 5090
# Usage: bash install_runpod.sh

set -e  # Exit on error

echo "=================================="
echo "RAG Pipeline Runpod Setup Script"
echo "=================================="
echo ""

# Step 1: Remove old PyTorch
echo "[1/5] Removing old PyTorch installation..."
pip uninstall torch torchvision torchaudio -y || true
pip cache purge

# Step 2: Install PyTorch 2.8.0+
echo ""
echo "[2/5] Installing PyTorch 2.8.0+ with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 3: Verify PyTorch
echo ""
echo "[3/5] Verifying PyTorch installation..."
python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA version: {torch.version.cuda}')
else:
    print('✗ ERROR: CUDA not available!')
    exit(1)
"

# Step 4: Install dependencies
echo ""
echo "[4/5] Installing RAG pipeline dependencies..."
pip install -r requirements_runpod.txt

# Step 5: Check environment
echo ""
echo "[5/5] Checking environment setup..."

if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "   Create it with: echo 'OPENAI_API_KEY=your-key-here' > .env"
else
    echo "✓ .env file exists"
fi

if [ ! -d "artifacts" ]; then
    echo "ℹ️  INFO: artifacts/ directory will be created on first run"
fi

echo ""
echo "=================================="
echo "✓ Setup complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Set your OpenAI API key:"
echo "     echo 'OPENAI_API_KEY=your-key-here' > .env"
echo ""
echo "  2. Run smoke test:"
echo "     python smoke_test.py --skip-prep"
echo ""
echo "  3. Start API server:"
echo "     python serve.py"
echo ""
