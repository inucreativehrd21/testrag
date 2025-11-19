#!/bin/bash
# Quick GPU check script

echo "==================================="
echo "GPU Information Check"
echo "==================================="
echo ""

echo "1. nvidia-smi GPU info:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

echo ""
echo "2. PyTorch GPU detection:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    cap = torch.cuda.get_device_capability(0)
    print(f'CUDA Capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})')

    if cap == (8, 9):
        print('')
        print('✓ RTX 4090 detected - PyTorch 2.6.0+ works fine')
        print('  No need for nightly build')
    elif cap == (12, 0):
        print('')
        print('⚠️  RTX 5090 detected - Needs PyTorch 2.8.0+ or nightly')
        print('  Current stable may not work')
    else:
        print(f'')
        print(f'ℹ️  GPU with sm_{cap[0]}{cap[1]} detected')
else:
    print('✗ CUDA not available')
" 2>/dev/null || echo "PyTorch not installed yet"

echo ""
echo "==================================="
