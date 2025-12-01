# Runpod Setup Guide for RAG Pipeline

This guide helps you set up the RAG pipeline on Runpod with RTX 5090 or other high-end GPUs.

## Prerequisites

- Runpod pod with RTX 5090 or similar GPU (CUDA 12.0+)
- Python 3.11 or 3.12
- Git installed

## Quick Setup

### Option A: Automated Setup (Recommended)

**Use the automated setup script for complete installation:**

```bash
git clone https://github.com/inucreativehrd21/testrag.git
cd testrag/experiments/rag_pipeline

# Run automated setup script
bash install_runpod.sh
```

The script will:
1. ✅ Check Python version and virtual environment
2. ✅ **AGGRESSIVELY remove all old PyTorch** installations
3. ✅ Verify PyTorch is completely removed
4. ✅ Install PyTorch 2.8.0+ with CUDA 12.4
5. ✅ Test RTX 5090 sm_120 kernel support
6. ✅ Install all dependencies
7. ✅ Verify FlagEmbedding works on GPU

---

### Option B: Manual Setup

If you prefer manual installation or the script fails:

#### 1. Clone Repository

```bash
git clone https://github.com/inucreativehrd21/testrag.git
cd testrag/experiments/rag_pipeline
```

#### 2. **CRITICAL**: Remove Old PyTorch & Install 2.8.0+

**⚠️ RTX 5090 requires PyTorch 2.8.0+ for CUDA capability sm_120 support**

```bash
# Step 1: AGGRESSIVELY remove old PyTorch
pip list | grep torch | awk '{print $1}' | xargs pip uninstall -y
pip uninstall torch torchvision torchaudio -y
pip cache purge

# Step 2: Install PyTorch 2.8.0+ with CUDA 12.4
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 3: Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

**Expected output:**
```
PyTorch: 2.8.0+cu124
CUDA: True
GPU: NVIDIA GeForce RTX 5090
```

**Why?** RTX 5090 has CUDA compute capability 12.0 (sm_120), which requires PyTorch 2.8.0 or later.

#### 3. Install Dependencies

```bash
pip install --no-cache-dir -r requirements_runpod.txt
```

---

### Verify Installation

After installation (automated or manual), run the diagnostic tool:

```bash
python diagnose_gpu.py
```

This will check:
- ✅ PyTorch version (must be 2.8.0+)
- ✅ CUDA availability
- ✅ GPU detection and compute capability
- ✅ sm_120 kernel support
- ✅ FlagEmbedding GPU compatibility
- ✅ All dependencies

If all checks pass, you're ready to run the pipeline!

---

### Configure for GPU

Edit [config/base.yaml](config/base.yaml):

```yaml
embedding:
  device: cuda  # Use GPU

retrieval:
  rerankers:
    stage1:
      device: cuda  # Single GPU
    stage2:
      device: cuda  # Can use same GPU or cuda:1 if available
```

### 5. Set Environment Variables

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 6. Run Smoke Test

```bash
# Full test (will create chunks and index)
python smoke_test.py

# Or skip data prep if chunks already exist
python smoke_test.py --skip-prep
```

---

## Expected Performance

With RTX 5090:
- **Data Preparation**: ~30 seconds (15,461 chunks)
- **Index Building**: ~2-5 minutes (BAAI/bge-m3)
- **Query Inference**: ~1-2 seconds per question

---

## Troubleshooting

### Error: "no kernel image is available for execution"

**Cause**: PyTorch version doesn't support your GPU's CUDA capability (sm_120 for RTX 5090).

**Solution**:

**Step 1**: Run diagnostics to confirm the issue:
```bash
python diagnose_gpu.py
```

**Step 2**: If PyTorch version < 2.8.0, completely remove and reinstall:
```bash
# AGGRESSIVE removal
pip list | grep torch | awk '{print $1}' | xargs pip uninstall -y
pip cache purge

# Verify removal (should fail with ImportError)
python -c "import torch"

# Reinstall PyTorch 2.8.0+
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Test sm_120 support
python -c "import torch; t = torch.randn(10,10,device='cuda'); print('✓ sm_120 works!')"
```

**Step 3**: If still failing, consider recreating your virtual environment:
```bash
# Exit current environment
deactivate

# Create fresh environment
python -m venv venv_rtx5090
source venv_rtx5090/bin/activate  # On Linux/Mac
# or: venv_rtx5090\Scripts\activate  # On Windows

# Run automated setup
bash install_runpod.sh
```

### Error: "CUDA out of memory"

**Solution 1**: Reduce batch size in `config/base.yaml`:
```yaml
embedding:
  batch_size: 16  # Reduce from 32
```

**Solution 2**: Use single reranker:
```yaml
retrieval:
  rerankers:
    stage1:
      device: cuda
    stage2:
      device: cpu  # Move to CPU
```

### Error: "No module named 'peft'"

**Solution**:
```bash
pip install peft==0.10.0
```

---

## Verification

### Quick Check

Run the comprehensive diagnostic tool:

```bash
python diagnose_gpu.py
```

This checks everything in one command and provides detailed troubleshooting if anything fails.

### Manual Verification

If you prefer to check manually:

#### Check CUDA Availability

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check compute capability
capability = torch.cuda.get_device_capability(0)
print(f"Compute capability: sm_{capability[0]}{capability[1]}")

# Test kernel execution
test_tensor = torch.randn(10, 10, device='cuda')
result = test_tensor @ test_tensor.T
print("✓ CUDA kernels working!")
```

Expected output:
```
PyTorch version: 2.8.0+cu124
CUDA available: True
CUDA version: 12.4
GPU: NVIDIA GeForce RTX 5090
Compute capability: sm_120
✓ CUDA kernels working!
```

#### Verify Embeddings Work

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
embeddings = model.encode(["Test sentence"], return_dense=True)
print(f"Embedding shape: {embeddings['dense_vecs'].shape}")
```

---

## Running the Pipeline

### 1. Data Preparation (if needed)

```bash
python data_prep.py
```

### 2. Build Index

```bash
python index_builder.py
```

### 3. Ask Questions

```bash
python answerer.py "Git의 브랜치란 무엇인가요?"
```

### 4. Start API Server

```bash
python serve.py --host 0.0.0.0 --port 8000
```

Test with curl:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Docker 컨테이너란?", "return_metadata": true}'
```

### 5. Run Batch Evaluation

```bash
python evaluate.py --questions sample_questions.txt
```

---

## Performance Tips

### 1. Use FP16 (Already Enabled)

Models automatically use FP16 for faster inference:
```python
BGEM3FlagModel(..., use_fp16=True)
```

### 2. Increase Batch Size (if GPU memory allows)

```yaml
embedding:
  batch_size: 64  # Increase from 32
```

### 3. Pin Models to Specific GPUs

If you have multiple GPUs:
```yaml
retrieval:
  rerankers:
    stage1:
      device: cuda:0
    stage2:
      device: cuda:1
```

---

## Common Commands

```bash
# Full pipeline test
python smoke_test.py

# Skip prep (if data already exists)
python smoke_test.py --skip-prep

# Skip both prep and indexing
python smoke_test.py --skip-prep --skip-index

# Run with debug logging
python answerer.py "Question" --log-level DEBUG

# Start server with auto-reload
python serve.py --reload
```

---

## File Locations

- **Config**: `config/base.yaml`
- **Chunks**: `artifacts/chunks.parquet`
- **Vector Index**: `artifacts/chroma_db/`
- **Eval Results**: `artifacts/evals/`
- **Logs**: Console output (or configure file logging)

---

## Next Steps

1. ✅ Verify GPU setup works
2. ✅ Run smoke test
3. ✅ Start API server
4. 📊 Run evaluations with your own questions
5. 🚀 Deploy to production

For more details, see [README.md](README.md).
