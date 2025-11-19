# RAG Pipeline for Development Q&A

A production-ready Retrieval-Augmented Generation (RAG) pipeline for answering development-related questions in Korean. The pipeline uses multi-stage retrieval with dense embeddings and cross-encoder reranking, powered by OpenAI's GPT models.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Index Building](#index-building)
  - [Question Answering](#question-answering)
  - [API Server](#api-server)
  - [Batch Evaluation](#batch-evaluation)
  - [Ragas Benchmark](#ragas-benchmark)
- [Configuration](#configuration)
- [Logging](#logging)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [TODO](#todo)

## Features

- **Multi-stage Retrieval**: Dense retrieval → Stage 1 reranking → Stage 2 reranking
- **Query Routing**: Automatic difficulty classification and strategy selection
- **Structured Logging**: Comprehensive logging with timing metrics
- **REST API**: FastAPI-based serving with OpenAPI documentation
- **Batch Evaluation**: Tools for systematic testing and metrics collection
- **Production-Ready**: Error handling, health checks, and monitoring hooks
- **Korean Language**: Optimized prompts and responses in Korean

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline Workflow                     │
└─────────────────────────────────────────────────────────────┘

1. Data Preparation (data_prep.py)
   Raw Documents → Text Chunking → chunks.parquet

2. Index Building (index_builder.py)
   chunks.parquet → Dense Embeddings → ChromaDB

3. Question Answering (answerer.py)
   User Question
      ↓
   Query Routing (difficulty + strategy)
      ↓
   Dense Retrieval (top 25)
      ↓
   Stage 1 Reranking (BAAI/bge-reranker-v2-m3)
      ↓
   Stage 2 Reranking (BAAI/bge-reranker-large → top 5)
      ↓
   LLM Generation (GPT-4.1)
      ↓
   Korean Answer with Citations
```

### Components

- **Embedding Model**: BAAI/bge-m3 (multilingual, 1024 dimensions)
- **Rerankers**: Two-stage CrossEncoder reranking for precision
- **Vector Store**: ChromaDB for efficient similarity search
- **LLM**: OpenAI GPT-4.1 with temperature=0.2
- **Chunking**: Recursive character splitting (1024/256 tokens)

## Setup

### Prerequisites

- Python 3.10 or 3.11
- CUDA-capable GPU (recommended for embeddings and reranking)
- OpenAI API key

### Installation

1. **Clone or navigate to the pipeline directory**:
   ```bash
   cd experiments/rag_pipeline
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   # Required for LLM calls
   export OPENAI_API_KEY=your-api-key-here
   ```

4. **Prepare your data**:
   Create the data directory structure:
   ```bash
   mkdir -p data/raw/{git,python,docker,aws}
   ```

   Add your documents (JSON or text files) to the domain directories:
   ```
   data/raw/
   ├── git/
   │   ├── commands.json
   │   └── workflows.txt
   ├── python/
   │   ├── stdlib.json
   │   └── best_practices.txt
   ├── docker/
   └── aws/
   ```

### Model Downloads

The first run will automatically download the following models (requires ~10GB disk space):

- BAAI/bge-m3 (~2.3GB)
- BAAI/bge-reranker-v2-m3 (~1.7GB)
- BAAI/bge-reranker-large (~1.3GB)

## Quick Start

### Run End-to-End Smoke Test

```bash
python smoke_test.py
```

This validates the entire pipeline:
1. Data preparation
2. Index building
3. Question answering with sample questions

### Manual Workflow

```bash
# 1. Prepare chunks from raw data
python data_prep.py --config config/base.yaml

# 2. Build vector index
python index_builder.py --config config/base.yaml

# 3. Answer a question
python answerer.py "Git의 브랜치란 무엇인가요?" --config config/base.yaml
```

## Usage

### Data Preparation

Convert raw documents into chunked text for indexing.

**Command**:
```bash
python data_prep.py [OPTIONS]
```

**Options**:
- `--config`: Path to config file (default: `config/base.yaml`)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

**Example**:
```bash
python data_prep.py --log-level DEBUG
```

**Output**:
- `artifacts/chunks.parquet`: Parquet file with columns:
  - `domain`: Source domain (git, python, etc.)
  - `chunk_id`: Unique identifier
  - `text`: Chunk content
  - `length`: Character count

**Logs**:
```
2025-01-15 10:23:45 - __main__ - INFO - Initializing DataPreparer
2025-01-15 10:23:45 - __main__ - INFO - Target domains: ['git', 'python', 'docker', 'aws']
2025-01-15 10:23:46 - __main__ - INFO - Domain 'git': loaded 12 files, 45 text segments
2025-01-15 10:23:46 - __main__ - INFO - Domain 'git': 128 chunks created in 0.34s (fallback=no)
2025-01-15 10:23:47 - __main__ - INFO - Data preparation complete: 512 total chunks
```

### Index Building

Generate embeddings and build ChromaDB vector index.

**Command**:
```bash
python index_builder.py [OPTIONS]
```

**Options**:
- `--config`: Path to config file
- `--log-level`: Logging level

**Example**:
```bash
python index_builder.py
```

**Output**:
- `artifacts/chroma_db/`: ChromaDB persistent storage

**Logs**:
```
2025-01-15 10:25:12 - __main__ - INFO - Loading embedding model BAAI/bge-m3 on device: cuda
2025-01-15 10:25:18 - __main__ - INFO - Embedding model loaded in 6.23s
2025-01-15 10:25:19 - __main__ - INFO - Processing 512 chunks in 1 batches of 512
2025-01-15 10:25:22 - __main__ - INFO - Batch 1/1: Encoding 512 chunks
2025-01-15 10:25:35 - __main__ - INFO - Batch 1/1: Completed in 13.45s
2025-01-15 10:25:35 - __main__ - INFO - Total time: 16.78s
```

### Question Answering

Interactive CLI for asking questions.

**Command**:
```bash
python answerer.py "YOUR_QUESTION" [OPTIONS]
```

**Options**:
- `--config`: Path to config file
- `--log-level`: Logging level

**Example**:
```bash
python answerer.py "Docker 컨테이너와 이미지의 차이는?" --log-level INFO
```

**Output**:
```
2025-01-15 10:30:15 - __main__ - INFO - Query routing: difficulty=easy, strategy=single
2025-01-15 10:30:15 - __main__ - INFO - Total retrieval time: 0.842s
2025-01-15 10:30:17 - __main__ - INFO - LLM response received in 1.523s
2025-01-15 10:30:17 - __main__ - INFO - Total answer time: 2.365s

Docker 이미지는 애플리케이션과 그 실행 환경을 패키징한 읽기 전용 템플릿이고,
컨테이너는 이 이미지를 기반으로 실행되는 인스턴스입니다. (근거 1, 근거 2)
```

### API Server

Production REST API with FastAPI.

**Start Server**:
```bash
python serve.py [OPTIONS]
```

**Options**:
- `--config`: Path to config file
- `--host`: Host to bind (default: `0.0.0.0`)
- `--port`: Port to bind (default: `8000`)
- `--reload`: Enable auto-reload for development
- `--log-level`: Logging level

**Example**:
```bash
# Production
python serve.py --port 8000

# Development with auto-reload
python serve.py --reload --log-level DEBUG
```

**API Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/ask` | POST | Answer question |
| `/docs` | GET | Interactive API docs |

**Example Request**:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Python의 데코레이터는 무엇인가요?",
    "return_contexts": true,
    "return_metadata": true
  }'
```

**Example Response**:
```json
{
  "question": "Python의 데코레이터는 무엇인가요?",
  "answer": "데코레이터는 함수나 클래스를 수정하지 않고 기능을 추가하는 Python의 디자인 패턴입니다...",
  "contexts": [
    "Python 데코레이터는 @symbol로 표시되며...",
    "데코레이터 패턴은 함수를 인자로 받아..."
  ],
  "metadata": {
    "routing": {
      "difficulty": "medium",
      "strategy": "single",
      "reason": "Medium-complexity technical concept"
    },
    "num_contexts_retrieved": 5,
    "timing": {
      "retrieval_sec": 0.823,
      "answer_sec": 1.456,
      "total_sec": 2.279
    }
  }
}
```

**Interactive Docs**:
Visit `http://localhost:8000/docs` for Swagger UI with live API testing.

### Batch Evaluation

Run systematic evaluations on multiple questions.

**Command**:
```bash
python evaluate.py [OPTIONS]
```

**Options**:
- `--questions`: Path to text file (one question per line)
- `--questions-json`: Path to JSON file
- `--config`: Path to config file
- `--output-dir`: Custom output directory
- `--log-level`: Logging level

**Example**:
```bash
# From text file
python evaluate.py --questions test_questions.txt

# From JSON file
python evaluate.py --questions-json questions.json --output-dir results/
```

**Input Format** (text file):
```
Git의 브랜치란 무엇인가요?
Docker 컨테이너의 개념을 설명해주세요.
Python의 가상환경은 왜 사용하나요?
```

**Input Format** (JSON):
```json
{
  "questions": [
    "Git의 브랜치란 무엇인가요?",
    "Docker 컨테이너의 개념을 설명해주세요?"
  ]
}
```

**Output** (`artifacts/evals/eval_TIMESTAMP.jsonl`):
```jsonl
{"question_id": 1, "question": "Git의 브랜치란...", "routing": {...}, "retrieved_chunks": [...], "answer": "...", "metadata": {...}}
{"question_id": 2, "question": "Docker 컨테이너의...", "routing": {...}, "retrieved_chunks": [...], "answer": "...", "metadata": {...}}
```

**Summary** (`artifacts/evals/eval_TIMESTAMP_summary.json`):
```json
{
  "total_questions": 10,
  "performance": {
    "avg_retrieval_time_sec": 0.845,
    "avg_answer_time_sec": 1.523,
    "avg_total_time_sec": 2.368
  },
  "retrieval_stats": {
    "avg_chunks_retrieved": 4.8,
    "questions_with_no_chunks": 0
  },
  "routing_distribution": {
    "by_difficulty": {"easy": 3, "medium": 5, "hard": 2},
    "by_strategy": {"single": 7, "multistep": 3}
  }
}
```

### Ragas Benchmark

Evaluate RAG pipeline quality using [Ragas](https://github.com/explodinggradients/ragas) metrics.

**Ragas Metrics** (5 core metrics):
- **Context Precision**: Relevance of retrieved contexts to the question
- **Context Recall**: Coverage of ground truth information in retrieved contexts
- **Faithfulness**: How faithful the answer is to the context (no hallucination)
- **Answer Relevancy**: Relevance of answer to the question
- **Answer Correctness**: Accuracy of answer compared to ground truth

**Command**:
```bash
python ragas_benchmark.py [OPTIONS]
```

**Options**:
- `--config`: Path to RAG config file (default: `config/base.yaml`)
- `--questions`: Path to questions JSON file (default: `ragas_questions.json`)
- `--output-dir`: Output directory (default: `artifacts/ragas_evals`)
- `--output-name`: Custom output file name

**Example**:
```bash
# Run benchmark with default questions
python ragas_benchmark.py

# Use custom questions
python ragas_benchmark.py --questions my_questions.json --output-name my_eval
```

**Questions File Format** (`ragas_questions.json`):
```json
{
  "questions": [
    {
      "id": 1,
      "difficulty": "easy",
      "domain": "git",
      "question": "Git에서 브랜치를 생성하는 명령어는 무엇인가요?",
      "ground_truth": "git branch <브랜치명> 명령어로 새로운 브랜치를 생성할 수 있습니다...",
      "reference_context": ["git branch 명령어는...", "..."]
    }
  ]
}
```

**Included Questions**:
The default `ragas_questions.json` includes 15 curated questions:
- **5 Easy**: Basic factual questions (commands, definitions)
- **6 Medium**: Conceptual understanding (comparisons, use cases)
- **4 Hard**: Complex integration scenarios (multi-step reasoning)
- **Domains**: git, python, docker, aws, integration

**Output Files**:

1. **JSON Results** (`ragas_eval_TIMESTAMP.json`):
```json
{
  "timestamp": "20251119_031500",
  "num_questions": 15,
  "summary": {
    "context_precision": 0.8245,
    "context_recall": 0.7891,
    "faithfulness": 0.9123,
    "answer_relevancy": 0.8567,
    "answer_correctness": 0.8234
  },
  "detailed_results": [...]
}
```

2. **CSV Results** (`ragas_eval_TIMESTAMP.csv`):
```csv
question,answer,contexts,ground_truth,context_precision,context_recall,faithfulness,answer_relevancy,answer_correctness,id,difficulty,domain
```

3. **Report** (`ragas_eval_TIMESTAMP_report.txt`):
```
======================================================================
RAGAS EVALUATION REPORT
======================================================================

OVERALL METRICS
----------------------------------------------------------------------
Context Precision:   0.8245
Context Recall:      0.7891
Faithfulness:        0.9123
Answer Relevancy:    0.8567
Answer Correctness:  0.8234

METRICS BY DIFFICULTY
----------------------------------------------------------------------
EASY (5 questions):
  Context Precision:   0.8654
  Context Recall:      0.8234
  Faithfulness:        0.9345
  Answer Relevancy:    0.8912
  Answer Correctness:  0.8756

MEDIUM (6 questions):
  Context Precision:   0.8123
  Context Recall:      0.7789
  Faithfulness:        0.9012
  Answer Relevancy:    0.8456
  Answer Correctness:  0.8123

HARD (4 questions):
  Context Precision:   0.7912
  Context Recall:      0.7456
  Faithfulness:        0.8934
  Answer Relevancy:    0.8234
  Answer Correctness:  0.7891
```

**Installation**:
```bash
# Ragas dependencies already in requirements.txt
pip install -r requirements.txt

# Or install separately
pip install ragas datasets
```

**Notes**:
- Ragas evaluation requires an OpenAI API key (uses GPT for metric computation)
- First run will download Ragas models (~500MB)
- Evaluation takes ~5-10 minutes for 15 questions
- Results are saved to `artifacts/ragas_evals/` directory

**Interpreting Results**:
- **0.9-1.0**: Excellent - production ready
- **0.8-0.9**: Good - minor improvements needed
- **0.7-0.8**: Fair - needs optimization
- **< 0.7**: Poor - major issues

## Configuration

Edit [config/base.yaml](config/base.yaml) to customize:

```yaml
project:
  name: dev-helper-rag
  artifacts_dir: experiments/rag_pipeline/artifacts

data:
  raw_dir: data/raw
  domains: [git, python, docker, aws]

chunking:
  primary:
    chunk_size: 1024
    chunk_overlap: 150
  fallback:
    chunk_size: 256
    chunk_overlap: 50

embedding:
  model_name: BAAI/bge-m3
  device: auto  # auto, cuda, cuda:0, cpu
  batch_size: 32
  max_length: 1024

retrieval:
  dense_top_k: 25
  rerank_top_k: 5
  rerankers:
    stage1:
      model_name: BAAI/bge-reranker-v2-m3
      device: cuda:0
    stage2:
      model_name: BAAI/bge-reranker-large
      device: cuda:1  # Use different GPU if available

llm:
  provider: openai
  model_name: gpt-4.1
  max_new_tokens: 300
  temperature: 0.2
  top_p: 0.9
  system_prompt_path: experiments/rag_pipeline/prompts/system.txt
```

### Key Configuration Parameters

- **chunk_size**: Target size for text chunks (characters)
- **chunk_overlap**: Overlap between consecutive chunks
- **dense_top_k**: Number of documents to retrieve before reranking
- **rerank_top_k**: Final number of documents sent to LLM
- **temperature**: LLM sampling temperature (lower = more deterministic)

## Logging

All scripts support structured logging with timing metrics.

**Log Levels**:
- `DEBUG`: Detailed debugging information
- `INFO`: General progress and timing (default)
- `WARNING`: Warnings and fallback usage
- `ERROR`: Errors and exceptions

**Example Log Output**:
```
2025-01-15 10:30:15 - __main__ - INFO - Answering question: Docker 컨테이너와 이미지의 차이는?...
2025-01-15 10:30:15 - __main__ - INFO - Query routing: difficulty=easy, strategy=single, reason=Simple comparison question
2025-01-15 10:30:15 - __main__ - INFO - Starting retrieval for question: Docker 컨테이너와 이미지의 차이는?...
2025-01-15 10:30:15 - __main__ - INFO - Query encoding took 0.045s
2025-01-15 10:30:15 - __main__ - INFO - Dense retrieval: retrieved 25 documents in 0.123s
2025-01-15 10:30:15 - __main__ - INFO - Stage 1 reranking: 25 documents in 0.456s
2025-01-15 10:30:15 - __main__ - INFO - Stage 2 reranking: 5 documents in 0.218s
2025-01-15 10:30:15 - __main__ - INFO - Total retrieval time: 0.842s
2025-01-15 10:30:16 - __main__ - INFO - Calling LLM: model=gpt-4.1, temperature=0.2
2025-01-15 10:30:17 - __main__ - INFO - LLM response received in 1.523s
2025-01-15 10:30:17 - __main__ - INFO - Total answer time: 2.365s (retrieval + LLM)
```

**Redirecting Logs**:
```bash
# Save to file
python answerer.py "question" 2>&1 | tee logs/run.log

# JSON logs for monitoring (implement custom formatter)
python answerer.py "question" --log-level INFO > logs/structured.log
```

## Development

### Project Structure

```
experiments/rag_pipeline/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config/
│   └── base.yaml            # Configuration file
├── prompts/
│   └── system.txt           # LLM system prompt
├── data_prep.py             # Data preparation script
├── index_builder.py         # Index building script
├── answerer.py              # Question answering CLI
├── router.py                # Query routing logic
├── evaluate.py              # Batch evaluation script
├── serve.py                 # FastAPI server
├── smoke_test.py            # End-to-end testing
└── artifacts/               # Generated files (gitignored)
    ├── chunks.parquet       # Chunked documents
    ├── chroma_db/           # Vector index
    └── evals/               # Evaluation results
```

### Running Tests

```bash
# Full smoke test
python smoke_test.py

# Skip data prep (if already done)
python smoke_test.py --skip-prep

# Debug mode
python smoke_test.py --log-level DEBUG
```

### Adding New Data

1. Add documents to `data/raw/{domain}/`:
   ```bash
   echo "New content" > data/raw/git/new_doc.txt
   ```

2. Re-run data prep and indexing:
   ```bash
   python data_prep.py
   python index_builder.py
   ```

### Customizing System Prompt

Edit [prompts/system.txt](prompts/system.txt):
```
당신은 15년차 프롬프트 엔지니어이자 개발 학습 도우미 챗봇으로서,
제공된 근거만으로 정확하고 간결한 답변을 작성합니다.
답변에는 반드시 근거 문장을 인용하고, 추가 조사가 필요하면 명시하세요.
```

## Troubleshooting

### Common Issues

**1. OPENAI_API_KEY not set**
```
RuntimeError: OPENAI_API_KEY not set
```
**Solution**: Export your API key:
```bash
export OPENAI_API_KEY=sk-...
```

**2. CUDA out of memory**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Solution**: Use CPU or reduce batch size in config:
```yaml
embedding:
  device: cpu  # or cuda:0
  batch_size: 16  # reduce from 32
```

**3. No documents found**
```
WARNING: No documents retrieved from dense search
```
**Solution**: Check if:
- ChromaDB has documents: `python smoke_test.py --skip-prep --skip-index`
- Raw data exists: `ls data/raw/`
- Chunks were created: `python -c "import pandas as pd; print(len(pd.read_parquet('experiments/rag_pipeline/artifacts/chunks.parquet')))"`

**4. Model download fails**
```
OSError: Can't load model 'BAAI/bge-m3'
```
**Solution**: Check internet connection or download manually:
```bash
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

**5. Port already in use**
```
OSError: [Errno 98] Address already in use
```
**Solution**: Use different port:
```bash
python serve.py --port 8080
```

### Debugging Tips

1. **Enable DEBUG logging**:
   ```bash
   python answerer.py "question" --log-level DEBUG
   ```

2. **Check chunk quality**:
   ```python
   import pandas as pd
   df = pd.read_parquet('experiments/rag_pipeline/artifacts/chunks.parquet')
   print(df.head())
   print(df['length'].describe())
   ```

3. **Verify ChromaDB**:
   ```python
   import chromadb
   client = chromadb.PersistentClient(path='experiments/rag_pipeline/artifacts/chroma_db')
   collection = client.get_collection('rag_chunks')
   print(f"Documents: {collection.count()}")
   ```

4. **Test retrieval only**:
   ```python
   from answerer import RAGPipeline
   pipeline = RAGPipeline('config/base.yaml')
   contexts = pipeline.retrieve("Your question")
   for i, ctx in enumerate(contexts):
       print(f"{i+1}. {ctx[:100]}...")
   ```

## TODO

### Remaining Tasks

- [ ] Add unit tests for individual components
- [ ] Implement human evaluation workflow
- [ ] Add metrics dashboard (Streamlit/Grafana)
- [ ] Implement caching layer for frequent queries
- [ ] Add hybrid search (dense + sparse/BM25)
- [ ] Multi-language support (English, Japanese)
- [ ] Document versioning and change tracking
- [ ] A/B testing framework for prompt variations
- [ ] Rate limiting and authentication for API
- [ ] Containerization (Docker/Docker Compose)
- [ ] CI/CD pipeline for automated testing
- [ ] Production deployment guide (AWS/GCP/Azure)

### Performance Optimizations

- [ ] Batch embedding generation for faster indexing
- [ ] Async API endpoints for concurrent requests
- [ ] Model quantization (INT8/FP16) for inference
- [ ] Redis caching for embeddings
- [ ] Connection pooling for ChromaDB

---

## License

This project is for internal use. Ensure compliance with model licenses:
- BAAI models: MIT License
- OpenAI API: OpenAI Terms of Service

## Support

For questions or issues:
1. Check [Troubleshooting](#troubleshooting)
2. Review logs with `--log-level DEBUG`
3. Run `python smoke_test.py` to verify setup
4. Contact the ML team

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0
