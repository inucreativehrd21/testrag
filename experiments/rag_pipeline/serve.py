"""
FastAPI serving endpoint for RAG pipeline.

This provides a REST API for the RAG question-answering system with the
pipeline kept warm in memory for fast responses.

Usage:
    # Start server
    python serve.py

    # Start with custom config
    python serve.py --config config/custom.yaml

    # Start on custom port
    python serve.py --port 8080

    # Test with curl
    curl -X POST http://localhost:8000/ask \\
        -H "Content-Type: application/json" \\
        -d '{"question": "Git의 브랜치란 무엇인가요?"}'
"""

import argparse
import logging
import time
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from answerer import RAGPipeline

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="REST API for development Q&A using RAG",
    version="1.0.0"
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (kept warm)
pipeline: Optional[RAGPipeline] = None


class AskRequest(BaseModel):
    """Request model for /ask endpoint."""
    question: str = Field(..., min_length=1, description="Question to answer")
    return_contexts: bool = Field(default=False, description="Include retrieved contexts in response")
    return_metadata: bool = Field(default=False, description="Include metadata (timing, routing) in response")


class AskResponse(BaseModel):
    """Response model for /ask endpoint."""
    question: str
    answer: str
    contexts: Optional[List[str]] = None
    metadata: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    pipeline_loaded: bool
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline when the server starts."""
    global pipeline
    logger.info("Starting up RAG Pipeline API server")
    config_path = app.state.config_path

    try:
        logger.info(f"Loading RAG pipeline with config: {config_path}")
        start_time = time.time()
        pipeline = RAGPipeline(config_path)
        load_time = time.time() - start_time
        logger.info(f"Pipeline loaded successfully in {load_time:.2f}s")
    except Exception as e:
        logger.exception(f"Failed to load pipeline: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if pipeline is None:
        return HealthResponse(
            status="unhealthy",
            pipeline_loaded=False,
            message="Pipeline not loaded"
        )
    return HealthResponse(
        status="healthy",
        pipeline_loaded=True,
        message="Pipeline ready"
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Answer a question using the RAG pipeline.

    Args:
        request: AskRequest with question and optional flags

    Returns:
        AskResponse with answer and optional contexts/metadata
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    logger.info(f"Received question: {request.question[:100]}...")
    start_time = time.time()

    try:
        # Route the query
        decision = pipeline.router.classify(request.question)

        # Retrieve contexts
        retrieval_start = time.time()
        contexts = pipeline.retrieve(request.question)
        retrieval_time = time.time() - retrieval_start

        # Generate answer
        answer_start = time.time()
        if contexts:
            context_block = "\n\n".join(f"근거 {i+1}: {chunk}" for i, chunk in enumerate(contexts))
            messages = [
                {"role": "system", "content": pipeline.system_prompt},
                {"role": "user", "content": f"질문 유형: {decision.reason}\n질문: {request.question}\n\n컨텍스트:\n{context_block}\n\n지침: 근거를 인용하며 한국어로 답변하고, 추가 확인이 필요하면 명시하세요."}
            ]

            response = pipeline.llm_client.chat.completions.create(
                model=pipeline.llm_cfg["model_name"],
                messages=messages,
                temperature=pipeline.llm_cfg.get("temperature", 0.2),
                top_p=pipeline.llm_cfg.get("top_p", 0.9),
                max_tokens=pipeline.llm_cfg.get("max_new_tokens", 300)
            )
            answer = response.choices[0].message.content.strip()
        else:
            answer = "관련 문서를 찾지 못했습니다."

        answer_time = time.time() - answer_start
        total_time = time.time() - start_time

        # Build response
        response_data = AskResponse(
            question=request.question,
            answer=answer
        )

        if request.return_contexts:
            response_data.contexts = contexts

        if request.return_metadata:
            response_data.metadata = {
                "routing": {
                    "difficulty": decision.difficulty,
                    "strategy": decision.strategy,
                    "reason": decision.reason,
                },
                "num_contexts_retrieved": len(contexts),
                "timing": {
                    "retrieval_sec": round(retrieval_time, 3),
                    "answer_sec": round(answer_time, 3),
                    "total_sec": round(total_time, 3),
                }
            }

        logger.info(f"Request completed in {total_time:.3f}s")
        return response_data

    except Exception as e:
        logger.exception(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline API Server")
    parser.add_argument("--config", default="config/base.yaml", help="Path to config file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    return parser.parse_args()


def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    # Store config path in app state for startup
    app.state.config_path = args.config

    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"API documentation available at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower()
    )
