"""
Unified FastAPI Server for RAG Systems
- Supports both Optimized RAG and LangGraph RAG
- Compatible with EC2 Django backend
- Session-based chat history support

Usage:
    # Start with Optimized RAG (fast)
    python serve_unified.py --rag-type optimized --port 8080

    # Start with LangGraph RAG (high-quality)
    python serve_unified.py --rag-type langgraph --port 8080

    # Test
    curl -X POST http://localhost:8080/api/v1/chat \
        -H "Content-Type: application/json" \
        -d '{"question": "git rebase란?", "user_id": "user123"}'
"""

import argparse
import logging
import time
import sys
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified RAG API",
    description="REST API for RAG chatbot with Optimized RAG and LangGraph RAG support",
    version="2.0.0"
)

# CORS middleware for EC2 server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # EC2 서버 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance and configuration
rag_instance = None
rag_type = None
config_path = None


# === Request/Response Models ===

class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request from EC2 server"""
    question: str = Field(..., min_length=1, description="User question")
    user_id: str = Field(..., description="User ID from Django")
    user_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User personalization context")
    enable_personalization: bool = Field(default=True, description="Enable personalization features")
    chat_history: List[ChatMessage] = Field(default_factory=list, description="Previous conversation history")
    session_id: Optional[str] = Field(None, description="Chat session ID")


class Source(BaseModel):
    """Document source with metadata (Django compatible)"""
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Document title")
    domain: str = Field(default="unknown", description="Domain (python, git, general)")


class ChatResponse(BaseModel):
    """Chat response to EC2 server"""
    success: bool = Field(..., description="Success status")
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="Retrieved documents")
    related_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    rag_type: str
    rag_loaded: bool
    message: str


# === Optimized RAG Loader ===

def load_optimized_rag(config_path: str):
    """Load Optimized RAG system"""
    from answerer_v2_optimized import RAGPipeline

    logger.info(f"Loading Optimized RAG with config: {config_path}")
    start_time = time.time()

    pipeline = RAGPipeline(config_path)

    load_time = time.time() - start_time
    logger.info(f"Optimized RAG loaded in {load_time:.2f}s")

    return pipeline


# === LangGraph RAG Loader ===

def load_langgraph_rag(config_path: str):
    """Load LangGraph RAG system"""
    from langgraph_rag.graph import create_rag_graph
    from langgraph_rag.config import get_config

    logger.info(f"Loading LangGraph RAG with config: {config_path}")
    start_time = time.time()

    # Initialize config first
    _ = get_config(config_path)

    # Create graphs (personalized + plain)
    graphs = {
        "personalized": create_rag_graph(enable_personalization=True),
        "plain": create_rag_graph(enable_personalization=False),
    }

    load_time = time.time() - start_time
    logger.info(f"LangGraph RAG loaded in {load_time:.2f}s (personalized + plain)")

    return graphs


# === Chat Processing ===


def extract_title_from_url(url: str) -> str:
    """Extract human-friendly title from a URL (filename without extension)."""
    if not url or url == "unknown":
        return "Unknown Document"

    filename = url.split("/")[-1]
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]

    title = filename.replace("-", " ").replace("_", " ")
    return title.title() if title else "Unknown Document"


def process_optimized_rag(
    question: str,
    chat_history: List[ChatMessage],
    user_id: str,
    enable_personalization: bool = False,
    session_id: Optional[str] = None,
) -> ChatResponse:
    """Process request with Optimized RAG"""
    global rag_instance

    try:
        start_time = time.time()

        # Retrieve contexts
        logger.info(f"[Optimized RAG] Retrieving contexts for: {question[:50]}... (session={session_id})")
        contexts = rag_instance.retrieve(question)

        if not contexts:
            return ChatResponse(
                success=True,
                answer="관련 문서를 찾지 못했습니다. 질문을 다시 작성해주세요.",
                sources=[],
                metadata={"retrieval_failed": True}
            )

        # Generate answer
        logger.info(f"[Optimized RAG] Generating answer with {len(contexts)} contexts")

        # Build context block
        context_block = "\n\n".join(
            f"문서 {i+1}: {ctx}" for i, ctx in enumerate(contexts)
        )

        # Build messages with chat history
        messages = [
            {"role": "system", "content": rag_instance.system_prompt}
        ]

        # Add chat history
        for msg in chat_history[-5:]:  # Last 5 messages
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add current question with context
        messages.append({
            "role": "user",
            "content": f"질문: {question}\n\n참고 문서:\n{context_block}\n\n위 문서를 바탕으로 질문에 답변해주세요. 문서에 없는 내용은 추측하지 마세요."
        })

        # Call LLM
        response = rag_instance.llm_client.chat.completions.create(
            model=rag_instance.llm_cfg["model_name"],
            messages=messages,
            temperature=rag_instance.llm_cfg.get("temperature", 0.2),
            max_tokens=rag_instance.llm_cfg.get("max_new_tokens", 500)
        )

        answer = response.choices[0].message.content.strip()

        # Extract source URLs
        sources = []
        for i, ctx in enumerate(contexts):
            # Try to extract URL from context metadata
            source_url = None
            if hasattr(rag_instance.retriever, 'get_document_url'):
                source_url = rag_instance.retriever.get_document_url(ctx)

            url = source_url or "unknown"
            title = extract_title_from_url(url) if source_url else f"Document {i + 1}"

            sources.append(Source(
                url=url,
                title=title,
                domain="unknown"
            ))

        total_time = time.time() - start_time

        return ChatResponse(
            success=True,
            answer=answer,
            sources=sources,
            metadata={
                "rag_type": "optimized",
                "num_contexts": len(contexts),
                "response_time": round(total_time, 2),
                "session_id": session_id,
                "chat_history_length": len(chat_history) if chat_history else 0,
                "enable_personalization": enable_personalization,
            }
        )

    except Exception as e:
        logger.exception(f"[Optimized RAG] Error: {e}")
        return ChatResponse(
            success=False,
            answer="",
            sources=[],
            error=str(e)
        )


def process_langgraph_rag(
    question: str,
    chat_history: List[ChatMessage],
    user_id: str,
    user_context: Dict[str, Any] = None,
    enable_personalization: bool = True,
    session_id: Optional[str] = None,
) -> ChatResponse:
    """Process request with LangGraph RAG"""
    global rag_instance

    try:
        start_time = time.time()

        logger.info(f"[LangGraph RAG] Processing: {question[:50]}...")
        logger.info(f"[LangGraph RAG] User ID: {user_id}, Personalization: {enable_personalization}, Session: {session_id}")

        # Create initial state with personalization
        from langgraph_rag.state import create_initial_state
        initial_state = create_initial_state(
            question=question,
            user_id=user_id,
            user_context=user_context or {}
        )

        # Attach session/chat metadata for downstream nodes or logging (best-effort)
        if session_id:
            initial_state["session_id"] = session_id
        if chat_history:
            initial_state["chat_history"] = [msg.model_dump() for msg in chat_history]

        # Select graph based on personalization flag (preloaded at startup)
        graph = None
        if isinstance(rag_instance, dict):
            graph = rag_instance.get("personalized") if enable_personalization else rag_instance.get("plain")
            if graph is None:
                graph = rag_instance.get("personalized") or rag_instance.get("plain")
        else:
            graph = rag_instance

        if graph is None:
            raise ValueError("LangGraph instance not initialized")

        # Run LangGraph with personalization control
        result = graph.invoke(initial_state)

        answer = result.get("generation", "답변 생성에 실패했습니다.")

        # Get final documents (after reranking)
        final_documents = result.get("final_documents", [])
        final_metadatas = result.get("final_metadatas", [])

        # Fallback to regular documents if final is empty
        if not final_documents:
            final_documents = result.get("reranked_documents", [])
            final_metadatas = result.get("reranked_metadatas", [])

        if not final_documents:
            final_documents = result.get("documents", [])
            final_metadatas = result.get("metadatas", [])

        # Build sources (Django compatible format)
        sources = []
        for i, _ in enumerate(final_documents[:10]):  # Top 10
            # Get metadata if available
            metadata = final_metadatas[i] if i < len(final_metadatas) else {}
            url = metadata.get("url") or metadata.get("source") or metadata.get("link") or "unknown"
            domain = metadata.get("domain") or metadata.get("source_type") or "unknown"
            title = metadata.get("title") or (
                extract_title_from_url(url) if url != "unknown" else f"Document {i + 1}"
            )

            sources.append(Source(url=url, title=title, domain=domain))

        # Get related questions
        related_questions = result.get("related_questions", [])

        total_time = time.time() - start_time

        return ChatResponse(
            success=True,
            answer=answer,
            sources=sources,
            related_questions=related_questions,
            metadata={
                "rag_type": "langgraph",
                "workflow": result.get("workflow_history", []),
                "intent": result.get("intent", "unknown"),
                "route": result.get("route", "unknown"),
                "document_relevance": result.get("document_relevance"),
                "hallucination_grade": result.get("hallucination_grade"),
                "answer_usefulness": result.get("answer_usefulness"),
                "retry_count": result.get("retry_count", 0),
                "document_count": len(final_documents),
                "personalization": {
                    "reminder_added": result.get("reminder_added", False),
                    "related_items": len(result.get("related_selections", [])),
                },
                "response_time": round(total_time, 2),
                "session_id": session_id,
                "chat_history_length": len(chat_history) if chat_history else 0,
                "enable_personalization": enable_personalization,
            }
        )


    except Exception as e:
        logger.exception(f"[LangGraph RAG] Error: {e}")
        return ChatResponse(
            success=False,
            answer="",
            sources=[],
            error=str(e)
        )


# === API Endpoints ===

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on server start"""
    global rag_instance, rag_type, config_path

    logger.info("Starting Unified RAG API Server")

    # Load from environment variables if not set
    if config_path is None:
        config_path = os.environ.get('RAG_CONFIG_PATH', 'config/enhanced.yaml')
    if rag_type is None:
        rag_type = os.environ.get('RAG_TYPE', 'langgraph')

    logger.info(f"RAG Type: {rag_type}")
    logger.info(f"Config: {config_path}")

    try:
        if rag_type == "optimized":
            rag_instance = load_optimized_rag(config_path)
        elif rag_type == "langgraph":
            rag_instance = load_langgraph_rag(config_path)
        else:
            raise ValueError(f"Unknown RAG type: {rag_type}")

        logger.info(f"RAG system ({rag_type}) loaded successfully")

    except Exception as e:
        logger.exception(f"Failed to load RAG system: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "name": "Unified RAG API",
        "version": "2.0.0",
        "rag_type": rag_type,
        "endpoints": {
            "/api/v1/chat": "POST - Chat with RAG system",
            "/api/v1/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if rag_instance is None:
        return HealthResponse(
            status="unhealthy",
            rag_type=rag_type or "unknown",
            rag_loaded=False,
            message="RAG system not loaded"
        )

    return HealthResponse(
        status="healthy",
        rag_type=rag_type,
        rag_loaded=True,
        message="RAG system ready"
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat with RAG system

    This endpoint is called by the EC2 Django backend.

    Note: This is a sync function (not async) because the RAG processing
    functions are blocking. FastAPI will automatically run this in a thread pool.
    """
    if rag_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )

    logger.info(f"[Chat] User: {request.user_id}, Question: {request.question[:50]}...")

    try:
        # Route to appropriate RAG system
        if rag_type == "optimized":
            response = process_optimized_rag(
                request.question,
                request.chat_history,
                request.user_id,
                enable_personalization=request.enable_personalization,
                session_id=request.session_id,
            )
        elif rag_type == "langgraph":
            response = process_langgraph_rag(
                request.question,
                request.chat_history,
                request.user_id,
                user_context=request.user_context,
                enable_personalization=request.enable_personalization,
                session_id=request.session_id,
            )
        else:
            raise ValueError(f"Unknown RAG type: {rag_type}")

        logger.info(f"[Chat] Response generated: {len(response.answer)} chars")
        return response

    except Exception as e:
        logger.exception(f"[Chat] Error processing request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# === CLI ===

def parse_args():
    parser = argparse.ArgumentParser(description="Unified RAG API Server")
    parser.add_argument(
        "--rag-type",
        choices=["optimized", "langgraph"],
        default="langgraph",
        help="RAG system type (default: langgraph)"
    )
    parser.add_argument(
        "--config",
        default="config/enhanced.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    return parser.parse_args()


def setup_logging(level: str = "INFO"):
    """Configure logging"""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    # Store config in global variables
    config_path = args.config
    rag_type = args.rag_type

    # Also set environment variables for reload support
    os.environ['RAG_CONFIG_PATH'] = args.config
    os.environ['RAG_TYPE'] = args.rag_type

    logger.info(f"Starting Unified RAG API Server")
    logger.info(f"RAG Type: {args.rag_type}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Server: http://{args.host}:{args.port}")
    logger.info(f"API Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "serve_unified:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower()
    )
