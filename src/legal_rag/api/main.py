"""
FastAPI application for the Legal RAG Engine.
Exposes: POST /rag/query, POST /ingestion/run, GET /ingestion/status
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
import json
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from legal_rag.config import get_config
from legal_rag.engine import LegalRagEngine


logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Request / Response Schemas
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="The legal research query")
    filters: dict[str, Any] | None = Field(None, description="Optional metadata filters (e.g. category)")
    model_mode: Literal["quality", "fast"] = Field("quality", description="Quality model vs Fast model mode")
    conversation_context: list[dict[str, Any]] | None = Field(None, description="Recent chat history context")


QueryRequest.model_rebuild()



class IngestionRequest(BaseModel):
    corpus_path: str | None = Field(None, description="Override corpus path (optional)")
    single_file: str | None = Field(None, description="Ingest a single file (optional)")


class IngestionStatusResponse(BaseModel):
    status: str
    message: str


# ------------------------------------------------------------------ #
# App state
# ------------------------------------------------------------------ #

_engine: LegalRagEngine | None = None
_last_ingestion_report: Any = None
_ingestion_running: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG engine on startup."""
    global _engine
    logger.info("Legal RAG API starting up...")
    config = get_config()
    _engine = LegalRagEngine(config)
    logger.info("RAG Engine initialized.")
    yield
    logger.info("Legal RAG API shutting down.")


# ------------------------------------------------------------------ #
# App creation
# ------------------------------------------------------------------ #

def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal RAG Engine API",
        description=(
            "Legal Contract Analysis RAG System — retrieval-augmented generation "
            "over Indian legal statutes, contract rules, and case law."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- #
    # Routes
    # ---------------------------------------------------------------- #

    @app.get("/health", tags=["system"])
    async def health_check():
        return {"status": "ok", "engine_ready": _engine is not None}

    @app.post("/rag/query", tags=["rag"])
    async def rag_query(request: QueryRequest):
        """
        Run a legal query against the indexed corpus.
        Returns a grounded answer with citations.
        """
        if _engine is None:
            raise HTTPException(503, detail="Engine not initialized")
        try:
            response = _engine.query(
                request.query,
                filters=request.filters,
                conversation_context=request.conversation_context,
                model_mode=request.model_mode,
            )
            return response.model_dump()
        except Exception as e:
            logger.exception("Error during query: %s", e)
            raise HTTPException(500, detail=f"Query error: {str(e)}")

    @app.post("/rag/query/stream", tags=["rag"])
    async def rag_query_stream(request: QueryRequest):
        """
        Stream legal query execution via Server-Sent Events (SSE).
        Yields structured status updates, metadata, and answer token deltas.
        """
        if _engine is None:
            raise HTTPException(503, detail="Engine not initialized")

        def event_generator():
            try:
                for event in _engine.query_stream(
                    request.query,
                    filters=request.filters,
                    conversation_context=request.conversation_context,
                    model_mode=request.model_mode,
                ):
                    event_type = event.get("type", "data")
                    yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
            except Exception as err:
                logger.exception("Streaming error: %s", err)
                error_event = {"type": "error", "error": "Unable to complete this legal query. The legal knowledge service encountered an internal error. Please try again."}
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/ingestion/run", tags=["ingestion"])
    async def run_ingestion(request: IngestionRequest):
        """
        Trigger ingestion of the entire corpus or a single file.
        NOTE: This is synchronous in this version — for large corpora use the CLI.
        """
        global _last_ingestion_report, _ingestion_running
        if _engine is None:
            raise HTTPException(503, detail="Engine not initialized")
        if _ingestion_running:
            raise HTTPException(409, detail="Ingestion already in progress")

        _ingestion_running = True
        try:
            from pathlib import Path
            if request.single_file:
                parents, children = _engine.ingest_and_index_document(Path(request.single_file))
                return {"parents_indexed": len(parents), "children_indexed": len(children)}
            else:
                corpus_path = Path(request.corpus_path) if request.corpus_path else None
                report = _engine.run_ingestion(corpus_path)
                _last_ingestion_report = report.model_dump()
                return _last_ingestion_report
        except Exception as e:
            logger.exception("Ingestion error: %s", e)
            raise HTTPException(500, detail=f"Ingestion error: {str(e)}")
        finally:
            _ingestion_running = False

    @app.get("/ingestion/status", tags=["ingestion"])
    async def get_ingestion_status():
        """Return status of the last ingestion run."""
        if _ingestion_running:
            return IngestionStatusResponse(status="running", message="Ingestion is in progress")
        if _last_ingestion_report:
            summary = _last_ingestion_report.get("summary", {})
            return IngestionStatusResponse(
                status="completed",
                message=f"Last run completed. Summary: {summary}",
            )
        return IngestionStatusResponse(status="idle", message="No ingestion has been run yet")

    return app


# Default app instance
app = create_app()
