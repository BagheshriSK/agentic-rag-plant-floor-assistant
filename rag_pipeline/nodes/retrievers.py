from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from rag_pipeline.config import settings
from rag_pipeline.state import PipelineState, ChunkSource

_chroma = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)

RRF_K = 60  # standard RRF constant


def _rrf_merge(
    bm25_ids: list[str],
    vector_ids: list[str],
    top_k: int,
    k: int = RRF_K,
) -> list[str]:
    """Merge two ranked lists of document IDs using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(vector_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda d: scores[d], reverse=True)
    return ranked[:top_k]


def _retrieve(state: PipelineState, collection_name: str) -> PipelineState:
    if state.short_circuit:
        return state

    collection = _chroma.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Fetch all documents for BM25 corpus
    all_data = collection.get(include=["documents", "metadatas"])
    all_ids: list[str] = all_data.get("ids", [])
    all_docs: list[str] = all_data.get("documents", []) or []
    all_metas: list[dict] = all_data.get("metadatas", []) or []

    if not all_ids:
        state.answer = "No relevant documentation found for your query."
        state.short_circuit = True
        state.reasoning_trace.append(
            f"Retriever: no documents in collection '{collection_name}'."
        )
        return state

    # BM25 search
    tokenized_corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = state.validated_query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[: settings.top_k]
    bm25_ids = [all_ids[i] for i in bm25_ranked_indices]

    # Vector search
    vector_results = collection.query(
        query_texts=[state.validated_query],
        n_results=min(settings.top_k, len(all_ids)),
        include=["documents", "metadatas", "distances"],
    )
    vector_ids: list[str] = vector_results.get("ids", [[]])[0]

    # RRF merge
    merged_ids = _rrf_merge(bm25_ids, vector_ids, settings.top_k)

    if not merged_ids:
        state.answer = "No relevant documentation found for your query."
        state.short_circuit = True
        state.reasoning_trace.append(
            f"Retriever: hybrid search returned no results from '{collection_name}'."
        )
        return state

    # Build id → (doc, meta) lookup from all_data
    id_to_doc = {doc_id: (doc, meta) for doc_id, doc, meta in zip(all_ids, all_docs, all_metas)}

    chunks: list[ChunkSource] = []
    for doc_id in merged_ids:
        if doc_id not in id_to_doc:
            continue
        doc_text, meta = id_to_doc[doc_id]
        chunks.append(
            ChunkSource(
                collection=collection_name,
                document=meta.get("document", "unknown"),
                section=meta.get("section", "unknown"),
                chunk_id=meta.get("chunk_id", doc_id),
                content=doc_text,
            )
        )

    state.chunks = chunks
    state.reasoning_trace.append(
        f"Retriever: retrieved {len(chunks)} chunks from '{collection_name}' via hybrid search."
    )
    return state


def retrieve_safety(state: PipelineState) -> PipelineState:
    return _retrieve(state, settings.collection_safety)


def retrieve_maintenance(state: PipelineState) -> PipelineState:
    return _retrieve(state, settings.collection_maintenance)


def retrieve_quality(state: PipelineState) -> PipelineState:
    return _retrieve(state, settings.collection_quality)
