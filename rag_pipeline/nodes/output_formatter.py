from rag_pipeline.state import PipelineState


def format_output(state: PipelineState) -> PipelineState:
    """Always executes — assembles final_response regardless of short_circuit."""
    if state.error:
        state.final_response = {
            "error": state.error,
            "reasoning_trace": state.reasoning_trace,
        }
        return state

    sources = [
        {
            "collection": chunk.collection,
            "document": chunk.document,
            "section": chunk.section,
            "chunk_id": chunk.chunk_id,
        }
        for chunk in state.chunks
    ]

    state.final_response = {
        "answer": state.answer,
        "reasoning_steps": state.cot_reasoning,
        "sources": sources,
        "eval_scores": state.eval_scores.model_dump() if state.eval_scores else None,
        "eval_flagged": state.eval_flagged,
        "reasoning_trace": state.reasoning_trace,
    }
    return state
