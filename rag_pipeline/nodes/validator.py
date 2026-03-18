import re
from rag_pipeline.state import PipelineState


def validate_input(state: PipelineState) -> PipelineState:
    if state.short_circuit:
        return state

    query = state.raw_query.strip()

    if not query:
        state.error = "Query is empty."
        state.short_circuit = True
        return state

    if len(query) > 2000:
        state.error = "Query exceeds maximum length of 2000 characters."
        state.short_circuit = True
        return state

    # strip control characters, normalize whitespace
    query = re.sub(r"[\x00-\x1f\x7f]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    state.validated_query = query
    state.reasoning_trace.append(f"Validated query: '{query[:80]}...' " if len(query) > 80 else f"Validated query: '{query}'")
    return state
