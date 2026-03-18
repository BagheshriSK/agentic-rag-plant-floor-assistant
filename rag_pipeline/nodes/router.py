from rag_pipeline.config import settings
from rag_pipeline.llm import get_classifier_client
from rag_pipeline.state import PipelineState

_client = get_classifier_client()

VALID_ROUTES = {"safety_procedures", "maintenance_manuals", "quality_control_standards"}

SYSTEM_PROMPT = """You are a routing agent for a plant floor documentation system.
Determine which documentation collection should handle the user query.
Respond with ONLY one of these routing keys, nothing else:
- safety_procedures
- maintenance_manuals
- quality_control_standards"""


def route_query(state: PipelineState) -> str:
    """LLM-based conditional edge for routing queries to the correct retriever."""
    if state.short_circuit:
        state.reasoning_trace.append("Router: short-circuit active, routing to output.")
        return "output"

    response = _client.chat.completions.create(
        model=settings.active_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state.validated_query},
        ],
        temperature=0,
        max_tokens=20,
    )

    raw = response.choices[0].message.content.strip().lower()

    if raw in VALID_ROUTES:
        state.reasoning_trace.append(f"Router: LLM-derived route → '{raw}'.")
        return raw

    state.reasoning_trace.append(f"Router: LLM returned invalid key '{raw}', falling back to output.")
    return "output"
