from rag_pipeline.config import settings
from rag_pipeline.llm import get_classifier_client
from rag_pipeline.state import PipelineState, Category

_client = get_classifier_client()

SYSTEM_PROMPT = """You are a query classifier for a plant floor documentation system.
Classify the user query into exactly one of these categories:
- safety_procedures: questions about safety rules, PPE, hazard protocols, emergency procedures
- maintenance_manuals: questions about equipment maintenance, repair, troubleshooting, parts
- quality_control_standards: questions about QC checks, tolerances, inspection criteria, defect standards
- unknown: the query does not fit any of the above categories

IMPORTANT: If the conversation history shows a prior category, and the current query is a follow-up
(e.g. "tell me more", "what about section X", "can you summarize that", "what happens if I skip that"),
reuse the same category from the prior turn instead of returning unknown.

Respond with ONLY the category name, nothing else."""


def classify_query(state: PipelineState) -> PipelineState:
    if state.short_circuit:
        return state

    # Build recent history snippet to give the classifier context
    window = state.conversation_history[-4:] if state.conversation_history else []
    history_text = ""
    if window:
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in window]
        history_text = "Recent conversation:\n" + "\n".join(lines) + "\n\n"

    user_content = f"{history_text}Current query: {state.validated_query}"

    response = _client.chat.completions.create(
        model=settings.active_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=20,
    )

    raw = response.choices[0].message.content.strip().lower()
    valid: list[Category] = ["safety_procedures", "maintenance_manuals", "quality_control_standards", "unknown"]
    category: Category = raw if raw in valid else "unknown"  # type: ignore[assignment]

    # Last-resort fallback: if still unknown but we have a prior category, reuse it
    if category == "unknown" and state.category and state.category != "unknown":
        category = state.category
        state.reasoning_trace.append(f"Classifier: follow-up detected, reusing prior category '{category}'.")
    else:
        state.reasoning_trace.append(f"Classified as: {category}")

    state.category = category

    if category == "unknown":
        state.answer = "I'm sorry, I can only answer questions related to safety procedures, maintenance manuals, or quality control standards."
        state.short_circuit = True

    return state
