import json
from rag_pipeline.config import settings
from rag_pipeline.llm import get_judge_client
from rag_pipeline.state import PipelineState, EvalScores

_client = get_judge_client()

SYSTEM_PROMPT = """You are an answer quality evaluator for a plant floor documentation RAG system.
Score the provided answer across four dimensions. Each score must be a float between 0.0 and 1.0.

Respond with ONLY a JSON object in this exact format:
{
  "answer_relevance": <float>,
  "context_relevance": <float>,
  "groundedness": <float>,
  "completeness": <float>
}

Scoring criteria:
- answer_relevance: Does the answer directly address the question?
- context_relevance: Is the retrieved context relevant to the question?
- groundedness: Is the answer fully supported by the provided context (no hallucination)?
- completeness: Does the answer cover all aspects of the question given the context?"""


def evaluate_answer(state: PipelineState) -> PipelineState:
    if state.short_circuit:
        return state

    context_parts = [f"[Chunk {i+1}]\n{chunk.content}" for i, chunk in enumerate(state.chunks)]
    context_text = "\n\n".join(context_parts)

    user_message = (
        f"Question: {state.validated_query}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Answer: {state.answer}"
    )

    response = _client.chat.completions.create(
        model=settings.active_judge_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)

    scores = EvalScores(
        answer_relevance=float(data.get("answer_relevance", 0.0)),
        context_relevance=float(data.get("context_relevance", 0.0)),
        groundedness=float(data.get("groundedness", 0.0)),
        completeness=float(data.get("completeness", 0.0)),
    )
    state.eval_scores = scores

    flagged = scores.below_threshold(settings.eval_threshold)
    state.eval_flagged = flagged

    state.reasoning_trace.append(
        f"Evaluator: answer_relevance={scores.answer_relevance:.2f}, "
        f"context_relevance={scores.context_relevance:.2f}, "
        f"groundedness={scores.groundedness:.2f}, "
        f"completeness={scores.completeness:.2f}, "
        f"flagged={flagged}."
    )

    if flagged and state.retry_count < settings.max_retries:
        state.retry_count += 1
        state.short_circuit = False  # allow retry edge back to Generator

    return state
