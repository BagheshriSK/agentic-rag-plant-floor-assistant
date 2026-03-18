from rag_pipeline.config import settings
from rag_pipeline.llm import get_chat_client
from rag_pipeline.state import PipelineState

_client = get_chat_client()

SYSTEM_PROMPT = """You are a plant floor documentation assistant.
First, output your reasoning steps prefixed with "Reasoning:" on its own line.
Then output the final answer prefixed with "Answer:" on its own line.
Answer ONLY from the provided context.
If the context does not contain enough information, respond with:
"Answer: I don't know based on the provided documentation."
Do not infer, fabricate, or use knowledge outside the provided context."""


def generate_answer(state: PipelineState) -> PipelineState:
    if state.short_circuit:
        return state

    # Build conversation history context (last N turns)
    window = settings.conversation_history_window
    history_turns = state.conversation_history[-window:] if state.conversation_history else []
    history_parts = []
    for turn in history_turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        history_parts.append(f"{role.capitalize()}: {content}")
    history_text = "\n".join(history_parts)

    # Build context from chunks
    context_parts = [f"[Chunk {i+1}]\n{chunk.content}" for i, chunk in enumerate(state.chunks)]
    context_text = "\n\n".join(context_parts)

    # Assemble user message
    user_message_parts = []
    if history_text:
        user_message_parts.append(f"Conversation history:\n{history_text}")
    user_message_parts.append(f"Context:\n{context_text}")
    user_message_parts.append(f"Question: {state.validated_query}")
    user_message = "\n\n".join(user_message_parts)

    response = _client.chat.completions.create(
        model=settings.active_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    raw_response = response.choices[0].message.content.strip()

    # Parse CoT response
    if "Answer:" in raw_response:
        answer_split = raw_response.split("Answer:", 1)
        reasoning_part = answer_split[0]
        answer_part = answer_split[1].strip()

        # Extract reasoning text (strip "Reasoning:" prefix if present)
        if "Reasoning:" in reasoning_part:
            cot_part = reasoning_part.split("Reasoning:", 1)[1].strip()
        else:
            cot_part = reasoning_part.strip()

        state.cot_reasoning = cot_part
        state.answer = answer_part
    else:
        state.cot_reasoning = ""
        state.answer = raw_response

    # Append to conversation history
    state.conversation_history.append({"role": "user", "content": state.validated_query})
    state.conversation_history.append({"role": "assistant", "content": state.answer})

    state.reasoning_trace.append("Generator: CoT answer generated from retrieved context.")
    return state
