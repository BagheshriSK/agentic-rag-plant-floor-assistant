from typing import Literal, Optional
from pydantic import BaseModel, Field


Category = Literal["safety_procedures", "maintenance_manuals", "quality_control_standards", "unknown"]


class ChunkSource(BaseModel):
    collection: str
    document: str
    section: str
    chunk_id: str
    content: str


class EvalScores(BaseModel):
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    groundedness: float = 0.0
    completeness: float = 0.0

    def below_threshold(self, threshold: float) -> bool:
        return any(
            score < threshold
            for score in [self.answer_relevance, self.context_relevance, self.groundedness, self.completeness]
        )


class PipelineState(BaseModel):
    # input
    raw_query: str = ""
    validated_query: str = ""

    # routing
    category: Optional[Category] = None

    # retrieval
    chunks: list[ChunkSource] = Field(default_factory=list)

    # generation
    answer: str = ""
    cot_reasoning: str = ""
    conversation_history: list[dict] = Field(default_factory=list)
    retry_count: int = 0

    # evaluation
    eval_scores: Optional[EvalScores] = None
    eval_flagged: bool = False

    # output
    final_response: Optional[dict] = None
    reasoning_trace: list[str] = Field(default_factory=list)

    # control
    error: Optional[str] = None
    short_circuit: bool = False
