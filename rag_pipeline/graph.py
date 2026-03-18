from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rag_pipeline.state import PipelineState
from rag_pipeline.nodes.validator import validate_input
from rag_pipeline.nodes.classifier import classify_query
from rag_pipeline.nodes.router import route_query
from rag_pipeline.nodes.retrievers import retrieve_safety, retrieve_maintenance, retrieve_quality
from rag_pipeline.nodes.generator import generate_answer
from rag_pipeline.nodes.evaluator import evaluate_answer
from rag_pipeline.nodes.output_formatter import format_output


def _evaluator_edge(state: PipelineState) -> str:
    """Return 'retry' if the evaluator flagged and incremented retry_count, else 'done'."""
    # The evaluator already incremented retry_count and cleared short_circuit when retrying.
    # We detect a retry in progress by checking: flagged AND short_circuit is False AND retry_count > 0.
    if state.eval_flagged and not state.short_circuit and state.retry_count > 0:
        return "retry"
    return "done"


graph = StateGraph(PipelineState)

graph.add_node("input_validator", validate_input)
graph.add_node("classifier", classify_query)
graph.add_node("safety_retriever", retrieve_safety)
graph.add_node("maintenance_retriever", retrieve_maintenance)
graph.add_node("quality_retriever", retrieve_quality)
graph.add_node("generator", generate_answer)
graph.add_node("evaluator", evaluate_answer)
graph.add_node("output_formatter", format_output)

graph.set_entry_point("input_validator")
graph.add_edge("input_validator", "classifier")
graph.add_conditional_edges(
    "classifier",
    route_query,
    {
        "safety_procedures": "safety_retriever",
        "maintenance_manuals": "maintenance_retriever",
        "quality_control_standards": "quality_retriever",
        "output": "output_formatter",
    },
)
graph.add_edge("safety_retriever", "generator")
graph.add_edge("maintenance_retriever", "generator")
graph.add_edge("quality_retriever", "generator")
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges(
    "evaluator",
    _evaluator_edge,
    {
        "retry": "generator",
        "done": "output_formatter",
    },
)
graph.add_edge("output_formatter", END)

pipeline = graph.compile(checkpointer=MemorySaver())
