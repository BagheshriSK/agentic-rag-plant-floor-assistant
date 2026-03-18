"""CLI entry point for the LangGraph RAG pipeline.

Usage:
    python main.py "your question here"
    python main.py --interactive
"""
import argparse
import uuid

from rag_pipeline.graph import pipeline


def pretty_print(response: dict) -> None:
    if "error" in response:
        print(f"\n[ERROR] {response['error']}")
        return

    print(f"\nAnswer:\n{response.get('answer', '')}")

    reasoning = response.get("reasoning_steps", "")
    if reasoning:
        print(f"\nReasoning Steps:\n{reasoning}")

    sources = response.get("sources", [])
    if sources:
        print("\nSources:")
        for s in sources:
            print(f"  [{s.get('collection', '')}] {s.get('document', '')} / {s.get('section', '')} ({s.get('chunk_id', '')})")

    scores = response.get("eval_scores")
    if scores:
        print("\nEvaluation Scores:")
        for k, v in scores.items():
            print(f"  {k}: {v:.2f}")

    if response.get("eval_flagged"):
        print("\n[WARNING] Answer quality flagged as below threshold.")


def run_query(query: str, thread_id: str) -> dict:
    result = pipeline.invoke(
        {"raw_query": query},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result.get("final_response") or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plant Floor Documentation Assistant")
    parser.add_argument("query", nargs="?", help="Question to ask the pipeline")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    thread_id = str(uuid.uuid4())

    if args.interactive:
        print("Plant Floor Documentation Assistant (interactive mode)")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye.")
                break
            response = run_query(query, thread_id)
            pretty_print(response)
            print()
    elif args.query:
        response = run_query(args.query, thread_id)
        pretty_print(response)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
