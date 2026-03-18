# Plant Floor Documentation Assistant

An agentic RAG pipeline built with LangGraph for plant floor supervisors to ask natural language questions and get grounded answers from internal documentation.

## Architecture

```
Input Validator → Classifier → Router → Retriever → Generator → Evaluator → Output Formatter
```

- **LangGraph** — stateful multi-node pipeline with conditional edges and retry logic
- **ChromaDB** — 3 separate vector collections (safety, maintenance, quality control)
- **Hybrid Search** — BM25 + semantic search merged with Reciprocal Rank Fusion (RRF)
- **LLM Routing** — Groq `llama-3.3-70b-versatile` for classification, routing, generation, and evaluation
- **Chain-of-Thought** — reasoning steps exposed before the final answer
- **LLM-as-Judge** — evaluates answer relevance, context relevance, groundedness, completeness
- **Multi-turn Memory** — LangGraph `MemorySaver` for conversation history
- **Streamlit UI** — chat interface with source attribution, reasoning trace, eval scores, PDF upload

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

Or with pyproject.toml:
```bash
pip install -e .
```

### 2. Set up environment
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 3. Populate the knowledge bases
```bash
python scripts/populate_db.py
```
This generates 9 synthetic PDFs (3 per domain) and ingests them into ChromaDB.

### 4. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501

## Sample Questions

**Safety:** "What PPE is required when working near chemical hazards?"  
**Maintenance:** "How do I perform scheduled maintenance on a conveyor motor?"  
**Quality:** "What are the acceptance criteria for incoming raw material inspection?"  
**Follow-up:** "Can you give me more detail on that?"

## Project Structure

```
rag_pipeline/
  nodes/          # classifier, router, retrievers, generator, evaluator, output_formatter
  config.py       # settings via pydantic-settings
  graph.py        # LangGraph pipeline definition
  llm.py          # LLM client factory (Groq / OpenAI / HuggingFace fallback)
  state.py        # PipelineState, ChunkSource, EvalScores
scripts/
  generate_pdfs.py   # synthetic PDF generation (reportlab)
  ingest_pdfs.py     # PDF chunking + ChromaDB ingestion
  populate_db.py     # convenience: generate + ingest all collections
app.py            # Streamlit UI
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key (required) | — |
| `GROQ_MODEL` | Model name | `llama-3.3-70b-versatile` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./chroma_db` |
| `TOP_K` | Chunks retrieved per query | `5` |
| `EVAL_THRESHOLD` | Min score before retry | `0.6` |

## Spec-Driven Development

This project was built using a spec-driven development workflow, where requirements, design, and implementation tasks were fully defined before any code was written. The spec lives in `.kiro/specs/langgraph-rag-pipeline/`.

### Workflow

```
Requirements → Design → Tasks → Implementation
```

### 1. Requirements (`requirements.md`)
16 requirements were defined upfront covering all pipeline nodes, data models, and non-functional concerns. Each requirement includes a user story and acceptance criteria. Key decisions captured at this stage:
- LLM-based classification and routing (not deterministic rules)
- Hybrid search: BM25 + ChromaDB + RRF
- Chain-of-thought prompting with `Reasoning:` / `Answer:` parsing
- Source attribution with collection field
- Multi-turn conversation via LangGraph MemorySaver
- Synthetic PDF dataset for testing
- Streamlit UI with PDF upload

### 2. Design (`design.md`)
A full architecture document covering:
- All 7 node contracts (inputs, outputs, side effects)
- Complete data models (`PipelineState`, `ChunkSource`, `EvalScores`)
- 17 correctness properties for property-based testing
- Error handling table per node
- Hybrid search algorithm (BM25 + vector + RRF)
- Retry logic flow for the evaluator

### 3. Tasks (`tasks.md`)
21 implementation tasks derived directly from the design, each mapped to specific files and acceptance criteria. This made it straightforward to track progress and ensured nothing was missed.

### Why Spec-First?
Writing the spec before coding forced clarity on architecture trade-offs (e.g. 3 separate collections vs. 1 with metadata filters, LLM routing vs. deterministic routing) before they became expensive to change. It also produced documentation that accurately reflects what was built, rather than documentation written after the fact.
