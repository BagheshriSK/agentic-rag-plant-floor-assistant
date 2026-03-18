# Self Reflection — Plant Floor Documentation Assistant

## The Scenario

Plant floor supervisors often need quick answers from large volumes of internal documentation — safety procedures, maintenance manuals, and quality control standards. Searching through PDFs manually is slow and error-prone, especially in a time-sensitive environment. The challenge was to build an intelligent assistant that could route a natural language question to the right knowledge base and return a grounded, explainable answer.

---

## What I Built

A production-style agentic RAG (Retrieval-Augmented Generation) pipeline using LangGraph, with a Streamlit front-end. The system consists of seven pipeline nodes:

1. **Input Validator** — sanitizes and validates the raw user query
2. **Classifier** — uses an LLM to categorize the query into safety, maintenance, or quality control
3. **Router** — conditional edge that directs the query to the correct retriever
4. **Retriever** — hybrid search combining BM25 (keyword) and ChromaDB (semantic) with Reciprocal Rank Fusion (RRF) merging
5. **Generator** — chain-of-thought prompting with conversation history windowing
6. **Evaluator** — LLM-as-a-judge scoring across four dimensions: answer relevance, context relevance, groundedness, and completeness
7. **Output Formatter** — structures the final response with answer, sources, reasoning steps, and eval scores

Supporting infrastructure:
- **LangGraph** — pipeline orchestration, conditional edges, retry logic, and stateful graph execution
- **LangGraph MemorySaver** — in-memory checkpointer for multi-turn conversation history
- **ChromaDB** — three separate persistent vector collections (one per domain) with cosine similarity
- **rank-bm25** — BM25Okapi keyword search for the hybrid retrieval layer
- **Reciprocal Rank Fusion (RRF)** — merges BM25 and vector search ranked lists into a single result set
- **Pydantic v2 / pydantic-settings** — data models for pipeline state, chunk sources, eval scores, and settings; also serves as the structural input guardrail
- **Groq API (llama-3.3-70b-versatile)** — LLM backend for classification, routing, generation, and evaluation (replaced HuggingFace mid-project)
- **HuggingFace Transformers** — initial LLM backend using bart-large-mnli (zero-shot classification) and flan-t5-base (generation); replaced due to compatibility issues and slow inference
- **OpenAI Python SDK** — used as the HTTP client for both Groq and OpenAI endpoints (Groq is OpenAI-compatible)
- **Streamlit** — front-end UI with chat interface, reasoning steps expander, sources table, eval scores, and PDF upload sidebar
- **pypdf** — PDF text extraction during document ingestion
- **reportlab** — synthetic PDF generation for the three knowledge base collections
- **Python-dotenv / pydantic-settings** — environment variable management via `.env` file

---

## What Worked Well

**LangGraph for pipeline orchestration** — The stateful graph made it straightforward to implement conditional routing, retry logic on low eval scores, and short-circuit paths for unknown queries. The MemorySaver integration for conversation history required minimal extra code.

**Hybrid search with RRF** — Combining BM25 and vector search noticeably improved retrieval quality over either method alone. BM25 handles exact keyword matches well (e.g. specific procedure names), while semantic search handles paraphrased or conceptual queries.

**Chain-of-thought prompting** — Exposing the reasoning steps in the UI made the system more transparent and trustworthy, which is important in a safety-critical environment.

**LLM-as-a-judge evaluation** — Having a separate evaluation node that scores every response and can trigger a retry added a meaningful quality gate without requiring labeled ground truth data.

**Follow-up question handling** — The classifier was made conversation-aware by injecting the last few turns of history into the prompt, allowing it to correctly infer the domain for follow-up questions like "can you summarize section 1.3?" without the user needing to repeat context. A hard fallback also reuses the prior category if the LLM still returns unknown. This made multi-turn conversations feel natural and coherent.

**Groq as the LLM backend** — Fast inference with a strong model (llama-3.3-70b-versatile) made the end-to-end latency acceptable for an interactive assistant.

---

## What Didn't Work / Challenges

**HuggingFace models and time management** — I initially chose HuggingFace models to avoid needing API keys, wanting a fully local setup. In practice this led to significant time loss — debugging pipeline task compatibility issues (`text2text-generation` was removed in newer transformers versions, requiring a switch to `AutoModelForSeq2SeqLM` directly), slow inference, and weak classification quality from bart-large-mnli on domain-specific labels. Switching to Groq's free tier resolved all of these issues and should have been the starting point.

**Model deprecation mid-development** — After switching to Groq, the configured model (`llama3-8b-8192`) was decommissioned without warning, causing a runtime error mid-session. Required identifying the replacement model (`llama-3.3-70b-versatile`) and updating the config.

**ChromaDB import path** — `chromadb.config.Settings` import path changed across versions, requiring verification against the installed version.

**Synthetic data limitations** — The generated PDFs are realistic in structure but limited in depth. A real deployment would need actual documentation, which would significantly improve retrieval quality and answer specificity.

---

## Lessons Learned

**Spec-driven development pays off** — Writing requirements, design, and tasks before coding forced clarity on the architecture upfront. It also made it easier to track what was done and what remained.

**LLM-based classification needs context** — A classifier that only sees the current query will fail on follow-ups. Conversation history is essential context, not optional.

**Start with a managed API, not local models** — Beginning with HuggingFace models to avoid API keys seemed practical but cost more time than it saved. Debugging compatibility issues, slow inference, and weak model quality on domain-specific tasks added up quickly. Starting directly with Groq's free tier would have been faster and produced better results from day one.

**Evaluation is a first-class concern** — Building the LLM-as-a-judge evaluator into the pipeline from the start, rather than as an afterthought, made it easy to surface quality issues during testing and gave the system a self-correction mechanism.

**Model versioning is a real operational risk** — LLM providers deprecate models without much notice. Pinning model names in config and having a clear upgrade path (as done here via `GROQ_MODEL` env var) is important for maintainability.

**Unit tests should have been included** — The pipeline has several moving parts — classifier, router, retriever, generator, evaluator — each of which can fail independently. Not writing unit tests meant issues were only caught at runtime during manual testing. Adding tests for each node in isolation would have caught bugs earlier and made debugging faster.

**Guardrails should be a first-class component** — A basic structural guardrail was implemented using Pydantic BaseModel for input schema validation, which enforces type safety and rejects malformed inputs before they reach the pipeline. For prompt injection, I addressed direct injection (where a user embeds malicious instructions in their query) through system prompt engineering — explicitly instructing the model to ignore any instructions in the user input and answer only from the provided context. However, indirect prompt injection — where malicious content is embedded inside the retrieved documents themselves — is a harder problem that requires additional mitigations: sanitizing documents before ingestion into ChromaDB, adding an output validation layer that checks the response stays within the expected domain, and using a dedicated guardrail framework like NeMo Guardrails to scan outputs before they are returned to the user.

**RAG evaluation needs a golden dataset and RAGAS** — The built-in LLM-as-a-judge provides per-response scoring but is not a systematic benchmark. Using RAGAS against a curated golden dataset of question-answer pairs would give repeatable, objective metrics (faithfulness, answer relevance, context precision, context recall) and make it easy to detect regressions when the pipeline changes.

**Knowledge base architecture trade-offs** — I initially considered using a single ChromaDB collection with metadata filters per category, which would simplify ingestion and management. The interviewer pointed out that three separate collections reduce retrieval latency by keeping each collection smaller and more focused. Both approaches are valid — a single collection with metadata filtering works well at small scale and is easier to maintain, while separate collections scale better and avoid cross-category noise in retrieval.

**Handling misclassification** — When the interviewer asked what happens if the classifier routes a query to the wrong knowledge base, I proposed adding confidence scoring to the classification step. If the model's confidence for the top category falls below a threshold, the system could either ask the user to clarify, fall back to searching all three collections, or flag the response for review. This would make the pipeline more robust to ambiguous queries that sit on the boundary between categories.


