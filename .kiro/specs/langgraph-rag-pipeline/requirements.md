# Requirements Document

## Introduction

A LangGraph-based agentic RAG pipeline that enables plant floor supervisors to ask natural language questions and receive accurate, grounded answers sourced from the correct internal documentation collection. The pipeline routes queries through a sequence of specialized nodes — validation, classification, routing, retrieval, generation, evaluation, and output formatting — to ensure responses are traceable, hallucination-resistant, and explainable.

## Glossary

- **Pipeline**: The full LangGraph graph that processes a supervisor query end-to-end.
- **Input_Validator**: The node that sanitizes and validates the raw user query before any LLM call.
- **Classifier**: The LLM-powered node that assigns a query to one of the defined categories.
- **Router**: The LLM-powered node that determines which retriever to invoke by classifying the query category via an LLM call, then returns a routing key mapped to the correct retriever node.
- **Retriever**: One of three sub-nodes (Safety_Retriever, Maintenance_Retriever, Quality_Retriever), each querying its own ChromaDB collection.
- **Generator**: The node that constructs a RAG prompt from retrieved chunks and calls the LLM to produce an answer.
- **Evaluator**: The LLM-as-a-Judge node that scores the generated answer across four quality dimensions.
- **Output_Formatter**: The node that assembles the final structured response including answer, citations, scores, and reasoning trace.
- **PipelineState**: The Pydantic v2 model that carries all data between nodes.
- **Category**: One of `safety_procedures`, `maintenance_manuals`, `quality_control_standards`, or `unknown`.
- **ChunkSource**: A retrieved document chunk with associated metadata (collection name, document name, section, chunk ID, content).
- **EvalScores**: A Pydantic v2 model holding four float scores (0.0–1.0): answer_relevance, context_relevance, groundedness, completeness.
- **Eval_Threshold**: A configurable float value below which any EvalScore dimension triggers a flagged response.
- **Short_Circuit**: A boolean flag on PipelineState that, when true, causes all subsequent nodes to skip processing and pass state directly to the Output_Formatter.
- **Reasoning_Trace**: An ordered list of strings appended by each node to record processing decisions for explainability.
- **Top_K**: A configurable integer specifying the maximum number of chunks to retrieve per query.
- **Hybrid_Search**: A retrieval strategy that combines keyword-based search (BM25) and semantic vector search (cosine similarity via ChromaDB), then merges and re-ranks the results before returning chunks.
- **BM25**: A probabilistic keyword-based ranking function used to score documents by term frequency and inverse document frequency, independent of vector embeddings.
- **Re-ranking**: The process of merging and scoring results from multiple retrieval methods (BM25 and vector search) into a single ranked list, using Reciprocal Rank Fusion (RRF) or a comparable score-fusion strategy.
- **Synthetic_Dataset**: A set of PDF documents generated programmatically to simulate real-world plant floor documentation across the three supported domains.
- **PDF_Ingestion_Script**: A standalone script that reads PDF files from a directory, splits content into chunks, extracts metadata, and loads the chunks into the correct ChromaDB collection.
- **Chunk_Metadata**: Structured metadata attached to each chunk, including document name, section heading, and chunk ID.
- **Document_Section**: A named subdivision of a PDF document, delimited by a heading, that groups related procedural or reference content.
- **CoT_Reasoning**: The chain-of-thought reasoning steps produced by the Generator before the final answer, captured separately for transparency.
- **Conversation_History**: An ordered list of prior turns in a multi-turn session, each entry containing a `role` (`user` or `assistant`) and `content` string.
- **MemorySaver**: The LangGraph built-in checkpointer that persists graph state across invocations using an in-memory store, enabling multi-turn conversation continuity.
- **Thread_ID**: A caller-supplied string that uniquely identifies a conversation session, passed in the LangGraph run config so the MemorySaver can isolate state per session.
- **Streamlit_App**: The `app.py` web application at the project root that provides a chat interface, PDF upload widget, and expandable result panels for the pipeline.

---

## Requirements

### Requirement 1: Pipeline State Management

**User Story:** As a pipeline developer, I want a single Pydantic v2 state model shared across all nodes, so that data flows consistently through the graph without type errors.

#### Acceptance Criteria

1. THE PipelineState SHALL be defined as a Pydantic v2 BaseModel containing fields for: raw_query, validated_query, category, chunks, answer, retry_count, eval_scores, eval_flagged, final_response, reasoning_trace, error, and short_circuit.
2. THE PipelineState SHALL enforce type constraints on all fields using Pydantic v2 field definitions.
3. WHEN a PipelineState is instantiated without explicit values, THE PipelineState SHALL initialize list fields to empty lists and optional fields to None.
4. THE EvalScores model SHALL expose a method that returns true when any of its four score fields falls below a caller-supplied threshold value.

---

### Requirement 2: Configuration Management

**User Story:** As a pipeline operator, I want all runtime parameters in a single configuration object loaded from environment variables, so that I can change behavior without modifying code.

#### Acceptance Criteria

1. THE Pipeline SHALL load configuration from a Pydantic v2 BaseSettings model that reads values from environment variables and a `.env` file.
2. THE Pipeline configuration SHALL include: OpenAI API key, OpenAI base URL, chat model name, judge model name, ChromaDB persist directory, three collection names (one per category), Top_K, Eval_Threshold, and max retry count.
3. WHEN an environment variable is absent, THE Pipeline configuration SHALL use a defined default value for that setting.
4. WHERE a custom OpenAI-compatible base URL is configured, THE Pipeline SHALL use that URL for all LLM calls instead of the default OpenAI endpoint.

---

### Requirement 3: Input Validation

**User Story:** As a plant floor supervisor, I want my query validated before processing, so that malformed or empty inputs are rejected with a clear message rather than causing downstream errors.

#### Acceptance Criteria

1. WHEN the raw_query field is empty or contains only whitespace, THE Input_Validator SHALL set the error field and set short_circuit to true.
2. WHEN the raw_query exceeds 2000 characters, THE Input_Validator SHALL set the error field to a message stating the limit and set short_circuit to true.
3. WHEN the raw_query passes length and content checks, THE Input_Validator SHALL strip leading and trailing whitespace, remove ASCII control characters (0x00–0x1F, 0x7F), normalize internal whitespace to single spaces, and store the result in validated_query.
4. THE Input_Validator SHALL append a description of the validation outcome to the reasoning_trace.
5. WHEN short_circuit is already true on entry, THE Input_Validator SHALL return state unchanged.

---

### Requirement 4: Query Classification

**User Story:** As a plant floor supervisor, I want my query automatically classified into the correct documentation domain, so that retrieval targets the most relevant collection.

#### Acceptance Criteria

1. WHEN the Input_Validator sets short_circuit to true, THE Classifier SHALL return state unchanged without making an LLM call.
2. WHEN the validated_query is provided, THE Classifier SHALL make a single LLM call with temperature 0 and a system prompt that defines the four valid Category values and instructs the model to respond with only the category name.
3. WHEN the LLM response matches one of the three documentation categories, THE Classifier SHALL set the category field to that value and append the classification to the reasoning_trace.
4. WHEN the LLM response does not match any of the three documentation categories, THE Classifier SHALL set category to `unknown`, set short_circuit to true, and set answer to a message stating the query is outside the supported domains.
5. THE Classifier SHALL treat any LLM response that is not an exact match to a valid Category as `unknown`.

---

### Requirement 5: LLM-Based Routing

**User Story:** As a pipeline developer, I want query routing to use an LLM call to determine the correct retriever, so that routing decisions can leverage language understanding rather than relying solely on the upstream classifier's category field.

#### Acceptance Criteria

1. THE Router SHALL be implemented as a LangGraph conditional edge function that makes a single LLM call with temperature 0 to determine the routing key.
2. WHEN short_circuit is true, THE Router SHALL return the routing key `"output"` without making an LLM call.
3. THE Router LLM call SHALL use a system prompt that defines the three valid routing targets (`safety_procedures`, `maintenance_manuals`, `quality_control_standards`) and instructs the model to respond with only the routing key name.
4. WHEN the LLM response matches one of the three valid routing targets, THE Router SHALL return that value as the routing key.
5. WHEN the LLM response does not match any valid routing target, THE Router SHALL return the routing key `"output"`.
6. THE Router SHALL append the routing decision and whether it was LLM-derived or a fallback to the reasoning_trace.

---

### Requirement 6: Document Retrieval

**User Story:** As a plant floor supervisor, I want my query matched against the correct documentation collection using hybrid search combining keyword and semantic similarity, so that retrieved chunks are relevant even when exact terms or paraphrased language are used.

#### Acceptance Criteria

1. THE Pipeline SHALL maintain three separate ChromaDB collections, one each for safety_procedures, maintenance_manuals, and quality_control_standards, each configured to use cosine similarity as the distance metric.
2. WHEN a retriever node is invoked, THE Retriever SHALL perform Hybrid_Search by executing both a BM25 keyword search and a ChromaDB vector search against its assigned collection using the validated_query, each retrieving up to Top_K candidates.
3. WHEN both BM25 and vector search results are available, THE Retriever SHALL merge and re-rank the combined candidate set using Reciprocal Rank Fusion (RRF) or a comparable score-fusion strategy, then return the top Top_K results.
4. WHEN chunks are returned after re-ranking, THE Retriever SHALL map each chunk to a ChunkSource containing document name, section, chunk ID, and content extracted from the result metadata and document fields.
5. WHEN short_circuit is true on entry, THE Retriever SHALL return state unchanged.
6. WHEN the merged result set contains no documents after Hybrid_Search, THE Retriever SHALL set answer to a message stating no relevant documentation was found, set short_circuit to true, and append the outcome to the reasoning_trace.
7. THE Retriever SHALL append the count of retrieved chunks, the collection name, and the search strategy (hybrid) to the reasoning_trace.

---

### Requirement 7: Answer Generation

**User Story:** As a plant floor supervisor, I want answers generated strictly from retrieved documentation, so that I can trust the response is grounded in actual plant records rather than model hallucination.

#### Acceptance Criteria

1. WHEN short_circuit is true on entry, THE Generator SHALL return state unchanged.
2. WHEN chunks are available in state, THE Generator SHALL construct a prompt that includes all retrieved chunk contents and instructs the LLM to answer only from the provided context.
3. THE Generator prompt SHALL explicitly instruct the LLM to state that it does not know the answer when the provided context is insufficient, rather than inferring or fabricating information.
4. WHEN the LLM returns a response, THE Generator SHALL store the response text in the answer field and append a generation note to the reasoning_trace.
5. THE Generator SHALL use the chat_model specified in configuration for the generation LLM call.

---

### Requirement 8: Response Evaluation

**User Story:** As a pipeline operator, I want generated answers automatically scored for quality, so that low-confidence responses are flagged before being returned to the supervisor.

#### Acceptance Criteria

1. WHEN short_circuit is true on entry, THE Evaluator SHALL return state unchanged.
2. WHEN an answer is present in state, THE Evaluator SHALL make a separate LLM call using the judge_model to score the answer across four dimensions: answer_relevance, context_relevance, groundedness, and completeness, each as a float between 0.0 and 1.0 inclusive.
3. THE Evaluator SHALL store the four scores in an EvalScores instance assigned to the eval_scores field.
4. WHEN any score in eval_scores falls below Eval_Threshold, THE Evaluator SHALL set eval_flagged to true.
5. WHEN eval_flagged is true and retry_count is less than max_retries, THE Evaluator SHALL increment retry_count and set short_circuit to false to allow the Generator to retry.
6. THE Evaluator SHALL append the four scores and the flagged status to the reasoning_trace.
7. THE Evaluator SHALL use a structured output format (JSON) for the judge LLM call to ensure parseable score values.

---

### Requirement 9: Output Formatting

**User Story:** As a plant floor supervisor, I want a structured response that includes the answer, source citations, quality scores, and a reasoning trace, so that I can verify where the answer came from and how confident the system is.

#### Acceptance Criteria

1. THE Output_Formatter SHALL always execute as the terminal node regardless of short_circuit state.
2. WHEN an error is set in state, THE Output_Formatter SHALL include the error message in the final_response and omit answer, sources, and scores.
3. WHEN an answer is present and no error is set, THE Output_Formatter SHALL assemble a final_response dict containing: answer text, a list of source citations derived from chunks, eval_scores (if present), eval_flagged status, and the full reasoning_trace.
4. WHEN chunks are present, THE Output_Formatter SHALL format each citation to include document name, section, and chunk ID.
5. THE Output_Formatter SHALL store the assembled structure in the final_response field of PipelineState.

---

### Requirement 10: Graph Assembly and Short-Circuit Propagation

**User Story:** As a pipeline developer, I want the LangGraph graph wired so that short-circuit state is respected by every node, so that a failure or out-of-scope query at any stage terminates processing cleanly without executing unnecessary nodes.

#### Acceptance Criteria

1. THE Pipeline SHALL define a LangGraph StateGraph using PipelineState as the state schema.
2. THE Pipeline graph SHALL include nodes for: Input_Validator, Classifier, Router (as a conditional edge), Safety_Retriever, Maintenance_Retriever, Quality_Retriever, Generator, Evaluator, and Output_Formatter.
3. THE Pipeline SHALL wire the Router as a LangGraph conditional edge from the Classifier node, mapping routing keys to the three retriever nodes and to the Output_Formatter node.
4. WHEN short_circuit is true, EACH processing node (Input_Validator, Classifier, Retriever, Generator, Evaluator) SHALL return state unchanged, delegating termination handling to the Output_Formatter.
5. THE Pipeline SHALL expose a compiled graph object that accepts a PipelineState-compatible dict as input and returns a final PipelineState.

---

### Requirement 11: Observability and Traceability

**User Story:** As a pipeline operator, I want every node to append its key decisions to the reasoning_trace, so that I can audit the full processing path for any query.

#### Acceptance Criteria

1. THE Input_Validator SHALL append the sanitized query (truncated to 80 characters) or the validation error to the reasoning_trace.
2. THE Classifier SHALL append the assigned category to the reasoning_trace.
3. THE Retriever SHALL append the collection name and chunk count (or a no-results message) to the reasoning_trace.
4. THE Generator SHALL append a note confirming generation was attempted to the reasoning_trace.
5. THE Evaluator SHALL append all four score values and the flagged status to the reasoning_trace.
6. THE Output_Formatter SHALL include the complete reasoning_trace in the final_response.

---

### Requirement 12: Synthetic Dataset and PDF Ingestion

**User Story:** As a pipeline developer, I want a realistic synthetic dataset of plant floor PDF documents and a script to ingest them into ChromaDB, so that I can test the full RAG pipeline end-to-end across all three domains with representative content.

#### Acceptance Criteria

1. THE Synthetic_Dataset SHALL include at least three PDF documents per domain (safety_procedures, maintenance_manuals, quality_control_standards), for a minimum of nine PDF files total.
2. WHEN generating safety procedure PDFs, THE Synthetic_Dataset SHALL include content representative of real plant floor safety documentation, including hazard codes (e.g., NFPA, GHS), lockout/tagout procedures, PPE requirements, and emergency response steps.
3. WHEN generating maintenance manual PDFs, THE Synthetic_Dataset SHALL include content representative of real equipment maintenance documentation, including named industrial equipment (e.g., conveyor motors, hydraulic presses, CNC machines), scheduled maintenance intervals, torque tolerances, and part numbers.
4. WHEN generating quality control standard PDFs, THE Synthetic_Dataset SHALL include content representative of real quality documentation, including inspection checklists, dimensional tolerances, defect classification codes, and sampling procedures.
5. EACH generated PDF SHALL contain structured content with a document title, numbered sections with headings, and numbered procedural steps or specification entries, so that Chunk_Metadata can be extracted during ingestion.
6. EACH domain's PDF set SHALL collectively produce a minimum of twenty chunks after ingestion, ensuring sufficient coverage for retrieval and re-ranking to operate meaningfully.
7. THE PDF_Ingestion_Script SHALL accept a source directory path and a target collection name as inputs, read all PDF files in that directory, split each document into chunks at section boundaries or at a configurable character limit, and extract Chunk_Metadata (document name, section heading, chunk ID) from each chunk.
8. WHEN loading chunks, THE PDF_Ingestion_Script SHALL embed each chunk using the same embedding model configured for the Pipeline and insert the chunk text and Chunk_Metadata into the specified ChromaDB collection.
9. WHEN a PDF file cannot be read or parsed, THE PDF_Ingestion_Script SHALL log the filename and error reason and continue processing remaining files.
10. THE PDF_Ingestion_Script SHALL print a summary upon completion stating the number of files processed, total chunks ingested, and any files skipped due to errors.

---

### Requirement 13: Chain-of-Thought Prompting

**User Story:** As a plant floor supervisor, I want the system to show its reasoning steps alongside the final answer, so that I can understand how the answer was derived from the retrieved documentation.

#### Acceptance Criteria

1. THE PipelineState SHALL include a `cot_reasoning` field of type string with a default value of empty string.
2. WHEN constructing the generation prompt, THE Generator SHALL instruct the LLM to first output its reasoning steps prefixed with `"Reasoning:"` and then output the final answer prefixed with `"Answer:"`.
3. WHEN the LLM returns a response, THE Generator SHALL parse the response to extract the text following `"Reasoning:"` into `cot_reasoning` and the text following `"Answer:"` into `answer`.
4. WHEN assembling the final_response on the success path, THE Output_Formatter SHALL include `cot_reasoning` under the key `"reasoning_steps"`.

---

### Requirement 14: Source Attribution Per Answer

**User Story:** As a plant floor supervisor, I want each source citation to identify which knowledge base collection it came from, so that I can trace answers back to the exact documentation domain.

#### Acceptance Criteria

1. THE ChunkSource model SHALL include a `collection` field of type string that holds the ChromaDB collection name the chunk was retrieved from.
2. WHEN building ChunkSource objects, THE Retriever SHALL populate the `collection` field with the name of the collection that was queried.
3. WHEN assembling citations in the final_response, THE Output_Formatter SHALL include `collection`, `document`, `section`, and `chunk_id` in each source entry.

---

### Requirement 15: Multi-Turn Conversation with MemorySaver

**User Story:** As a plant floor supervisor, I want to ask follow-up questions that reference prior answers in the same session, so that I can have a natural multi-turn conversation with the pipeline.

#### Acceptance Criteria

1. THE PipelineState SHALL include a `conversation_history` field of type list of dicts (each with `role` and `content` keys) with a default value of empty list.
2. WHEN constructing the generation prompt, THE Generator SHALL include the last N turns from `conversation_history` in the prompt, where N is a configurable value with a default of 5.
3. WHEN the Generator produces an answer, THE Generator SHALL append the user query (role `"user"`) and the assistant answer (role `"assistant"`) to `conversation_history`.
4. THE Pipeline SHALL be compiled with a LangGraph `MemorySaver` instance as the checkpointer so that state is persisted across invocations within the same session.
5. WHEN invoking the pipeline, THE Pipeline SHALL accept a `thread_id` string in the run config that the MemorySaver uses to isolate state per conversation session.
6. THE Pipeline configuration SHALL include a `conversation_history_window` integer setting with a default value of 5 that controls how many prior turns are included in the generation prompt.

---

### Requirement 16: Streamlit UI

**User Story:** As a plant floor supervisor, I want a web-based chat interface that shows conversation history, reasoning steps, source citations, and evaluation scores, and lets me upload PDFs to the knowledge base, so that I can interact with the pipeline without using the command line.

#### Acceptance Criteria

1. THE Streamlit_App SHALL display a chat interface that renders the full conversation history, showing user messages and assistant answers in order.
2. WHEN the latest answer includes `cot_reasoning`, THE Streamlit_App SHALL display an expandable "Reasoning Steps" section showing the `cot_reasoning` text.
3. WHEN the latest answer includes sources, THE Streamlit_App SHALL display an expandable "Sources" section listing each citation's `collection`, `document`, `section`, and `chunk_id`.
4. WHEN the latest answer includes eval_scores, THE Streamlit_App SHALL display an expandable "Evaluation Scores" section showing all four EvalScores dimensions.
5. THE Streamlit_App SHALL include a sidebar with a PDF upload widget that allows the user to select a target knowledge base (`safety_procedures`, `maintenance_manuals`, or `quality_control_standards`) and upload one or more PDF files.
6. WHEN PDF files are uploaded, THE Streamlit_App SHALL save the files to a temporary directory and invoke the ingestion script to load them into the selected ChromaDB collection.
7. WHEN ingestion completes, THE Streamlit_App SHALL display a success message; IF ingestion fails, THE Streamlit_App SHALL display an error message with the failure reason.
8. THE Streamlit_App SHALL maintain a `thread_id` in Streamlit session state to support multi-turn conversation continuity across chat submissions.
