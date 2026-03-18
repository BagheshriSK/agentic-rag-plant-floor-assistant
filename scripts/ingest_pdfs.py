"""Ingest PDFs into a ChromaDB collection.

Usage:
    python scripts/ingest_pdfs.py --source-dir data/pdfs/safety_procedures --collection safety_procedures
"""
import argparse
import os
import re
import sys

import chromadb
from chromadb.config import Settings as ChromaSettings

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_pipeline.config import settings as app_settings  # noqa: E402

SECTION_PATTERN = re.compile(r"^\d+\.\s+[A-Z]")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def split_into_chunks(text: str, doc_name: str) -> list[dict]:
    """Split text at section boundaries; fall back to character-based chunking."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "General"
    current_lines: list[str] = []

    for line in lines:
        if SECTION_PATTERN.match(line.strip()):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))

    chunks: list[dict] = []
    chunk_index = 0

    for heading, sec_lines in sections:
        section_text = "\n".join(sec_lines).strip()
        if not section_text:
            continue

        # If section fits in one chunk, keep it whole
        if len(section_text) <= CHUNK_SIZE:
            chunks.append({
                "content": section_text,
                "document": doc_name,
                "section": heading,
                "chunk_id": f"{doc_name}_chunk_{chunk_index:03d}",
            })
            chunk_index += 1
        else:
            # Character-based chunking with overlap
            start = 0
            while start < len(section_text):
                end = start + CHUNK_SIZE
                chunk_text = section_text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "content": chunk_text,
                        "document": doc_name,
                        "section": heading,
                        "chunk_id": f"{doc_name}_chunk_{chunk_index:03d}",
                    })
                    chunk_index += 1
                start = end - CHUNK_OVERLAP
                if start >= len(section_text):
                    break

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument("--source-dir", required=True, help="Directory containing PDF files.")
    parser.add_argument("--collection", required=True, help="ChromaDB collection name.")
    args = parser.parse_args()

    source_dir = args.source_dir
    collection_name = args.collection

    if not os.path.isdir(source_dir):
        print(f"Error: source directory '{source_dir}' does not exist.")
        sys.exit(1)

    client = chromadb.PersistentClient(
        path=app_settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    pdf_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
    files_processed = 0
    files_skipped = 0
    total_chunks = 0

    for filename in sorted(pdf_files):
        filepath = os.path.join(source_dir, filename)
        doc_name = os.path.splitext(filename)[0]
        try:
            from pypdf import PdfReader  # noqa: PLC0415
            reader = PdfReader(filepath)
            full_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as exc:
            print(f"Warning: skipping '{filename}' — {exc}")
            files_skipped += 1
            continue

        chunks = split_into_chunks(full_text, doc_name)
        if not chunks:
            print(f"Warning: no chunks extracted from '{filename}', skipping.")
            files_skipped += 1
            continue

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {"document": c["document"], "section": c["section"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ]

        # Upsert to avoid duplicate key errors on re-ingestion
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        total_chunks += len(chunks)
        files_processed += 1

    print(f"Processed {files_processed} files, ingested {total_chunks} chunks, skipped {files_skipped} files.")


if __name__ == "__main__":
    main()
