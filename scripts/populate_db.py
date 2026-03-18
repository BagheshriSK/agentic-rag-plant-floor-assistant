"""Convenience script: generate synthetic PDFs then ingest all three collections.

Usage:
    python scripts/populate_db.py
"""
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COLLECTIONS = [
    ("safety_procedures", os.path.join(PROJECT_ROOT, "data", "pdfs", "safety_procedures")),
    ("maintenance_manuals", os.path.join(PROJECT_ROOT, "data", "pdfs", "maintenance_manuals")),
    ("quality_control_standards", os.path.join(PROJECT_ROOT, "data", "pdfs", "quality_control_standards")),
]


def run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    # Step 1: generate PDFs
    run([sys.executable, os.path.join(SCRIPT_DIR, "generate_pdfs.py")])

    # Step 2: ingest each collection
    for collection_name, source_dir in COLLECTIONS:
        run([
            sys.executable,
            os.path.join(SCRIPT_DIR, "ingest_pdfs.py"),
            "--source-dir", source_dir,
            "--collection", collection_name,
        ])

    print("\nAll collections populated successfully.")


if __name__ == "__main__":
    main()
