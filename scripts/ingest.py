"""Incremental ingestion: `uv run python scripts/ingest.py [--dry-run | --rebuild [--yes]]`."""
import argparse
import sys

import _bootstrap  # noqa: F401

from insurellm_rag.config import load_settings
from insurellm_rag.errors import InsurellmError
from insurellm_rag.logging_config import setup_logging
from insurellm_rag.service import KnowledgeAssistant


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="show what would change; write nothing")
    ap.add_argument("--rebuild", action="store_true",
                    help="DESTRUCTIVE: delete the vector index and rebuild from scratch")
    ap.add_argument("--yes", action="store_true", help="skip the rebuild confirmation prompt")
    args = ap.parse_args()
    try:
        settings = load_settings()
        setup_logging(settings.log_level)
        if args.rebuild and not args.dry_run and not args.yes:
            answer = input("This deletes the existing vector index and re-embeds everything. "
                           "Type 'rebuild' to continue: ")
            if answer.strip().lower() != "rebuild":
                print("Aborted.")
                return 1
        report = KnowledgeAssistant(settings).ingest(dry_run=args.dry_run, rebuild=args.rebuild)
    except InsurellmError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(report.summary())
    if report.discovered == 0:
        print("\nNo Markdown files found. Copy the original Knowledge Base into knowledge-base/ "
              "and run this command again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
