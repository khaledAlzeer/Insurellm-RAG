"""Startup diagnostics: `uv run python scripts/health_check.py [--load-models]`."""
import argparse
import sys

import _bootstrap  # noqa: F401

from insurellm_rag.config import load_settings
from insurellm_rag.errors import ConfigError
from insurellm_rag.service import KnowledgeAssistant

ICON = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--load-models", action="store_true",
                    help="also load the embedding model (downloads it on first use)")
    args = ap.parse_args()
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc)
        return 2
    checks = KnowledgeAssistant(settings).health(load_models=args.load_models)
    for c in checks:
        print(f"{ICON[c.status]} {c.name}: {c.detail}")
    return 1 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
