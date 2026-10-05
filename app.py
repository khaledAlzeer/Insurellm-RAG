"""Entry point: `uv run python app.py`."""

import logging
import sys
from pathlib import Path

# Works with `uv sync` (editable install) and as a plain checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from insurellm_rag.config import load_settings
from insurellm_rag.errors import ConfigError
from insurellm_rag.logging_config import setup_logging
from insurellm_rag.service import KnowledgeAssistant
from insurellm_rag.ui import launch


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    setup_logging(settings.log_level)
    assistant = KnowledgeAssistant(settings)
    for check in assistant.health():
        level = logging.INFO if check.status == "ok" else logging.WARNING
        logging.getLogger("startup").log(level, "%s: %s - %s", check.name, check.status, check.detail)
    launch(assistant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
