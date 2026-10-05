"""Exception types with actionable messages (no stack traces are shown to UI users)."""


class InsurellmError(Exception):
    """Base class for all expected application errors."""


class ConfigError(InsurellmError):
    """Invalid configuration."""


class KnowledgeBaseError(InsurellmError):
    """Knowledge Base directory is missing or unusable."""


class EmbeddingError(InsurellmError):
    """Embedding model could not be loaded or produced invalid vectors."""


class VectorStoreError(InsurellmError):
    """Vector database is missing, corrupted or incompatible."""


class ModelUnavailableError(InsurellmError):
    """Ollama is not reachable or the requested model is not installed."""


class GenerationError(InsurellmError):
    """Model request failed, timed out or returned unusable output."""
