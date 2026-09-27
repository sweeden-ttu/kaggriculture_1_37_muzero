"""Type stubs for the script-only `docs/vsa/local_llm.py` helper module.

`local_llm.py` lives in `docs/vsa/` and is loaded at runtime by the VS code
agents via a `sys.path.insert` of their own directory; it is not an installed
package and its directory is not an import root. This stub lets pyright /
pyrefly resolve `from local_llm import ...` from the repository's configured
`typings` stubPath while runtime behavior is unchanged.
"""

from pathlib import Path
from typing import Any, Optional


def set_offline(offline: bool) -> None: ...


def is_offline() -> bool: ...


def get_llm() -> Any:
    """Return a langchain_ollama ChatOllama instance, or None when offline/unavailable."""
    ...


def llm_json(system: str, user: str, *, max_context: int = 8000) -> Optional[dict]:
    """Invoke the local LLM and parse a JSON object from the response (or None when offline/failed)."""
    ...


def project_root() -> Path: ...