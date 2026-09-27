# Copyright 2026 Scott Weeden
# Author: Scott Weeden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared Ollama / local-LLM helper for VSA agents (spec + refactor)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

_PROJECT = Path(__file__).resolve().parent.parent.parent
if (_PROJECT / ".env").is_file():
    load_dotenv(_PROJECT / ".env")

_OFFLINE = False


def set_offline(offline: bool) -> None:
    global _OFFLINE
    _OFFLINE = bool(offline)


def is_offline() -> bool:
    return _OFFLINE or os.getenv("VSA_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def get_llm():
    """Return ChatOllama or None when offline / unavailable."""
    if is_offline():
        return None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = (
        os.getenv("OLLAMA_MODEL")
        or os.getenv("OLLAMA_MODEL_NAME")
        or os.getenv(".env")
        or "gemma4:12b"
    )
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "90"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
    reasoning = os.getenv("OLLAMA_REASONING", "0").strip().lower() in {"1", "true", "yes"}
    try:
        from langchain_ollama import ChatOllama

        kwargs = dict(
            model=model,
            base_url=base_url,
            temperature=0.1,
            timeout=timeout_s,
            num_predict=num_predict,
        )
        try:
            return ChatOllama(**kwargs, reasoning=reasoning)
        except TypeError:
            return ChatOllama(**kwargs)
    except Exception as exc:
        print(f"[vsa] LLM unavailable ({exc}); continuing offline", flush=True)
        return None


def llm_json(system: str, user: str, *, max_context: int = 8000) -> Optional[dict]:
    """Invoke local LLM and parse a JSON object from the response."""
    llm = get_llm()
    if not llm:
        return None
    user = user[:max_context]
    try:
        print(f"[vsa] LLM invoke (user={len(user)} chars)...", flush=True)
        response = llm.invoke([("system", system), ("user", user)])
        content = getattr(response, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                (c.get("text", "") if isinstance(c, dict) else str(c)) for c in content
            )
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            print(f"[vsa] LLM returned no JSON ({len(content)} chars)", flush=True)
            return None
        return json.loads(match.group())
    except Exception as exc:
        print(f"[vsa] LLM JSON failed: {exc}", flush=True)
        return None


def project_root() -> Path:
    return _PROJECT
