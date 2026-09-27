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

"""
Kaggriculture MuZero Repository Refactor & Housekeeping Agent
=============================================================

LangGraph + local Ollama agent that inventories the repo, finds structural gaps,
proposes (and optionally applies) scaffolding / IDE / conda / docs housekeeping.

Default mode is **plan-only** (dry-run). Pass ``--apply`` for safe writes
(settings, env.yml, .cursor rules, mkdocs build). Destructive moves/deletes
require ``--apply --force``.

Tasks:
  inventory, organize, gaps, dedupe, naming, lsp, cursor, conda,
  align_docs, housekeeping, publish_docs, all
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

# Allow running as a script without installing the package
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from local_llm import llm_json, project_root  # noqa: E402  # pyright: ignore[reportMissingModuleSource]

ROOT = project_root()

# ---------------------------------------------------------------------------
# Known root-level re-export aliases (docs-facing entrypoints → muzero/)
# ---------------------------------------------------------------------------
ROOT_ALIASES: Dict[str, str] = {
    "kaggriculture_muzero_chassis.py": "muzero/chassis.py",
    "kaggriculture_observation_encoder.py": "muzero/observation_encoder.py",
    "sampled_muzero_mcts.py": "muzero/mcts.py",
    "simsiam_consistency.py": "muzero/chassis.py",
    "prioritized_muzero_buffer.py": "muzero/buffer.py",
    "muzero_replay_buffer.py": "muzero/buffer.py",
    "muzero_trainer_loop.py": "muzero/trainer.py",
    "league_tracker.py": "evaluation/league.py",
}

CANONICAL_PACKAGES = [
    "muzero",
    "pipelines",
    "hybrid_chassis",
    "evaluation",
    "packaging",
    "scripts",
    "tests",
    "docs",
    "lsp",
    "typings",
]

_OFFLINE = False


def set_offline(offline: bool) -> None:
    """Force static analysis only (no Ollama / LLM calls)."""
    global _OFFLINE
    _OFFLINE = bool(offline)


def get_llm():
    """Get LLM for analysis (Ollama default). Returns None when offline or unavailable."""
    if _OFFLINE or os.getenv("VSA_OFFLINE", "").strip() in {"1", "true", "yes"}:
        return None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    # Prefer OLLAMA_MODEL; tolerate legacy/mistyped ".env" key if someone set it.
    model = (
        os.getenv("OLLAMA_MODEL")
        or os.getenv("OLLAMA_MODEL_NAME")
        or os.getenv(".env")
        or "gemma4:12b"
    )
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "90"))
    num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
    # gemma4 "thinking" burns the token budget and often yields empty content
    reasoning_env = os.getenv("OLLAMA_REASONING", "0").strip().lower()
    reasoning = reasoning_env in {"1", "true", "yes"}
    try:
        from langchain_ollama import ChatOllama

        kwargs = dict(
            model=model,
            base_url=base_url,
            temperature=0.1,
            timeout=timeout_s,
            num_predict=num_predict,
        )
        # langchain-ollama >=1.0 may accept reasoning=; ignore if unsupported
        try:
            return ChatOllama(**kwargs, reasoning=reasoning)
        except TypeError:
            return ChatOllama(**kwargs)
    except Exception as exc:
        print(f"[vsa] LLM unavailable ({exc}); continuing with static analysis", flush=True)
        return None


def extract_code_content(file_path: Path) -> str:
    """Read and return file content."""
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR reading {file_path}: {e}"


def parse_component_analysis(component: str, source_files: List[str], doc_files: List[str]) -> Dict[str, Any]:
    """Parse source and doc files to extract structured analysis."""
    project_root = Path(__file__).resolve().parent.parent.parent

    analysis: Dict[str, Any] = {
        "component": component,
        "description": COMPONENT_REGISTRY.get(component, {}).get("description", ""),
        "source_files": [],
        "doc_files": [],
        "key_classes": [],
        "key_functions": [],
        "hyperparameters": {},
        "architecture_notes": [],
        "training_workflow": [],
        "testing_approach": [],
        "evaluation_metrics": [],
        "missing_sources": [],
        "missing_docs": [],
    }

    # Process source files
    for rel_path in source_files:
        full_path = project_root / rel_path
        if full_path.exists():
            content = extract_code_content(full_path)
            classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
            functions = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
            analysis["source_files"].append({
                "path": rel_path,
                "size": len(content),
                "classes": classes,
                "functions": functions,
                "imports": re.findall(r"^(?:from|import)\s+([\w.]+)", content, re.MULTILINE),
            })
            analysis["key_classes"].extend(classes)
            analysis["key_functions"].extend(functions)
        else:
            analysis["missing_sources"].append(rel_path)

    # Process doc files
    for rel_path in doc_files:
        full_path = project_root / rel_path
        if full_path.exists():
            content = extract_code_content(full_path)
            analysis["doc_files"].append({
                "path": rel_path,
                "size": len(content),
                "sections": re.findall(r"^#+\s+(.+)", content, re.MULTILINE),
            })
        else:
            analysis["missing_docs"].append(rel_path)

    # De-dupe while preserving order
    analysis["key_classes"] = list(dict.fromkeys(analysis["key_classes"]))
    analysis["key_functions"] = list(dict.fromkeys(analysis["key_functions"]))

    if analysis["missing_sources"]:
        analysis["architecture_notes"].append(
            f"Missing source files: {', '.join(analysis['missing_sources'])}"
        )
    if analysis["missing_docs"]:
        analysis["architecture_notes"].append(
            f"Missing doc files: {', '.join(analysis['missing_docs'])}"
        )

    return analysis

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "inventory": {
        "description": "Scan topology: packages, root clutter, duplicate hashes, site/docs drift",
    },
    "organize": {
        "description": "Propose folder layout for root aliases and stray modules",
    },
    "gaps": {
        "description": "Find missing scaffolding (__init__, tests, env.yml, IDE configs)",
    },
    "dedupe": {
        "description": "Detect duplicate / near-duplicate modules and thin re-export aliases",
    },
    "naming": {
        "description": "Suggest simpler library / class names aligned with muzero.* APIs",
    },
    "lsp": {
        "description": "Tune pyright + VS Code / Visual Studio Python language service",
    },
    "cursor": {
        "description": "Ensure .cursor rules/settings for agent workflows",
    },
    "conda": {
        "description": "Create / refresh conda environment.yml for miniforge env `kagg`",
    },
    "align_docs": {
        "description": "Cross-check README.md vs AGENTS.md vs ADR 0001 (spatial production)",
    },
    "housekeeping": {
        "description": "gitignore, __pycache__, .DS_Store, stale site copies of agents",
    },
    "publish_docs": {
        "description": "Build MkDocs site (and optionally gh-deploy)",
    },
}


class RefactorState(TypedDict):
    tasks: list
    apply: bool
    force: bool
    inventory: dict
    findings: list
    actions: list
    llm_plan: dict
    report: str
    history: list
    current_step: str


# ---------------------------------------------------------------------------
# Static scanners
# ---------------------------------------------------------------------------

def _file_sha1(path: Path, limit: int = 256_000) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:12]


def scan_inventory() -> Dict[str, Any]:
    """Build a structured inventory of the repository."""
    root_py = sorted(p.name for p in ROOT.glob("*.py") if p.is_file())
    packages = [d.name for d in ROOT.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
    missing_pkgs = [p for p in CANONICAL_PACKAGES if not (ROOT / p).is_dir()]

    alias_status = []
    for alias, target in ROOT_ALIASES.items():
        ap = ROOT / alias
        tp = ROOT / target
        alias_status.append({
            "alias": alias,
            "target": target,
            "alias_exists": ap.is_file(),
            "target_exists": tp.is_file(),
            "thin_reexport": bool(
                ap.is_file()
                and any(
                    kw in ap.read_text(encoding="utf-8", errors="ignore")[:800]
                    for kw in ("from muzero", "from evaluation", "from pipelines", "from hybrid_chassis", "from packaging")
                )
            ),
        })

    # Exact-content duplicates among small root/scripts files
    by_hash: Dict[str, List[str]] = {}
    for pattern in ("*.py", "docs/**/*.md"):
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            if any(part in {".git", "site", "artifacts", "replays", "dist", "__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            if path.stat().st_size > 200_000:
                continue
            digest = _file_sha1(path)
            by_hash.setdefault(digest, []).append(str(path.relative_to(ROOT)))
    duplicates = {k: v for k, v in by_hash.items() if len(v) > 1}

    ide = {
        "vscode_settings": (ROOT / ".vscode" / "settings.json").is_file(),
        "pyrightconfig": (ROOT / "pyrightconfig.json").is_file(),
        "cursor_dir": (ROOT / ".cursor").is_dir(),
        "cursor_rules": (ROOT / ".cursor" / "rules").is_dir() or (ROOT / ".cursorrules").is_file(),
        "environment_yml": (ROOT / "environment.yml").is_file(),
        "mkdocs_yml": (ROOT / "mkdocs.yml").is_file(),
        "gitignore": (ROOT / ".gitignore").is_file(),
        "agents_md": (ROOT / "AGENTS.md").is_file(),
        "readme": (ROOT / "README.md").is_file(),
    }

    # Docs vs site footprint
    docs_vsa = list((ROOT / "docs" / "vsa").glob("*.py")) if (ROOT / "docs" / "vsa").is_dir() else []
    site_vsa = list((ROOT / "site" / "vsa").glob("*.py")) if (ROOT / "site" / "vsa").is_dir() else []

    return {
        "root_py": root_py,
        "packages_with_init": packages,
        "missing_canonical_dirs": missing_pkgs,
        "aliases": alias_status,
        "content_duplicates": duplicates,
        "ide": ide,
        "docs_vsa_agents": [p.name for p in docs_vsa],
        "site_vsa_py_copies": [p.name for p in site_vsa],
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


def find_gaps(inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    ide = inventory.get("ide", {})

    if not ide.get("environment_yml"):
        findings.append({
            "severity": "high",
            "gap": "missing_environment_yml",
            "detail": "No environment.yml for conda/miniforge `kagg` — README assumes it exists.",
            "fix": "write_environment_yml",
        })
    if not ide.get("cursor_rules"):
        findings.append({
            "severity": "medium",
            "gap": "missing_cursor_rules",
            "detail": ".cursor/ exists but no rules for agent conventions.",
            "fix": "write_cursor_rules",
        })
    if not ide.get("vscode_settings"):
        findings.append({
            "severity": "high",
            "gap": "missing_vscode_settings",
            "detail": ".vscode/settings.json missing for Python LSP.",
            "fix": "write_vscode_settings",
        })

    # pyright include list may omit spatial modules
    pr = ROOT / "pyrightconfig.json"
    if pr.is_file():
        try:
            cfg = json.loads(pr.read_text(encoding="utf-8"))
            include = set(cfg.get("include") or [])
            for needed in ("docs/vsa", "muzero/chassis.py"):
                # chassis is under muzero/ already if "muzero" in include
                pass
            if "docs" not in include and "docs/vsa" not in include:
                findings.append({
                    "severity": "low",
                    "gap": "pyright_excludes_docs_vsa",
                    "detail": "pyrightconfig include does not cover docs/vsa agents.",
                    "fix": "update_pyrightconfig",
                })
        except Exception as exc:
            findings.append({
                "severity": "medium",
                "gap": "pyrightconfig_unreadable",
                "detail": str(exc),
                "fix": "update_pyrightconfig",
            })

    for alias in inventory.get("aliases", []):
        if alias["alias_exists"] and not alias["target_exists"]:
            findings.append({
                "severity": "high",
                "gap": "broken_alias",
                "detail": f"{alias['alias']} → missing {alias['target']}",
                "fix": "fix_alias_target",
            })

    if inventory.get("site_vsa_py_copies"):
        findings.append({
            "severity": "medium",
            "gap": "site_contains_python_sources",
            "detail": (
                f"site/vsa/ has {len(inventory['site_vsa_py_copies'])} .py files "
                "(MkDocs should publish HTML, not agent sources)."
            ),
            "fix": "clean_site_vsa_py",
        })

    # Exact duplicate markdown (e.g. Representation_Function == Consistency_Loss)
    for digest, paths in (inventory.get("content_duplicates") or {}).items():
        md_paths = [p for p in paths if p.endswith((".md", ".MD"))]
        if len(md_paths) >= 2:
            findings.append({
                "severity": "medium",
                "gap": "duplicate_docs",
                "detail": f"Identical content: {md_paths}",
                "fix": "dedupe_docs",
                "paths": md_paths,
            })

    # Package __init__ gaps
    for pkg in ("muzero", "pipelines", "evaluation", "packaging", "hybrid_chassis", "tests"):
        d = ROOT / pkg
        if d.is_dir() and not (d / "__init__.py").is_file():
            findings.append({
                "severity": "high",
                "gap": "missing_init",
                "detail": f"{pkg}/ lacks __init__.py",
                "fix": "write_init",
                "path": str(d / "__init__.py"),
            })

    return findings


def propose_organize(inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for alias in inventory.get("aliases", []):
        if alias["alias_exists"] and alias["thin_reexport"]:
            actions.append({
                "kind": "keep_thin_alias",
                "path": alias["alias"],
                "note": f"Keep as docs-facing re-export of {alias['target']}",
            })
        elif alias["alias_exists"] and not alias["thin_reexport"]:
            actions.append({
                "kind": "convert_to_reexport",
                "path": alias["alias"],
                "target": alias["target"],
                "note": "Root module should re-export from muzero/ to avoid duplication",
            })

    clutter = [
        name for name in inventory.get("root_py", [])
        if name not in ROOT_ALIASES
        and name not in {
            "main.py",
            "train_muzero.py",
            "package_submission.py",
            "tournament.py",
            "learn_from_replay.py",
            "self_improving_loop.py",
            "boost_loop.py",
        }
    ]
    for name in clutter:
        actions.append({
            "kind": "relocate_candidate",
            "path": name,
            "suggest": "scripts/" if name.startswith(("download", "prune", "gate", "embed", "build", "render")) else "pipelines/",
            "note": "Root .py not in alias/launcher allowlist — consider moving",
        })
    return actions


def propose_naming(inventory: Dict[str, Any]) -> List[Dict[str, Any]]:
    suggestions = [
        {
            "current": "kaggriculture_muzero_chassis.py",
            "prefer": "import muzero → KaggricultureMuZeroChassis",
            "rationale": "Canonical package import; root file is a compatibility shim",
        },
        {
            "current": "muzero/_core.py (legacy macro)",
            "prefer": "muzero.legacy_macro",
            "rationale": "ADR 0001: spatial chassis is production; macro is ablation-only",
        },
        {
            "current": "docs/Representation_Function.md",
            "prefer": "docs/Consistency_Loss.md (or delete duplicate)",
            "rationale": "Byte-identical SimSiam content under a misleading name",
        },
    ]
    return suggestions


def align_readme_agents() -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="ignore") if (ROOT / "README.md").is_file() else ""
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8", errors="ignore") if (ROOT / "AGENTS.md").is_file() else ""

    spatial_markers = ("Sampled MuZero", "28-channel", "KaggricultureMuZeroChassis", "spatial")
    readme_spatial = sum(1 for m in spatial_markers if m in readme)
    agents_macro = "macro-option" in agents.lower() or "MacroOption" in agents
    agents_spatial = "spatial" in agents.lower() or "Sampled MuZero" in agents

    if readme_spatial >= 2 and agents_macro and not agents_spatial:
        findings.append({
            "severity": "high",
            "gap": "agents_md_stale_vs_readme",
            "detail": "README describes spatial Sampled MuZero; AGENTS.md still centers macro-option MLP.",
            "fix": "refresh_agents_md_spatial",
        })

    if "environment.yml" in readme and not (ROOT / "environment.yml").is_file():
        findings.append({
            "severity": "medium",
            "gap": "readme_mentions_missing_env",
            "detail": "README/setup mentions conda but environment.yml is absent.",
            "fix": "write_environment_yml",
        })

    adr = ROOT / "docs" / "adr" / "0001-spatial-sampled-muzero-deferred.md"
    if adr.is_file() and "Production" in adr.read_text(encoding="utf-8", errors="ignore"):
        if "ADR 0001" not in agents and "spatial Sampled" not in agents:
            findings.append({
                "severity": "medium",
                "gap": "agents_missing_adr0001",
                "detail": "AGENTS.md should reference ADR 0001 (spatial production).",
                "fix": "refresh_agents_md_spatial",
            })
    return findings


# ---------------------------------------------------------------------------
# Apply helpers (safe writes)
# ---------------------------------------------------------------------------

ENVIRONMENT_YML = """# Kaggriculture MuZero — Miniforge / conda env
# Create:  conda env create -f environment.yml
# Update:  conda env update -f environment.yml --prune
# Activate: conda activate kagg
name: kagg
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - numpy>=1.24
  - pandas>=2.0
  - pip:
      - torch>=2.0
      - kaggle>=1.6
      - kaggle-environments>=1.14
      - pytest>=7.0
      - pyright>=1.1.360
      - mkdocs-material>=9.0
      - mkdocs-git-revision-date-localized-plugin>=1.2
      - mkdocs-minify-plugin>=0.8
      - langgraph>=0.2
      - langchain-ollama>=0.2
      - langchain-core>=0.3
      - python-dotenv>=1.0
"""

CURSOR_RULE = """---
description: Kaggriculture MuZero agent conventions (spatial production)
globs:
alwaysApply: true
---

# Kaggriculture MuZero — Agent Rules

- Production architecture is **spatial Sampled MuZero** (ADR 0001): `muzero.chassis.KaggricultureMuZeroChassis`.
- Legacy MLP / 8-macro path lives in `muzero.legacy_macro` (`--arch macro` only).
- Prefer `muzero.*` imports over root alias shims (`kaggriculture_muzero_chassis.py`, etc.).
- Use **miniforge conda env `kagg`**, never pip venv.
- Do not delete canonical `replays/[0-9]+.json`; prune only `selfplay_*.json`.
- Docs live under `docs/`; publish with `mkdocs build` / `make` docs targets.
- Refactor agent: `docs/vsa/kaggriculture_muzero_refactor_agent.py` (plan-only by default).
- Spec agent: `docs/vsa/kaggriculture_muzero_spec_agent.py`.
"""

VSCODE_SETTINGS = {
    "python.defaultInterpreterPath": "/Users/sweeden/miniforge3/envs/kagg/bin/python",
    "python.terminal.activateEnvironment": True,
    "python.condaPath": "/Users/sweeden/miniforge3/bin/conda",
    "pyright.serverPath": "${workspaceFolder}/lsp/pyright_lsp.py",
    "python.languageServer": "Pylance",
    "python.analysis.stubPath": "${workspaceFolder}/typings",
    "python.analysis.extraPaths": [
        "${workspaceFolder}",
        "${workspaceFolder}/typings",
        "${workspaceFolder}/muzero",
        "${workspaceFolder}/pipelines",
    ],
    "python.analysis.autoSearchPaths": True,
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.diagnosticMode": "workspace",
    "[python]": {
        "editor.formatOnSave": False,
        "editor.defaultFormatter": "ms-python.python",
    },
    "files.exclude": {
        "**/__pycache__": True,
        "**/.pytest_cache": True,
        "**/replays/selfplay_*.json": True,
    },
    "files.watcherExclude": {
        "**/artifacts/**": True,
        "**/replays/**": True,
        "**/site/**": True,
        "**/_eval_*/**": True,
    },
    "search.exclude": {
        "**/site/**": True,
        "**/artifacts/**": True,
        "**/replays/**": True,
        "**/dist/**": True,
    },
}


def _write_text(path: Path, content: str, apply: bool) -> Dict[str, Any]:
    rel = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    if not apply:
        return {"status": "planned", "path": rel, "bytes": len(content)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"status": "written", "path": rel, "bytes": len(content)}


def apply_safe_actions(
    findings: List[Dict[str, Any]],
    *,
    apply: bool,
    force: bool,
    tasks: Optional[set] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    fixes = {f.get("fix") for f in findings if f.get("fix")}
    tasks = tasks or {"all"}
    want_all = "all" in tasks

    if want_all or tasks & {"conda", "gaps"}:
        if "write_environment_yml" in fixes or not (ROOT / "environment.yml").is_file():
            results.append({
                "action": "write_environment_yml",
                **_write_text(ROOT / "environment.yml", ENVIRONMENT_YML, apply),
            })

    if want_all or tasks & {"cursor", "gaps"}:
        if "write_cursor_rules" in fixes or not (ROOT / ".cursor" / "rules").exists():
            rule_path = ROOT / ".cursor" / "rules" / "kaggriculture-muzero.mdc"
            results.append({
                "action": "write_cursor_rules",
                **_write_text(rule_path, CURSOR_RULE, apply),
            })

    if want_all or tasks & {"lsp", "gaps"}:
        vs_path = ROOT / ".vscode" / "settings.json"
        payload = json.dumps(VSCODE_SETTINGS, indent=2) + "\n"
        results.append({
            "action": "write_vscode_settings",
            **_write_text(vs_path, payload, apply),
        })

    if (want_all or tasks & {"lsp", "gaps"}) and "update_pyrightconfig" in fixes:
        if (ROOT / "pyrightconfig.json").is_file():
            cfg = json.loads((ROOT / "pyrightconfig.json").read_text(encoding="utf-8"))
            include = list(cfg.get("include") or [])
            if "docs/vsa" not in include:
                include.append("docs/vsa")
            cfg["include"] = include
            results.append({
                "action": "update_pyrightconfig",
                **_write_text(
                    ROOT / "pyrightconfig.json",
                    json.dumps(cfg, indent=2) + "\n",
                    apply,
                ),
            })

    if want_all or tasks & {"gaps"}:
        for f in findings:
            if f.get("fix") == "write_init" and f.get("path"):
                path = Path(f["path"])
                if not path.is_absolute():
                    path = ROOT / path
                results.append({
                    "action": "write_init",
                    **_write_text(path, '"""Package marker."""\n', apply),
                })

    if want_all or tasks & {"housekeeping"}:
        if "clean_site_vsa_py" in fixes and force and apply:
            site_vsa = ROOT / "site" / "vsa"
            removed = []
            if site_vsa.is_dir():
                for p in site_vsa.glob("*.py"):
                    p.unlink()
                    removed.append(str(p.relative_to(ROOT)))
            results.append({"action": "clean_site_vsa_py", "status": "removed", "paths": removed})
        elif "clean_site_vsa_py" in fixes:
            results.append({
                "action": "clean_site_vsa_py",
                "status": "planned",
                "note": "Pass --apply --force to delete site/vsa/*.py source copies",
            })

        gi = ROOT / ".gitignore"
        if gi.is_file():
            text = gi.read_text(encoding="utf-8")
            needed = ["site/", "docs/vsa/out/", ".cursor/*.log", "*.egg-info/"]
            missing = [n for n in needed if n not in text]
            if missing:
                new_text = (
                    text.rstrip()
                    + "\n\n# Refactor agent / docs publish\n"
                    + "\n".join(missing)
                    + "\n"
                )
                results.append({
                    "action": "update_gitignore",
                    **_write_text(gi, new_text, apply),
                    "added": missing,
                })

    return results


def publish_docs(*, apply: bool, deploy: bool = False) -> Dict[str, Any]:
    if not (ROOT / "mkdocs.yml").is_file():
        return {"status": "error", "detail": "mkdocs.yml missing"}
    if not apply:
        return {
            "status": "planned",
            "commands": ["mkdocs build", "mkdocs gh-deploy --force"] if deploy else ["mkdocs build"],
        }
    cmd = ["mkdocs", "gh-deploy", "--force"] if deploy else ["mkdocs", "build"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-1500:],
            "stderr_tail": (proc.stderr or "")[-1500:],
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "detail": "mkdocs not installed — pip install mkdocs-material (or conda env update)",
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "detail": "mkdocs timed out"}


def refresh_agents_md_stub(*, apply: bool) -> Dict[str, Any]:
    """Append a spatial production banner to AGENTS.md if missing (non-destructive)."""
    path = ROOT / "AGENTS.md"
    if not path.is_file():
        return {"status": "skip", "detail": "AGENTS.md missing"}
    text = path.read_text(encoding="utf-8")
    banner = (
        "\n\n---\n\n"
        "## Production Architecture Note (ADR 0001)\n\n"
        "**Spatial Sampled MuZero is production.** Prefer "
        "`muzero.chassis.KaggricultureMuZeroChassis`, "
        "`muzero.mcts.SampledMuZeroMCTS`, and "
        "`pipelines/train_muzero.py --arch spatial`. "
        "Legacy MLP / 8-`MacroOption` code is under `muzero.legacy_macro` "
        "(`--arch macro` for ablation only).\n"
    )
    if "ADR 0001" in text and "Spatial Sampled MuZero is production" in text:
        return {"status": "skip", "detail": "ADR 0001 banner already present"}
    return {
        "action": "refresh_agents_md_spatial",
        **_write_text(path, text.rstrip() + banner, apply),
    }


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def node_inventory(state: RefactorState) -> Dict[str, Any]:
    print("[refactor] inventory...", flush=True)
    inv = scan_inventory()
    history = list(state.get("history") or [])
    history.append({"step": "inventory", "root_py": len(inv.get("root_py", []))})
    return {"inventory": inv, "history": history, "current_step": "analyze"}


def node_analyze(state: RefactorState) -> Dict[str, Any]:
    print("[refactor] analyze gaps / organize / naming / docs...", flush=True)
    inv = state["inventory"]
    tasks = set(state.get("tasks") or ["all"])
    findings: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []

    if tasks & {"all", "gaps", "inventory"}:
        findings.extend(find_gaps(inv))
    if tasks & {"all", "organize"}:
        actions.extend(propose_organize(inv))
    if tasks & {"all", "naming"}:
        actions.extend([{**s, "kind": "naming"} for s in propose_naming(inv)])
    if tasks & {"all", "align_docs", "dedupe"}:
        findings.extend(align_readme_agents())
    if tasks & {"all", "dedupe"}:
        for a in inv.get("aliases", []):
            if a.get("thin_reexport"):
                findings.append({
                    "severity": "info",
                    "gap": "thin_alias_ok",
                    "detail": f"{a['alias']} correctly re-exports {a['target']}",
                    "fix": None,
                })

    history = list(state.get("history") or [])
    history.append({"step": "analyze", "findings": len(findings), "actions": len(actions)})
    return {
        "findings": findings,
        "actions": actions,
        "history": history,
        "current_step": "llm_plan",
    }


def node_llm_plan(state: RefactorState) -> Dict[str, Any]:
    print("[refactor] LLM prioritization...", flush=True)
    summary = {
        "findings": state.get("findings", [])[:40],
        "actions": state.get("actions", [])[:40],
        "ide": (state.get("inventory") or {}).get("ide"),
        "root_py": (state.get("inventory") or {}).get("root_py"),
    }
    plan = llm_json(
        system=(
            "You are a repository refactor architect for Kaggriculture MuZero. "
            "Reply ONLY with JSON: "
            '{"priority_fixes":[str],"risks":[str],"suggested_order":[str],"summary":str}. '
            "Prefer safe scaffolding over mass deletion. Keep notes short."
        ),
        user=json.dumps(summary, indent=2)[:7000],
    ) or {
        "priority_fixes": [
            f["gap"] for f in state.get("findings", [])
            if f.get("severity") in {"high", "medium"} and f.get("fix")
        ][:10],
        "risks": [
            "Do not delete root aliases without updating docs imports",
            "site/ is generated — clean .py copies only with --force",
        ],
        "suggested_order": [
            "conda", "lsp", "cursor", "housekeeping", "align_docs", "publish_docs"
        ],
        "summary": "Static plan (LLM offline or failed).",
    }
    history = list(state.get("history") or [])
    history.append({"step": "llm_plan", "keys": list(plan.keys())})
    return {"llm_plan": plan, "history": history, "current_step": "apply"}


def node_apply(state: RefactorState) -> Dict[str, Any]:
    apply = bool(state.get("apply"))
    force = bool(state.get("force"))
    tasks = set(state.get("tasks") or ["all"])
    print(f"[refactor] apply={'yes' if apply else 'dry-run'} force={force}...", flush=True)

    applied: List[Dict[str, Any]] = []
    findings = list(state.get("findings") or [])

    if tasks & {"all", "lsp", "cursor", "conda", "gaps", "housekeeping", "align_docs"}:
        applied.extend(
            apply_safe_actions(findings, apply=apply, force=force, tasks=tasks)
        )

    if tasks & {"all", "align_docs"}:
        if any(f.get("fix") == "refresh_agents_md_spatial" for f in findings):
            applied.append(refresh_agents_md_stub(apply=apply))

    if tasks & {"all", "publish_docs"}:
        applied.append({"action": "publish_docs", **publish_docs(apply=apply, deploy=force)})

    history = list(state.get("history") or [])
    history.append({"step": "apply", "n": len(applied)})
    return {
        "actions": list(state.get("actions") or []) + applied,
        "history": history,
        "current_step": "report",
    }


def node_report(state: RefactorState) -> Dict[str, Any]:
    print("[refactor] report...", flush=True)
    inv = state.get("inventory") or {}
    plan = state.get("llm_plan") or {}
    lines = [
        "# Repository Refactor Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        f"**Mode**: {'APPLY' if state.get('apply') else 'DRY-RUN (plan only)'}",
        f"**Tasks**: {', '.join(state.get('tasks') or [])}",
        "",
        "## Summary",
        plan.get("summary", "(no LLM summary)"),
        "",
        "## Priority Fixes",
    ]
    for item in plan.get("priority_fixes") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Suggested Order"])
    for item in plan.get("suggested_order") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Risks"])
    for item in plan.get("risks") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Inventory", ""])
    lines.append(f"- Root `.py` files: `{', '.join(inv.get('root_py') or [])}`")
    lines.append(f"- Packages with `__init__.py`: `{', '.join(inv.get('packages_with_init') or [])}`")
    ide = inv.get("ide") or {}
    lines.append(
        "- IDE: "
        + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in ide.items())
    )

    lines.extend(["", "## Findings"])
    for f in state.get("findings") or []:
        lines.append(
            f"- **{f.get('severity', '?')}** `{f.get('gap')}` — {f.get('detail')} "
            f"(fix: `{f.get('fix')}`)"
        )

    lines.extend(["", "## Actions / Apply Results"])
    for a in state.get("actions") or []:
        if "status" in a or "action" in a:
            lines.append(
                f"- `{a.get('action') or a.get('kind')}` → "
                f"**{a.get('status', 'note')}** "
                f"{a.get('path') or a.get('note') or a.get('detail') or ''}"
            )
        else:
            lines.append(f"- `{a.get('kind')}` {a.get('path', '')}: {a.get('note', a)}")

    lines.extend([
        "",
        "## Next Commands",
        "```bash",
        "# Plan only",
        "python docs/vsa/kaggriculture_muzero_refactor_agent.py all --offline",
        "",
        "# Apply safe scaffolding (env, .vscode, .cursor, gitignore, AGENTS banner)",
        "python docs/vsa/kaggriculture_muzero_refactor_agent.py all --apply",
        "",
        "# Build docs site",
        "python docs/vsa/kaggriculture_muzero_refactor_agent.py publish_docs --apply",
        "",
        "# Delete stray site/vsa/*.py copies",
        "python docs/vsa/kaggriculture_muzero_refactor_agent.py housekeeping --apply --force",
        "```",
        "",
    ])

    report = "\n".join(lines)
    out_dir = ROOT / "docs" / "vsa" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "refactor_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"[refactor] wrote {out_path}", flush=True)

    history = list(state.get("history") or [])
    history.append({"step": "report", "path": str(out_path.relative_to(ROOT))})
    return {"report": report, "history": history, "current_step": "complete"}


def build_refactor_graph():
    g = StateGraph(RefactorState)
    g.add_node("inventory", node_inventory)
    g.add_node("analyze", node_analyze)
    g.add_node("llm_plan", node_llm_plan)
    g.add_node("apply", node_apply)
    g.add_node("report", node_report)
    g.add_edge("inventory", "analyze")
    g.add_edge("analyze", "llm_plan")
    g.add_edge("llm_plan", "apply")
    g.add_edge("apply", "report")
    g.add_edge("report", END)
    g.set_entry_point("inventory")
    return g.compile()


def run_refactor(
    tasks: List[str],
    *,
    apply: bool = False,
    force: bool = False,
    output_file: Optional[str] = None,
) -> str:
    graph = build_refactor_graph()
    initial: RefactorState = {
        "tasks": tasks,
        "apply": apply,
        "force": force,
        "inventory": {},
        "findings": [],
        "actions": [],
        "llm_plan": {},
        "report": "",
        "history": [],
        "current_step": "init",
    }
    result = graph.invoke(initial)
    report = result["report"]
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(report, encoding="utf-8")
        print(f"[refactor] also wrote {output_file}", flush=True)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Kaggriculture MuZero refactor / housekeeping agent (LangGraph + Ollama)"
    )
    parser.add_argument(
        "task",
        nargs="?",
        default="all",
        help=f"Task name or 'all'. Choices: {', '.join(TASK_REGISTRY)}",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List tasks")
    parser.add_argument("--offline", action="store_true", help="Skip Ollama; static plan only")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write safe scaffolding (env.yml, .vscode, .cursor, gitignore, AGENTS banner)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow destructive apply (e.g. delete site/vsa/*.py) and mkdocs gh-deploy",
    )
    parser.add_argument("-o", "--output", help="Extra copy of refactor_report.md")
    args = parser.parse_args()

    if args.list:
        print("Refactor agent tasks:")
        for name, info in TASK_REGISTRY.items():
            print(f"  {name}: {info['description']}")
        return 0

    if args.offline:
        set_offline(True)
        print("[refactor] offline mode", flush=True)

    if args.task != "all" and args.task not in TASK_REGISTRY:
        print(f"Unknown task: {args.task}", file=sys.stderr)
        print(f"Available: {', '.join(TASK_REGISTRY)}", file=sys.stderr)
        return 2

    tasks = list(TASK_REGISTRY.keys()) if args.task == "all" else [args.task]
    run_refactor(tasks, apply=args.apply, force=args.force, output_file=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
