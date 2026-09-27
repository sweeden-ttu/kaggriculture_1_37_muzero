#!/usr/bin/env python3
"""Modified Pyright Language Server Protocol (LSP) Server for Kaggriculture MuZero.

This server:
1. Imports and initializes `kaggle_environments` and `kaggriculture` in Python.
2. Synchronizes PEP 561 / PEP 484 type stubs in `typings/kaggle_environments/`.
3. Verifies workspace pyright configuration (`pyrightconfig.json`).
4. Executes the Pyright Language Server over stdio or proxy mode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project directories and sys.path setup
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

TYPINGS_DIR = WORKSPACE_DIR / "typings" / "kaggle_environments"
DEFAULT_PYTHON = sys.executable

# Step 1: Pre-flight import of kaggle_environments and kaggriculture
try:
    import kaggle_environments
    import kaggle_environments.core
    import kaggle_environments.utils
    try:
        import kaggle_environments.envs.kaggriculture.kaggriculture as kagg_env
    except Exception as kagg_err:
        kagg_env = None
        sys.stderr.write(f"[pyright-lsp] Warning importing kaggriculture: {kagg_err}\n")
    KAGGLE_ENVIRONMENTS_AVAILABLE = True
except Exception as e:
    kaggle_environments = None
    kagg_env = None
    KAGGLE_ENVIRONMENTS_AVAILABLE = False
    sys.stderr.write(f"[pyright-lsp] Warning: kaggle_environments import failed: {e}\n")

# Step 2: Ensure stubs are imported and ready
from lsp.stub_generator import generate_kaggle_environments_stubs


def verify_and_sync_stubs(quiet: bool = True) -> List[Path]:
    """Ensure typing stubs exist and are up to date."""
    files = generate_kaggle_environments_stubs(TYPINGS_DIR)
    if not quiet:
        sys.stderr.write(f"[pyright-lsp] Verified {len(files)} stub files in {TYPINGS_DIR}\n")
    return files


def ensure_pyrightconfig(quiet: bool = True) -> Path:
    """Ensure pyrightconfig.json exists with project defaults."""
    config_path = WORKSPACE_DIR / "pyrightconfig.json"
    if not config_path.exists():
        default_config = {
            "venvPath": str(Path(sys.executable).parent.parent.parent),
            "venv": Path(sys.executable).parent.parent.name,
            "stubPath": "typings",
            "pythonVersion": "3.12",
            "pythonPlatform": "Darwin",
            "include": [
                "muzero",
                "hybrid_chassis",
                "pipelines",
                "evaluation",
                "packaging",
                "dist/main.py",
                "tests",
                "lsp",
                "main.py",
                "train_muzero.py",
                "package_submission.py",
                "tournament.py",
                "learn_from_replay.py",
            ],
            "exclude": [
                "**/node_modules",
                "**/__pycache__",
                ".git",
                "replays",
                "artifacts",
                "baselines",
                "_eval_*",
                "hybrid_chassis/layers",
            ],
            "reportMissingTypeStubs": False,
            "reportUnknownMemberType": False,
            "reportUnknownVariableType": False,
            "reportUnknownArgumentType": False,
        }
        config_path.write_text(json.dumps(default_config, indent=2), encoding="utf-8")
        if not quiet:
            sys.stderr.write(f"[pyright-lsp] Generated default configuration at {config_path}\n")
    return config_path


def run_pyright_langserver(args: List[str]) -> int:
    """Execute pyright language server with appropriate environment configuration."""
    verify_and_sync_stubs(quiet=True)
    ensure_pyrightconfig(quiet=True)

    # Configure environment
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    additional_paths = [str(WORKSPACE_DIR), str(WORKSPACE_DIR / "typings")]
    env["PYTHONPATH"] = (
        os.pathsep.join(additional_paths)
        + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    env["PYRIGHT_PYTHON_PATH"] = sys.executable

    # Check if pyright is installed
    try:
        from pyright.langserver import run as run_langserver
    except ImportError:
        # Fallback to binary invocation
        bin_path = Path(sys.executable).parent / "pyright-langserver"
        if not bin_path.exists():
            sys.stderr.write(f"[pyright-lsp] Error: pyright-langserver binary not found at {bin_path}\n")
            return 1
        return subprocess.run([str(bin_path), *args], env=env).returncode

    try:
        # Directly invoke pyright language server run
        res = run_langserver(*args, env=env)
        return res.returncode if hasattr(res, "returncode") else 0
    except Exception as e:
        sys.stderr.write(f"[pyright-lsp] Error executing pyright-langserver: {e}\n")
        return 1


def run_proxy_langserver(args: List[str]) -> int:
    """Execute pyright langserver as a supervised child process, proxying stdio JSON-RPC."""
    verify_and_sync_stubs(quiet=True)
    ensure_pyrightconfig(quiet=True)

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    additional_paths = [str(WORKSPACE_DIR), str(WORKSPACE_DIR / "typings")]
    env["PYTHONPATH"] = (
        os.pathsep.join(additional_paths)
        + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    env["PYRIGHT_PYTHON_PATH"] = sys.executable

    bin_path = Path(sys.executable).parent / "pyright-langserver"
    cmd = [str(bin_path) if bin_path.exists() else "pyright-langserver", "--stdio"]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
        bufsize=0,
    )

    import threading

    def stdin_forwarder():
        try:
            stdin_raw = sys.stdin.buffer
            while True:
                chunk = stdin_raw.read(4096)
                if not chunk:
                    break
                if proc.stdin:
                    proc.stdin.write(chunk)
                    proc.stdin.flush()
        except Exception:
            pass
        finally:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

    def stdout_forwarder():
        try:
            stdout_raw = sys.stdout.buffer
            while True:
                if not proc.stdout:
                    break
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                stdout_raw.write(chunk)
                stdout_raw.flush()
        except Exception:
            pass

    t_in = threading.Thread(target=stdin_forwarder, daemon=True)
    t_out = threading.Thread(target=stdout_forwarder, daemon=True)
    t_in.start()
    t_out.start()

    proc.wait()
    return proc.returncode


def print_status() -> None:
    """Print status of environment, kaggle_environments, and pyright setup."""
    print("=" * 60)
    print("Kaggriculture MuZero Modified Pyright LSP Server")
    print("=" * 60)
    print(f"Python Binary       : {sys.executable}")
    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"Workspace Directory : {WORKSPACE_DIR}")
    print(f"Typings Directory   : {TYPINGS_DIR}")
    print(f"Kaggle Envs Ready   : {KAGGLE_ENVIRONMENTS_AVAILABLE}")

    if KAGGLE_ENVIRONMENTS_AVAILABLE and kaggle_environments:
        print(f"Kaggle Envs Version : {getattr(kaggle_environments, '__version__', 'unknown')}")
        envs = list(getattr(kaggle_environments, "environments", {}).keys())
        has_kagg = "kaggriculture" in envs
        print(f"Registered Envs     : {len(envs)} (kaggriculture: {'READY' if has_kagg else 'MISSING'})")
        try:
            test_env = kaggle_environments.make("kaggriculture")
            print(f"Kaggriculture Test  : SUCCESS (version {test_env.specification.get('version', '0.1.0')})")
        except Exception as e:
            print(f"Kaggriculture Test  : FAILED ({e})")

    stubs = verify_and_sync_stubs(quiet=True)
    print(f"Active Stub Files   : {len(stubs)}")
    for s in stubs:
        print(f"  - {s.relative_to(WORKSPACE_DIR)}")
    print("=" * 60)


def run_type_check() -> int:
    """Run pyright CLI type checker against the workspace."""
    verify_and_sync_stubs(quiet=False)
    pyright_bin = Path(sys.executable).parent / "pyright"
    cmd = [str(pyright_bin) if pyright_bin.exists() else "pyright"]
    print(f"[pyright-lsp] Running {' '.join(cmd)} ...")
    return subprocess.run(cmd, cwd=WORKSPACE_DIR).returncode


def main() -> int:
    # If no arguments or called with --stdio, default to LSP server mode
    if len(sys.argv) == 1:
        # Default LSP invocation by IDE clients
        return run_pyright_langserver(["--stdio"])

    parser = argparse.ArgumentParser(
        description="Modified Pyright Language Server importing kaggle_environments",
        add_help=False,
    )
    parser.add_argument("--status", action="store_true", help="Display LSP status and environment diagnostics")
    parser.add_argument("--check", action="store_true", help="Run workspace pyright type check")
    parser.add_argument("--generate-stubs", action="store_true", help="Force stub regeneration for kaggle_environments")
    parser.add_argument("--proxy", action="store_true", help="Run in supervised JSON-RPC proxy mode")
    parser.add_argument("--stdio", action="store_true", help="Run LSP over stdio (standard IDE mode)")
    parser.add_argument("-h", "--help", action="store_true", help="Show help message")

    args, unknown = parser.parse_known_args()

    if args.status:
        print_status()
        return 0

    if args.generate_stubs:
        stubs = verify_and_sync_stubs(quiet=False)
        print(f"Generated {len(stubs)} stubs in {TYPINGS_DIR}")
        return 0

    if args.check:
        return run_type_check()

    if args.proxy:
        return run_proxy_langserver(unknown)

    if args.help:
        parser.print_help()
        print("\nPyright Langserver options:")
        # Forward help to pyright-langserver if requested
        return run_pyright_langserver(["--help"])

    # If --stdio or unknown arguments passed, forward to pyright language server
    langserver_args = list(unknown)
    if args.stdio and "--stdio" not in langserver_args:
        langserver_args.append("--stdio")
    if not langserver_args:
        langserver_args = ["--stdio"]

    return run_pyright_langserver(langserver_args)


if __name__ == "__main__":
    sys.exit(main())
