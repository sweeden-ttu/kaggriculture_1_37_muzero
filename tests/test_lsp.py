"""Unit tests for the modified Pyright LSP server and kaggle_environments typings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import kaggle_environments

from lsp.stub_generator import generate_kaggle_environments_stubs


def test_kaggle_environments_import():
    """Verify kaggle_environments is active and kaggriculture is registered."""
    assert "kaggriculture" in kaggle_environments.environments
    env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 10})
    assert env.name == "kaggriculture"
    assert env.configuration.get("episodeSteps") == 10
    state = env.reset()
    assert len(state) == 2
    assert hasattr(state[0], "reward")
    assert hasattr(state[0], "observation")
    assert state[0].reward == 0.0


def test_stub_generation(tmp_path: Path):
    """Test generating stubs produces all required PEP 561 / PEP 484 files."""
    stubs = generate_kaggle_environments_stubs(tmp_path)
    assert len(stubs) >= 8

    py_typed = tmp_path / "py.typed"
    assert py_typed.exists()

    init_pyi = tmp_path / "__init__.pyi"
    assert init_pyi.exists()
    content = init_pyi.read_text(encoding="utf-8")
    assert '"make"' in content
    assert '"evaluate"' in content
    assert '"Environment"' in content

    core_pyi = tmp_path / "core.pyi"
    assert core_pyi.exists()
    core_content = core_pyi.read_text(encoding="utf-8")
    assert "class Environment:" in core_content
    assert "def make(" in core_content
    assert "def evaluate(" in core_content
    assert "def reset(" in core_content
    assert "def step(" in core_content

    utils_pyi = tmp_path / "utils.pyi"
    assert utils_pyi.exists()
    utils_content = utils_pyi.read_text(encoding="utf-8")
    assert "class Struct(dict[str, Any]):" in utils_content
    assert "reward: Optional[float]" in utils_content

    kagg_pyi = tmp_path / "envs" / "kaggriculture" / "kaggriculture.pyi"
    assert kagg_pyi.exists()
    kagg_content = kagg_pyi.read_text(encoding="utf-8")
    assert "ANIMALS" in kagg_content
    assert "CROPS" in kagg_content
    assert "PRODUCTS" in kagg_content


def test_lsp_status_mode():
    """Verify pyright_lsp.py --status command executes cleanly."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "lsp" / "pyright_lsp.py"

    res = subprocess.run(
        [sys.executable, str(script_path), "--status"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert res.returncode == 0
    assert "Kaggle Envs Ready   : True" in res.stdout
    assert "kaggriculture: READY" in res.stdout
    assert "Active Stub Files" in res.stdout


def test_lsp_stdio_protocol():
    """Verify the LSP server communicates properly over stdio via JSON-RPC."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "lsp" / "pyright_lsp.py"

    proc = subprocess.Popen(
        [sys.executable, str(script_path), "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=repo_root,
    )

    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    pin = proc.stdin
    pout = proc.stdout

    try:
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": f"file://{repo_root}",
                "capabilities": {},
            },
        }
        body = json.dumps(init_req).encode("utf-8")
        pin.write(b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body)
        pin.flush()

        def read_msg():
            header = b""
            while not header.endswith(b"\r\n\r\n"):
                b = pout.read(1)
                if not b:
                    return None
                header += b
            length = int(header.split(b"Content-Length: ")[1].split(b"\r\n")[0])
            data = pout.read(length)
            return json.loads(data)

        # Pyright may send window/logMessage before the initialize response
        messages = []
        for _ in range(10):
            msg = read_msg()
            if not msg:
                break
            messages.append(msg)
            if msg.get("id") == 1:
                break

        init_responses = [m for m in messages if m.get("id") == 1]
        assert len(init_responses) == 1
        capabilities = init_responses[0].get("result", {}).get("capabilities", {})
        assert "hoverProvider" in capabilities
        assert "completionProvider" in capabilities
        assert "definitionProvider" in capabilities

        # Graceful shutdown
        shutdown = {"jsonrpc": "2.0", "id": 2, "method": "shutdown"}
        s_body = json.dumps(shutdown).encode("utf-8")
        pin.write(b"Content-Length: " + str(len(s_body)).encode("ascii") + b"\r\n\r\n" + s_body)
        pin.flush()
        s_resp = read_msg()
        assert s_resp is not None
        assert s_resp.get("id") == 2

        # Exit
        exit_req = {"jsonrpc": "2.0", "method": "exit"}
        e_body = json.dumps(exit_req).encode("utf-8")
        pin.write(b"Content-Length: " + str(len(e_body)).encode("ascii") + b"\r\n\r\n" + e_body)
        pin.flush()

        proc.wait(timeout=5)
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_pyright_resolves_kaggle_environments():
    """Verify Pyright type checks files using kaggle_environments without errors."""
    repo_root = Path(__file__).resolve().parent.parent
    pyright_bin = Path(sys.executable).parent / "pyright"
    if not pyright_bin.exists():
        pytest.skip("pyright binary not found")

    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=repo_root, delete=False) as f:
        f.write(
            """import kaggle_environments

env = kaggle_environments.make("kaggriculture")
state = env.reset()
r = state[0].reward
step_res = env.step(["PASS"])
"""
        )
        temp_file = Path(f.name)

    try:
        res = subprocess.run(
            [str(pyright_bin), str(temp_file)],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        assert res.returncode == 0
        assert "0 errors" in res.stdout
    finally:
        if temp_file.exists():
            temp_file.unlink()
