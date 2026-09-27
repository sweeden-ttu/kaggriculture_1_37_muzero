"""
Kaggriculture MuZero Comprehensive Specification & Documentation Agent
=====================================================================

Analyzes and documents the complete Kaggriculture MuZero architecture:
- Observation Encoder (spatial 28-channel 10x10 encoder)
- Factorized Prediction Head (Farmer/Hands/Market sub-heads)
- Latent SimSiam Consistency (asymmetric stop-gradient self-supervision)
- Sampled MCTS Search Tree (PUCT with Dirichlet, MinMaxStats)
- MuZero Core Chassis (h_θ, g_θ, f_θ spatial ResNet networks)
- PER Buffer & Reanalyze (trajectory-based, K-step unroll, n-step bootstrap, IS weights)
- Trainer Loop & Polyak EMA (dual network, multi-objective loss, Reanalyze passes)
- League Tracker & PFSP (AlphaStar-style matchmaking, win-rate prioritization)
- Consolidated Prioritized System (end-to-end training pipeline)
- GitHub Repository Setup (CI/CD, tests, integration tests)

Generates structured documentation covering architecture, testing, training, and evaluation.
"""

from __future__ import annotations

import os
import json
import re
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:  # pragma: no cover
    from typing_extensions import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Load environment
_project_dir = Path(__file__).resolve().parent.parent.parent
if (_project_dir / ".env").is_file():
    load_dotenv(_project_dir / ".env")


# ============================================================================
# STATE DEFINITION
# ============================================================================

class SpecAgentState(TypedDict):
    """State for the spec documentation agent."""
    component: str                          # Component being documented
    source_files: list                      # Source implementation files
    doc_files: list                         # Documentation files
    analysis: dict                          # Extracted analysis
    output: str                             # Generated documentation
    history: list                           # Processing history
    current_step: str                       # Current processing step


# ============================================================================
# COMPONENT REGISTRY - Maps components to their files
# ============================================================================

COMPONENT_REGISTRY: Dict[str, Dict[str, List[str]]] = {
    "observation_encoder": {
        "source": [
            "muzero/observation_encoder.py",
            "kaggriculture_observation_encoder.py",
        ],
        "docs": [
            "docs/Observation_Encoder.MD",
        ],
        "description": "Fixed maximum-grid 10x10 spatial feature encoder producing 28-channel tensors",
    },
    "factorized_prediction_head": {
        "source": [
            "muzero/chassis.py",
            "kaggriculture_muzero_chassis.py",
        ],
        "docs": [
            "docs/FACTORIZED_PREDICTION_HEAD.md",
        ],
        "description": "Factorized policy heads (Farmer 15, Hands 32, Market 20) with joint action sampling",
    },
    "simsiam_consistency": {
        "source": [
            "muzero/chassis.py",
            "simsiam_consistency.py",
        ],
        "docs": [
            "docs/Consistency_Loss.md",
        ],
        "description": "Asymmetric SimSiam projection/prediction heads with explicit stop-gradient for latent consistency",
    },
    "sampled_mcts": {
        "source": [
            "muzero/mcts.py",
            "sampled_muzero_mcts.py",
        ],
        "docs": [
            "docs/MCTS_Search_Tree.md",
        ],
        "description": "Sampled MCTS in latent space with PUCT, MinMaxStats, Dirichlet noise, factorized action sampling",
    },
    "muzero_core_chassis": {
        "source": [
            "muzero/chassis.py",
            "kaggriculture_muzero_chassis.py",
            "muzero/_core.py",
        ],
        "docs": [
            "docs/MuZero_Core.md",
        ],
        "description": "Unified spatial ResNet chassis: h_θ (representation), g_θ (dynamics), f_θ (prediction) networks",
    },
    "per_buffer_reanalyze": {
        "source": [
            "muzero/buffer.py",
            "prioritized_muzero_buffer.py",
            "muzero_replay_buffer.py",
        ],
        "docs": [
            "docs/prioritized_muzero_system.py",
            "docs/MCTS_Search_Tree.md",
        ],
        "description": "Trajectory-based PER buffer with K-step unroll, n-step bootstrap, Reanalyze target refreshing",
    },
    "trainer_loop_ema": {
        "source": [
            "muzero/trainer.py",
            "muzero_trainer_loop.py",
        ],
        "docs": [
            "docs/MCTS_Search_Tree.md",
            "docs/prioritized_muzero_system.py",
        ],
        "description": "Trainer with Polyak EMA target network, IS-weighted PER loss, SimSiam consistency, Reanalyze passes",
    },
    "league_tracker_pfsp": {
        "source": [
            "evaluation/league.py",
            "league_tracker.py",
        ],
        "docs": [
            "docs/League_Tracker.md",
        ],
        "description": "AlphaStar-style league with PFSP matchmaking: hard/var/uniform modes, role-specific ratios",
    },
    "consolidated_system": {
        "source": [
            "pipelines/train_muzero.py",
            "pipelines/low_rank_adaptation.py",
            "pipelines/adaptive_moment_estimation.py",
            "pipelines/replay_manager.py",
            "pipelines/self_improving_loop.py",
            "train.sh",
            "boost.sh",
        ],
        "docs": [
            "docs/prioritized_muzero_system.py",
            "AGENTS.md",
        ],
        "description": "End-to-end training pipeline: 4-phase curriculum, LoRA fine-tuning, continuous self-play loop",
    },
    "github_ci_cd": {
        "source": [
            ".github/workflows/ci.yml",
            ".github/workflows/cd.yml",
        ],
        "docs": [
            "docs/deployment/github-actions.md",
        ],
        "description": "GitHub repository setup: CI/CD workflows, unit tests, integration tests, and repository initialization",
    },
}


# ============================================================================
# LLM HELPER
# ============================================================================

def get_llm():
    """Get LLM for analysis (Ollama default). Always attempts to connect."""
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


# ============================================================================
# GITHUB CLI HELPERS
# ============================================================================

def run_gh_command(args: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Run a gh CLI command and return structured result."""
    cmd = ["gh"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
            "command": " ".join(cmd),
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "gh CLI not found. Install from https://cli.github.com/",
            "returncode": -1,
            "command": " ".join(cmd),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "gh command timed out",
            "returncode": -1,
            "command": " ".join(cmd),
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "command": " ".join(cmd),
        }


def check_gh_auth() -> Dict[str, Any]:
    """Check if gh is authenticated."""
    return run_gh_command(["auth", "status"])


def get_repo_info(owner: Optional[str] = None, repo: Optional[str] = None) -> Dict[str, Any]:
    """Get repository information."""
    args = ["repo", "view"]
    if owner and repo:
        args.append(f"{owner}/{repo}")
    return run_gh_command(args)


def create_repo(name: str, private: bool = False, description: str = "") -> Dict[str, Any]:
    """Create a new GitHub repository."""
    args = ["repo", "create", name]
    if private:
        args.append("--private")
    else:
        args.append("--public")
    if description:
        args.extend(["--description", description])
    return run_gh_command(args)


def push_to_github(remote_url: str, branch: str = "main") -> Dict[str, Any]:
    """Push local repository to GitHub."""
    result = run_gh_command(["repo", "sync", "--force"])
    if not result["success"]:
        return result
    return run_gh_command(["git", "push", remote_url, branch])


def create_workflow_file(workflow_name: str, content: str, repo_path: Path) -> Path:
    """Create a GitHub Actions workflow file."""
    workflows_dir = repo_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = workflows_dir / f"{workflow_name}.yml"
    workflow_path.write_text(content, encoding="utf-8")
    return workflow_path


def create_test_file(test_path: Path, content: str) -> Path:
    """Create a test file."""
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(content, encoding="utf-8")
    return test_path


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


# ============================================================================
# GRAPH NODES
# ============================================================================

def load_component_spec(state: SpecAgentState) -> Dict[str, Any]:
    """Load and parse the component specification."""
    component = state["component"]
    registry = COMPONENT_REGISTRY.get(component, {})
    
    source_files = registry.get("source", [])
    doc_files = registry.get("docs", [])
    
    analysis = parse_component_analysis(component, source_files, doc_files)
    
    history = state.get("history", [])
    history.append({"step": "load_spec", "component": component, "source_count": len(source_files), "doc_count": len(doc_files)})
    
    return {
        "source_files": source_files,
        "doc_files": doc_files,
        "analysis": analysis,
        "history": history,
        "current_step": "analyze",
    }


def analyze_architecture(state: SpecAgentState) -> Dict[str, Any]:
    """Analyze the architecture details using LLM (or static notes when offline)."""
    llm = get_llm()
    analysis = state["analysis"]
    component = state["component"]

# Build context from source files (keep prompts small — large contexts hang Ollama)
    project_root = Path(__file__).resolve().parent.parent.parent
    context_parts = []
    max_src = int(os.getenv("VSA_MAX_SRC_CHARS", "4000"))
    max_doc = int(os.getenv("VSA_MAX_DOC_CHARS", "2000"))
    max_total = int(os.getenv("VSA_MAX_CONTEXT_CHARS", "8000"))

    for src in state["source_files"]:
        full_path = project_root / src
        if full_path.exists():
            content = extract_code_content(full_path)
            context_parts.append(f"=== {src} ===\n{content[:max_src]}")

    for doc in state["doc_files"]:
        full_path = project_root / doc
        if full_path.exists():
            content = extract_code_content(full_path)
            context_parts.append(f"=== {doc} (DOC) ===\n{content[:max_doc]}")

    context = "\n\n".join(context_parts)[:max_total]

    system_prompt = (
        f"You are documenting the Kaggriculture MuZero `{component}` component. "
        "Reply with ONLY a single JSON object (no markdown fences) containing: "
        "architecture_notes (list[str]), hyperparameters (object), integration_points (list[str]). "
        "Keep each note under 200 characters."
    )

    user_prompt = f"Component: {component}\n\nContext:\n{context}"

    try:
        print(f"[vsa] LLM analyzing {component} (context={len(context)} chars)...", flush=True)
        response = llm.invoke([
            ("system", system_prompt),
            ("user", user_prompt),
        ])
        content = getattr(response, "content", "") or ""
        if isinstance(content, list):
            # Some providers return content blocks
            content = "".join(
                (c.get("text", "") if isinstance(c, dict) else str(c)) for c in content
            )
        # Strip optional markdown fences / thinking wrappers
        content = re.sub(r"thinking[\s\S]*?thinking", "", content).strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object in LLM response ({len(content)} chars)")
        parsed = json.loads(match.group())
        # Merge without wiping static class/function discovery
        for key in ("key_classes", "key_functions"):
            if key in parsed and isinstance(parsed[key], list):
                merged = list(dict.fromkeys(list(analysis.get(key) or []) + parsed[key]))
                analysis[key] = merged
                del parsed[key]
        analysis.update(parsed)
        print(f"[vsa] LLM analysis ok for {component}", flush=True)
    except Exception as e:
        analysis["architecture_notes"].append(f"LLM analysis failed: {e}")
        print(f"[vsa] LLM analysis failed for {component}: {e}", flush=True)

    history = state.get("history", [])
    history.append({"step": "analyze_architecture", "component": component})

    return {"analysis": analysis, "history": history, "current_step": "document_training"}


def document_training_workflow(state: SpecAgentState) -> Dict[str, Any]:
    """Document the training workflow for this component."""
    analysis = state["analysis"]
    component = state["component"]
    
    # Component-specific training workflows
    workflows = {
        "observation_encoder": [
            "Encode raw game observations to 28-channel 10x10 spatial tensors",
            "Used as input to RepresentationNetwork (h_θ) during initial_inference",
            "Action masks generated per factorized head (farmer/hands/market)",
        ],
        "factorized_prediction_head": [
            "Shared trunk processes latent state (64 channels, 10x10) to 256-dim features",
            "Three independent linear heads: farmer(15), hands(32), market(20)",
            "Value head outputs 601-bin distributional support",
            "sample_joint_actions() draws K candidates for MCTS expansion",
            "Loss: sum of 3 cross-entropy terms + value MSE",
        ],
        "simsiam_consistency": [
            "Projection head: 3-layer MLP (latent→proj→proj→proj) with BatchNorm",
            "Prediction head: 2-layer bottleneck MLP (proj→pred→proj)",
            "Stop-gradient on target branch: target_z = proj(h_θ(o)).detach()",
            "Loss: -cosine_similarity(predict(proj(s_unrolled)), target_z)",
            "Weight λ_cons ∈ [0.1, 0.5] (Regime 1) or [0.5, 1.0] (Regime 2)",
        ],
        "sampled_mcts": [
            "Root: initial_inference(obs) → s_0, priors, root_value",
            "Sample K=8 joint actions from factorized heads with Dirichlet noise",
            "Selection: PUCT with MinMaxStats Q-normalization",
            "Expansion: recurrent_inference(s, action_plane) → s_next, r, preds, v",
            "Backup: discounted value propagation, update MinMaxStats",
            "Output: visit-count policy π, best action, root value",
        ],
        "muzero_core_chassis": [
            "h_θ: SpatialRepresentationNetwork - Conv2d(28→64) + 3 ResNet blocks → L2 norm",
            "g_θ: SpatialDynamicsNetwork - Conv2d(80→64) + 3 ResNet blocks → reward_head + L2 norm",
            "f_θ: SpatialPredictionNetwork - Conv2d(64→32) + FC(3200→256) → 3 policy heads + value head",
            "initial_inference: obs → h_θ → f_θ",
            "recurrent_inference: (s, action_plane) → g_θ → f_θ",
            "DiscreteSupport: 601 bins, B=300, φ(x)=sign(x)(√(|x|+1)-1+ε|x|)",
        ],
        "per_buffer_reanalyze": [
            "GameTrajectory stores: obs, actions, rewards, policies (visit dist), values, priorities",
            "K-step slice sampling: random trajectory, random start, extract K+1 obs, K actions",
            "n-step bootstrap targets: z_t = Σγ^j r_{t+j} + γ^n v_{t+n}",
            "Factorized policy targets: marginalize joint visit dist to 3 sub-heads",
            "PER: priority = max_k |v_{t+k} - z_{t+k}| + ε, P(i) ∝ p_i^α",
            "IS weights: w_i = (N·P(i))^{-β}, β annealed 0.4→1.0",
            "Reanalyze: re-run MCTS on stored obs with latest weights, overwrite π_t, v_t",
        ],
        "trainer_loop_ema": [
            "Dual network: online (θ) + target (θ⁻) with Polyak EMA τ=0.995",
            "Joint optimizer over online model + SimSiam heads (AdamW, lr=3e-4)",
            "K-step unroll loss: policy + 0.25*value + reward + λ_cons*consistency",
            "PER: sample batch with IS weights, scale per-sample loss, update priorities",
            "Reanalyze pass: periodic (e.g. every 100 steps) refresh stale targets",
            "CosineAnnealingLR for uniform trainer, constant LR for PER trainer",
        ],
        "league_tracker_pfsp": [
            "LeagueTracker maintains checkpoints list + wins_matrix + games_matrix",
            "Laplace smoothing: win_rate = (wins + 1) / (games + 2)",
            "PFSP modes: hard=(1-x)^p, var=x(1-x), uniform=1",
            "Main agent: 35% self-play, 50% hard PFSP, 15% var PFSP",
            "Main exploiter: 50% vs current main, 50% var PFSP past mains",
            "League exploiter: 100% hard PFSP (p=3.0)",
        ],
        "consolidated_system": [
            "Phase 1 (BC): Behavioral cloning on gold replays with PER",
            "Phase 2 (Bootstrap): Analytical unroll targets, consistency loss",
            "Phase 3 (MCTS Distill): Gumbel MCTS targets distilled back to network",
            "Phase 4 (LoRA/PPO): LoRA rank-16 adapters on frozen champion + PPO microcontroller",
            "train.sh: standard loss schedule (consistency 0.1→0.5)",
            "boost.sh: inverted schedule (consistency 1.0→0.5, antagonistic gradients)",
            "Continuous loop: self-play → buffer → train → eval → promote champion",
            "Discrete head exports: policy_head.pt, value_head.pt per checkpoint",
            "Submission archetypes: head_pl (pure replay), head_hybrid, head_muzero",
        ],
    }
    
    analysis["training_workflow"] = workflows.get(component, ["See component documentation"])
    
    history = state.get("history", [])
    history.append({"step": "document_training", "component": component})
    
    return {"analysis": analysis, "history": history, "current_step": "document_testing"}


def document_testing_approach(state: SpecAgentState) -> Dict[str, Any]:
    """Document testing and validation approaches."""
    analysis = state["analysis"]
    component = state["component"]
    
    testing = {
        "observation_encoder": [
            "Unit test: encode() produces [1, 28, 10, 10] tensor",
            "Verify channel semantics: quadrant mask, crop types, livestock, units, global context",
            "Test action masks: land expansion gating, price thresholds",
            "Integration: feed into chassis.initial_inference() without shape errors",
        ],
        "factorized_prediction_head": [
            "Unit test: forward() returns dict with 4 logit tensors",
            "Verify shapes: farmer[B,15], hands[B,32], market[B,20], value[B,601]",
            "Test sample_joint_actions(): returns [B, K, 3] actions + [B, K] log_probs",
            "Verify masking: masked logits → -inf, softmax → 0 probability",
            "Gradient check: backward() flows through all three heads independently",
        ],
        "simsiam_consistency": [
            "Unit test: SimSiamConsistencyLoss forward returns scalar loss ∈ [-1, 0]",
            "Verify stop-gradient: target branch has no grad_fn",
            "Verify predictor head breaks symmetry (non-zero grad wrt unrolled latent)",
            "Integration: compute_muzero_unroll_loss_with_consistency() runs K-step loop",
            "Test L2 normalization prevents collapse to constant vectors",
        ],
        "sampled_mcts": [
            "Unit test: search() returns (best_action, probs_dict, root_value)",
            "Verify PUCT selection prefers high prior + high value",
            "Verify MinMaxStats normalizes Q-values across tree",
            "Test Dirichlet noise at root adds exploration",
            "Integration: chassis.recurrent_inference called for each expansion",
            "Stress test: num_simulations=50 completes in <1s on CPU",
        ],
        "muzero_core_chassis": [
            "Unit test: initial_inference(obs[B,28,10,10]) → s_0[B,64,10,10], preds, v_scalar",
            "Unit test: recurrent_inference(s[B,64,10,10], action[B,16,10,10]) → s_next, r, preds, v",
            "Verify L2 normalization on latent states (dim=1)",
            "Verify DiscreteSupport scalar conversion matches φ⁻¹(softmax(logits) @ support)",
            "Test LoRA injection: inject_lora(model, rank=4) adds trainable adapters",
        ],
        "per_buffer_reanalyze": [
            "Unit test: sample_k_step_batch() returns dict with correct tensor shapes",
            "Verify n-step targets use discount γ=0.997 correctly",
            "Test factorized target conversion: joint dict → 3 marginal arrays",
            "PER test: priorities updated after train_step, sampling probability ∝ p^α",
            "Reanalyze test: policies/values overwritten, priorities reset",
            "Integration: buffer + trainer + mcts end-to-end training step",
        ],
        "trainer_loop_ema": [
            "Unit test: train_step() returns loss dict with all components",
            "Verify EMA update: target params = τ*target + (1-τ)*online",
            "Test IS weighting: per-sample loss scaled by is_weights",
            "Verify gradient clipping at grad_clip=5.0",
            "Test Reanalyze pass updates buffer targets in-place",
            "Integration: 3 training steps reduce total loss",
        ],
        "league_tracker_pfsp": [
            "Unit test: add_checkpoint() expands matrices correctly",
            "Test get_win_rate() with Laplace smoothing: 0 games → 0.5",
            "Test PFSP hard mode: low win_rate → high weight",
            "Test PFSP var mode: win_rate≈0.5 → highest weight",
            "Test select_matchmaking_opponent() respects role ratios",
            "Integration: tournament.py uses league for opponent selection",
        ],
        "consolidated_system": [
            "Integration test: train.sh runs 1 epoch without errors",
            "Verify checkpoint saves: muzero_checkpoints.pt + policy_head.pt + value_head.pt",
            "Test LoRA fine-tune: --use-lora --lora-rank 4 freezes backbone",
            "Test submission packaging: build_head_submissions.py creates 3 zips",
            "Tournament test: benchmark against baselines/ and /Volumes/BASELINES/muzero/baselines",
            "Score validation: candidate scores vs Kaggle-scored baselines (2021-2033)",
        ],
    }
    
    analysis["testing_approach"] = testing.get(component, ["See component documentation"])
    
    history = state.get("history", [])
    history.append({"step": "document_testing", "component": component})
    
    return {"analysis": analysis, "history": history, "current_step": "document_evaluation"}


def document_evaluation(state: SpecAgentState) -> Dict[str, Any]:
    """Document evaluation metrics and benchmarking."""
    analysis = state["analysis"]
    component = state["component"]
    
    evaluation = {
        "observation_encoder": [
            "Encoding fidelity: reconstruction accuracy of key game state features",
            "Action mask coverage: % of illegal actions correctly masked",
            "Latency: <5ms per encode on CPU",
        ],
        "factorized_prediction_head": [
            "Policy accuracy: cross-entropy vs MCTS visit distributions",
            "Value accuracy: MSE vs bootstrapped n-step returns",
            "Joint action diversity: entropy across K sampled candidates",
            "Sub-head correlation: measure independence assumption validity",
        ],
        "simsiam_consistency": [
            "Consistency loss magnitude: track λ_cons * L_cons vs other losses",
            "Latent drift: cosine similarity between h_θ(o_{t+k}) and g_θ^k(s_0)",
            "Representation collapse check: variance of projected embeddings > threshold",
            "Ablation: training with/without consistency loss",
        ],
        "sampled_mcts": [
            "Search quality: root value vs final game outcome correlation",
            "Policy improvement: visit-count policy vs raw network policy (KL divergence)",
            "Simulation efficiency: nodes/sec, latency per search",
            "Ablation: num_samples (K=4,8,16), num_simulations (25,50,100)",
        ],
        "muzero_core_chassis": [
            "Parameter count: h_θ/g_θ/f_θ total ~1.2M params",
            "Inference latency: initial_inference + recurrent_inference on M1/M2",
            "Gradient flow: verify all params receive gradients in K-step unroll",
            "Checkpoint compatibility: load/save round-trip preserves weights",
        ],
        "per_buffer_reanalyze": [
            "Buffer utilization: trajectory count vs max_trajectories",
            "Priority distribution: histogram of TD-errors",
            "Reanalyze frequency: target staleness vs compute budget",
            "Sample efficiency: loss reduction per environment step",
        ],
        "trainer_loop_ema": [
            "Training curves: policy/value/reward/consistency loss over steps",
            "EMA tracking: online vs target network weight distance",
            "Learning rate schedule adherence",
            "Gradient norm statistics",
            "Reanalyze impact: loss on refreshed vs stale targets",
        ],
        "league_tracker_pfsp": [
            "League diversity: win-rate matrix heatmap",
            "PFSP effectiveness: main agent win-rate vs hard/var/uniform opponents",
            "Exploiter performance: main_exploiter win-rate vs current main",
            "Population collapse detection: exploiters >70% win-rate trigger reset",
        ],
        "consolidated_system": [
            "Kaggle submission scores: public/private leaderboard",
            "Tournament vs baselines: head-to-head win rates",
            "Self-play Elo progression over iterations",
            "Replay buffer rotation: canonical replay immunity, disk budget adherence",
            "Champion promotion gate: eval-episodes threshold",
            "LoRA merge: adapter weights folded into base for deployment",
        ],
    }
    
    analysis["evaluation_metrics"] = evaluation.get(component, ["See component documentation"])
    
    history = state.get("history", [])
    history.append({"step": "document_evaluation", "component": component})
    
    return {"analysis": analysis, "history": history, "current_step": "generate_output"}


def generate_output(state: SpecAgentState) -> Dict[str, Any]:
    """Generate final structured documentation output."""
    analysis = state["analysis"]
    component = state["component"]
    
    output_lines = [
        f"# {component.replace('_', ' ').title()} - Architecture Specification",
        f"",
        f"**Source**: {', '.join(state['source_files'])}",
        f"**Documentation**: {', '.join(state['doc_files'])}",
        f"**Generated**: {__import__('datetime').datetime.now().isoformat()}",
        f"",
        f"## Description",
        f"{analysis['description']}",
        f"",
        f"## Key Classes",
    ]
    
    for cls in analysis.get("key_classes", []):
        output_lines.append(f"- `{cls}`")
    
    output_lines.extend([
        f"",
        f"## Key Functions",
    ])
    
    for fn in analysis.get("key_functions", []):
        output_lines.append(f"- `{fn}()`")
    
    output_lines.extend([
        f"",
        f"## Hyperparameters",
    ])

    hyper = analysis.get("hyperparameters", {}) or {}
    if isinstance(hyper, dict):
        for k, v in hyper.items():
            output_lines.append(f"- `{k}`: `{v}`")
    elif isinstance(hyper, list):
        for item in hyper:
            output_lines.append(f"- {item}")
    else:
        output_lines.append(f"- {hyper}")

    if analysis.get("integration_points"):
        output_lines.extend(["", "## Integration Points"])
        for point in analysis["integration_points"]:
            output_lines.append(f"- {point}")

    output_lines.extend([
        f"",
        f"## Architecture Notes",
    ])
    
    for note in analysis.get("architecture_notes", []):
        output_lines.append(f"- {note}")
    
    output_lines.extend([
        f"",
        f"## Training Workflow",
    ])
    
    for step in analysis.get("training_workflow", []):
        output_lines.append(f"- {step}")
    
    output_lines.extend([
        f"",
        f"## Testing Approach",
    ])
    
    for test in analysis.get("testing_approach", []):
        output_lines.append(f"- {test}")
    
    output_lines.extend([
        f"",
        f"## Evaluation Metrics",
    ])
    
    for metric in analysis.get("evaluation_metrics", []):
        output_lines.append(f"- {metric}")
    
    # Artifact Analysis
    artifact_analysis = analysis.get("artifact_analysis", {})
    if artifact_analysis:
        output_lines.extend([
            f"",
            f"## Training Artifact Analysis",
        ])
        
        for category, findings in artifact_analysis.items():
            if findings:
                output_lines.extend([f"", f"### {category.replace('_', ' ').title()}"])
                for finding in findings:
                    output_lines.append(f"- ⚠️ {finding}")
            else:
                output_lines.extend([f"", f"### {category.replace('_', ' ').title()}", f"- ✅ No issues detected"])
    
    output = "\n".join(output_lines)
    
    history = state.get("history", [])
    history.append({"step": "generate_output", "component": component, "output_length": len(output)})
    
    return {"output": output, "history": history, "current_step": "complete"}


# ============================================================================
# BUILD GRAPH
# ============================================================================

def analyze_training_artifacts(state: SpecAgentState) -> Dict[str, Any]:
    """Analyze training artifacts (weights, logs, predictions, trajectories) for anti-patterns and issues."""
    analysis = state["analysis"]
    component = state["component"]
    project_root = Path(__file__).resolve().parent.parent.parent
    
    artifact_findings = {
        "weight_issues": [],
        "training_trend_issues": [],
        "prediction_issues": [],
        "trajectory_issues": [],
        "missing_files": [],
        "path_issues": [],
    }
    
    # Check artifacts directory
    artifacts_dir = project_root / "artifacts"
    if not artifacts_dir.exists():
        artifact_findings["missing_files"].append("artifacts/ directory not found")
    else:
        # Check for champion checkpoints
        champion_files = list(artifacts_dir.glob("*champion*"))
        if not champion_files:
            artifact_findings["missing_files"].append("No champion checkpoint found in artifacts/")
        
        # Check for policy/value head exports
        policy_heads = list(artifacts_dir.glob("*policy_head*.pt"))
        value_heads = list(artifacts_dir.glob("*value_head*.pt"))
        if not policy_heads:
            artifact_findings["missing_files"].append("No policy_head.pt exports found")
        if not value_heads:
            artifact_findings["missing_files"].append("No value_head.pt exports found")
        
        # Check for stale checkpoints (older than 7 days)
        import time
        now = time.time()
        for ckpt in artifacts_dir.glob("*.pt"):
            if (now - ckpt.stat().st_mtime) > 7 * 86400:
                artifact_findings["weight_issues"].append(f"Stale checkpoint: {ckpt.name} (older than 7 days)")
    
    # Check replay buffer
    replays_dir = project_root / "replays"
    if replays_dir.exists():
        canonical_replays = list(replays_dir.glob("[0-9]*.json"))
        selfplay_replays = list(replays_dir.glob("selfplay_*.json"))
        if len(canonical_replays) < 50:
            artifact_findings["trajectory_issues"].append(f"Only {len(canonical_replays)} canonical replays (expected 50+)")
        if len(selfplay_replays) > 1000:
            artifact_findings["trajectory_issues"].append(f"Excessive selfplay replays: {len(selfplay_replays)} (may need pruning)")
    
    # Check training logs for trend issues
    log_dirs = [
        project_root / "logs",
        project_root / "runs",
        project_root / "_eval_*",
    ]
    for log_pattern in log_dirs:
        for log_dir in project_root.glob(log_pattern.name) if "*" in log_pattern.name else [log_pattern]:
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("*.txt"))
                for log_file in log_files:
                    try:
                        content = log_file.read_text(encoding="utf-8", errors="ignore")
                        lines = content.strip().split("\n")
                        # Look for loss trends
                        loss_values = []
                        for line in lines[-100:]:  # Last 100 lines
                            if "loss" in line.lower():
                                import re
                                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                                if nums:
                                    loss_values.append(float(nums[-1]))
                        if len(loss_values) >= 10:
                            # Check if loss is increasing (bad trend)
                            recent_avg = sum(loss_values[-10:]) / 10
                            older_avg = sum(loss_values[-20:-10]) / 10 if len(loss_values) >= 20 else recent_avg
                            if recent_avg > older_avg * 1.1:
                                artifact_findings["training_trend_issues"].append(
                                    f"Loss increasing in {log_file.name}: {older_avg:.4f} → {recent_avg:.4f}"
                                )
                        # Check for NaN/inf
                        if "nan" in content.lower() or "inf" in content.lower():
                            artifact_findings["training_trend_issues"].append(f"NaN/Inf detected in {log_file.name}")
                    except Exception:
                        pass
    
    # Check for common path issues
    common_wrong_paths = [
        ("/Volumes/BASELINES/muzero", "read-only baselines"),
        ("/Volumes/TRAINREPLAYBOOST/muzero", "external training replay"),
        ("/Users/sweeden/muzero", "legacy scratch workspace"),
    ]
    for wrong_path, desc in common_wrong_paths:
        if Path(wrong_path).exists():
            artifact_findings["path_issues"].append(f"External path exists: {wrong_path} ({desc}) - ensure not accidentally writing to it")
    
    # Check dist/ for submission packages
    dist_dir = project_root / "dist"
    if dist_dir.exists():
        submissions = list(dist_dir.glob("*.zip"))
        if not submissions:
            artifact_findings["missing_files"].append("dist/ exists but no submission zips found")
    
    # Check for missing __init__.py in canonical packages
    for pkg in ["muzero", "pipelines", "evaluation", "packaging", "hybrid_chassis", "tests"]:
        pkg_dir = project_root / pkg
        if pkg_dir.exists() and not (pkg_dir / "__init__.py").exists():
            artifact_findings["missing_files"].append(f"{pkg}/__init__.py missing")
    
    # Add findings to analysis
    analysis["artifact_analysis"] = artifact_findings
    
    history = state.get("history", [])
    history.append({"step": "analyze_training_artifacts", "component": component, "findings": sum(len(v) for v in artifact_findings.values())})
    
    return {"analysis": analysis, "history": history, "current_step": "generate_output"}


def build_spec_agent_graph() -> StateGraph:
    """Build the specification documentation agent graph."""
    g = StateGraph(SpecAgentState)
    
    g.add_node("load_spec", load_component_spec)
    g.add_node("analyze_architecture", analyze_architecture)
    g.add_node("document_training", document_training_workflow)
    g.add_node("document_testing", document_testing_approach)
    g.add_node("document_evaluation", document_evaluation)
    g.add_node("analyze_training_artifacts", analyze_training_artifacts)
    g.add_node("generate_output", generate_output)
    
    g.add_edge("load_spec", "analyze_architecture")
    g.add_edge("analyze_architecture", "document_training")
    g.add_edge("document_training", "document_testing")
    g.add_edge("document_testing", "document_evaluation")
    g.add_edge("document_evaluation", "analyze_training_artifacts")
    g.add_edge("analyze_training_artifacts", "generate_output")
    g.add_edge("generate_output", END)
    
    g.set_entry_point("load_spec")
    
    return g.compile()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_spec_agent(component: str, output_file: Optional[str] = None) -> str:
    """Run the spec agent for a single component."""
    if component not in COMPONENT_REGISTRY:
        raise ValueError(f"Unknown component: {component}. Available: {list(COMPONENT_REGISTRY.keys())}")

    print(f"[vsa] Building graph for {component}...", flush=True)
    graph = build_spec_agent_graph()

    initial_state: SpecAgentState = {
        "component": component,
        "source_files": [],
        "doc_files": [],
        "analysis": {},
        "output": "",
        "history": [],
        "current_step": "init",
    }

    print(f"[vsa] Invoking graph...", flush=True)
    result = graph.invoke(initial_state)

    output = result["output"]

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)
        print(f"Documentation written to {output_file}", flush=True)

    return output


def run_all_components(output_dir: Optional[str] = None) -> Dict[str, str]:
    """Run spec agent for all registered components."""
    results = {}
    graph = build_spec_agent_graph()

    for component in COMPONENT_REGISTRY.keys():
        print(f"\n{'='*60}", flush=True)
        print(f"Processing: {component}", flush=True)
        print(f"{'='*60}", flush=True)

        initial_state: SpecAgentState = {
            "component": component,
            "source_files": [],
            "doc_files": [],
            "analysis": {},
            "output": "",
            "history": [],
            "current_step": "init",
        }

        result = graph.invoke(initial_state)
        results[component] = result["output"]

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{component}_spec.md"
            out_path.write_text(result["output"])
            print(f"  → Written to {out_path}", flush=True)

    # Generate master index
    if output_dir:
        index_lines = [
            "# Kaggriculture MuZero - Complete Architecture Specification Index",
            "",
            f"Generated: {__import__('datetime').datetime.now().isoformat()}",
            "",
            "## Components",
            "",
        ]
        for comp in COMPONENT_REGISTRY.keys():
            index_lines.append(f"- [{comp.replace('_', ' ').title()}]({comp}_spec.md)")

        index_path = Path(output_dir) / "INDEX.md"
        index_path.write_text("\n".join(index_lines))
        print(f"\nMaster index written to {index_path}", flush=True)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kaggriculture MuZero Spec Agent")
    parser.add_argument("component", nargs="?", help="Component to document (or 'all')")
    parser.add_argument("--output-dir", "-o", help="Output directory for markdown files")
    parser.add_argument("--list", "-l", action="store_true", help="List available components")

    args = parser.parse_args()

    if args.list:
        print("Available components:")
        for name, info in COMPONENT_REGISTRY.items():
            print(f"  {name}: {info['description']}")
        sys.exit(0)

    if not args.component:
        parser.print_help()
        sys.exit(1)

    if args.component == "all":
        run_all_components(args.output_dir)
    else:
        output_file = None
        if args.output_dir:
            output_file = str(Path(args.output_dir) / f"{args.component}_spec.md")
        run_spec_agent(args.component, output_file)