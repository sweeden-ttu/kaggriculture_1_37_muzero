#!/usr/bin/env bash
# =============================================================================
# boost.sh - Kaggriculture MuZero Inverted Schedule Training Launcher
# =============================================================================
# Runs autonomous training with the INVERTED loss schedule:
#   - Loss weights start at the high end (e.g. policy ~0.897, value ~0.9637)
#     and work downward towards ~0.1357.
#   - Uses the exact same champion checkpoints, submissions, and snapshots
#     as train.sh.
# =============================================================================

set -euo pipefail

# Ensure execution from project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Resolve Python interpreter
if [[ -n "${PYTHON:-}" ]]; then
    PY_BIN="$PYTHON"
elif [[ -x "/Users/sweeden/miniforge3/envs/kagg/bin/python" ]]; then
    PY_BIN="/Users/sweeden/miniforge3/envs/kagg/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" ]] && [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
    PY_BIN="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PY_BIN="$(command -v python)"
else
    echo "[boost.sh] Error: No python interpreter found." >&2
    exit 1
fi

# Performance & concurrency environment safeguards
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Ensure root directory is on PYTHONPATH so sitecustomize executes across all child processes
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Activate inverted boost schedule mode across all parent and child processes
export MUZERO_SCHEDULE_MODE="boost"

# Trap signals for graceful shutdown of child worker processes
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo -e "\n[boost.sh] Process interrupted or exited with code ${exit_code}." >&2
    fi
    exit $exit_code
}
trap cleanup INT TERM

show_help() {
    cat << 'EOF'
Usage: ./boost.sh [COMMAND|MODE] [OPTIONS...]

Architecture & Principles:
  - Top Tier Replay (Training Data): Raw fuel feeding the network to calculate loss & gradients.
  - Trained Champion Weights (Model State): Foundation model parameters. Freezes base when LoRA active.
  - Adam Optimizer: Applies gradient updates to model weights (or exclusively to LoRA adapter matrices).
  - Hyperparameters: Fixed dials (lr, rank, loss weights) tested via experimental sweeps (grid/adaptive), NOT gradients.
  - Three Phases: 1. Behavioral Cloning (mimic expert replays) -> 2. Bootstrapping (ground dynamics) -> 3. Distillation (MCTS lookahead).
  - Discrete Weight Files: Saves discrete policy head and value head weights alongside checkpoints.

Commands & Modes:
  hybrid        Continuous hybrid loop (interleaved self-play + reanalysis) with boost schedule (default)
                Flags passed: --mode hybrid --continuous --search-strategy adaptive --eval-workers 8

  online        Continuous online self-play loop with adaptive hyperparameter strategy
                Flags passed: --mode online --continuous --search-strategy adaptive --eval-workers 8

   offline       Targeted offline reanalyze grid search (default: 20 iterations)
                 Flags passed: --mode offline --iterations 20 --search-strategy grid --eval-episodes 20 --eval-workers 8
                 Grid sweeps MuZero dials AND PPO microcontroller macrocontroller dials
                 (lr, gamma, clip, entropy, gae_lambda) — every 3rd trial is a PPO trial.

  benchmark     Evaluate current dist/main.py agent against Tier 1-3 baseline packages

  custom        Pass raw CLI flags directly to boost_loop.py

  help, -h      Display this help message

Description:
  boost.sh utilizes the inverted loss schedule dial where loss weights start at the high end:
    - Policy weight: starts high (~0.897) and works down towards ~0.1757
    - Value weight: starts high (~0.9637) and works down towards ~0.1957
    - Reward weight: starts high (~0.7357) and works down towards ~0.1357
    - Consistency weight: starts high (~0.5057) and works down towards ~0.1557
  All runs use the same champion foundation models, submissions, and snapshots as train.sh.
EOF
}

# Hardware optimization defaults for Apple Silicon M2 Pro (8 P-Cores, 4 E-Cores, 16GB RAM)
EVAL_WORKERS="${EVAL_WORKERS:-8}"
BASE_ITERATIONS="${BASE_ITERATIONS:-1000}"
SELFPLAY_GAMES="${SELFPLAY_GAMES:-8}"
LOSS_SCHEDULE="${LOSS_SCHEDULE:-adaptive_moment_estimation}"

# Inspect primary command or flag (defaults to continuous hybrid mode)
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "help" ]]; then
    show_help
    exit 0
fi

if [[ "${1:-}" == -* ]] || [[ $# -eq 0 ]]; then
    MODE="hybrid"
else
    MODE="$1"
    shift 1 2>/dev/null || true
fi

case "$MODE" in
    hybrid)
        echo "======================================================================="
        echo "  Starting Boost Hybrid Loop (Inverted Schedule Dial: High -> Low)"
        echo "  Interpreter   : $PY_BIN"
        echo "  Root          : $PROJECT_ROOT"
        echo "  Eval Workers  : $EVAL_WORKERS"
        echo "  Base Iters    : $BASE_ITERATIONS (P1=500, P2=300, P3=10)"
        echo "  Self-Play     : $SELFPLAY_GAMES games (Parallel)"
        echo "  Search        : ADAPTIVE (Experimental Sweep across Hyperparameter Dials)"
        echo "  Loss Schedule : $LOSS_SCHEDULE (Boost / Inverted Schedule)"
        echo "  Training Fuel : Top-Tier Replays (Raw Data Fuel -> Loss & Gradients)"
        echo "  Optimizer     : Adam (Applies Updates to Weights / LoRA Adapters)"
        echo "  Champion Model: artifacts/muzero_checkpoints_champion.pt (Foundation)"
        echo "======================================================================="
        exec "$PY_BIN" boost_loop.py \
            --mode hybrid \
            --continuous \
            --search-strategy adaptive \
            --eval-workers "$EVAL_WORKERS" \
            --base-iterations "$BASE_ITERATIONS" \
            --selfplay-games "$SELFPLAY_GAMES" \
            --loss-schedule "$LOSS_SCHEDULE" \
            "$@"
        ;;

    online)
        echo "======================================================================="
        echo "  Starting Boost Online Loop (Inverted Schedule Dial: High -> Low)"
        echo "  Interpreter   : $PY_BIN"
        echo "  Root          : $PROJECT_ROOT"
        echo "  Eval Workers  : $EVAL_WORKERS"
        echo "  Base Iters    : $BASE_ITERATIONS"
        echo "  Self-Play     : $SELFPLAY_GAMES games (Parallel)"
        echo "  Search        : ADAPTIVE (Experimental Sweep across Hyperparameter Dials)"
        echo "  Loss Schedule : $LOSS_SCHEDULE (Boost / Inverted Schedule)"
        echo "  Training Fuel : Top-Tier Replays (Raw Data Fuel -> Loss & Gradients)"
        echo "  Optimizer     : Adam (Applies Updates to Weights / LoRA Adapters)"
        echo "  Champion Model: artifacts/muzero_checkpoints_champion.pt (Foundation)"
        echo "======================================================================="
        exec "$PY_BIN" boost_loop.py \
            --mode online \
            --continuous \
            --search-strategy adaptive \
            --eval-workers "$EVAL_WORKERS" \
            --base-iterations "$BASE_ITERATIONS" \
            --selfplay-games "$SELFPLAY_GAMES" \
            --loss-schedule "$LOSS_SCHEDULE" \
            "$@"
        ;;

    offline)
        echo "======================================================================="
        echo "  Starting Boost Offline Reanalyse Grid Search"
        echo "  Interpreter   : $PY_BIN"
        echo "  Root          : $PROJECT_ROOT"
        echo "  Eval Workers  : $EVAL_WORKERS"
        echo "  Base Iters    : $BASE_ITERATIONS"
        echo "  Loss Schedule : $LOSS_SCHEDULE (Boost / Inverted Schedule)"
        echo "  Training Fuel : Top-Tier Replays (Raw Data Fuel -> Loss & Gradients)"
        echo "  Optimizer     : Adam (Applies Updates to Weights / LoRA Adapters)"
        echo "  Champion Model: artifacts/muzero_checkpoints_champion.pt (Foundation)"
        echo "  PPO Grid      : Every 3rd trial sweeps PPO microcontroller dials"
        echo "======================================================================="
        exec "$PY_BIN" boost_loop.py \
            --mode offline \
            --iterations 20 \
            --search-strategy grid \
            --eval-episodes 20 \
            --eval-workers "$EVAL_WORKERS" \
            --base-iterations "$BASE_ITERATIONS" \
            --loss-schedule "$LOSS_SCHEDULE" \
            "$@"
        ;;

    benchmark)
        echo "======================================================================="
        echo "  Running Benchmark Tournament vs Baseline Tiers"
        echo "  Interpreter   : $PY_BIN"
        echo "  Root          : $PROJECT_ROOT"
        echo "  Workers       : $EVAL_WORKERS"
        echo "======================================================================="
        exec "$PY_BIN" tournament.py benchmark \
            --agent dist/main.py \
            --baselines baselines \
            --games-per-baseline 6 \
            --seed 10101 \
            --workers "$EVAL_WORKERS" \
            --save-replays \
            "$@"
        ;;

    custom)
        echo "[boost.sh] Executing boost_loop.py with custom arguments: $*"
        exec "$PY_BIN" boost_loop.py "$@"
        ;;

    help|-h|--help)
        show_help
        exit 0
        ;;

    *)
        echo "[boost.sh] Unknown command: '$MODE'" >&2
        echo "Run './boost.sh --help' for available commands." >&2
        exit 1
        ;;
esac
