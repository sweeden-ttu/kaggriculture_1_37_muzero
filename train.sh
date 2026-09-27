#!/usr/bin/env bash
# =============================================================================
# train.sh - Kaggriculture MuZero Training & Self-Improvement Launcher
# =============================================================================
# Provides unified execution for:
#   1. Continuous Online Loop (Dirichlet self-play + snapshot league)
#   2. Targeted Offline Grid Search (Replay reanalyse + Bayesian search)
#   3. Hybrid Training (Warmup offline reanalyse -> continuous online loop)
#   4. Benchmark Tournament Evaluation (Tier 1-3 baselines)
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
    echo "[train.sh] Error: No python interpreter found." >&2
    exit 1
fi

# Performance & concurrency environment safeguards
# Prevents OpenMP / MKL thread oversaturation when running parallel evaluation workers
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Trap signals for graceful shutdown of child worker processes
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo -e "\n[train.sh] Process interrupted or exited with code ${exit_code}." >&2
    fi
    exit $exit_code
}
trap cleanup INT TERM

show_help() {
    cat << 'EOF'
Usage: ./train.sh [COMMAND|MODE] [OPTIONS...]

Architecture & Principles:
  - Top Tier Replay (Training Data): Raw fuel feeding the network to calculate loss & gradients.
  - Trained Champion Weights (Model State): Foundation model parameters. Freezes base when LoRA active.
  - Adam Optimizer: Applies gradient updates to model weights (or exclusively to LoRA adapter matrices).
  - Hyperparameters: Fixed dials (lr, rank, loss weights) tested via experimental sweeps (grid/adaptive), NOT gradients.
  - Three Phases: 1. Behavioral Cloning (mimic expert replays) -> 2. Bootstrapping (ground dynamics) -> 3. Distillation (MCTS lookahead).
  - Discrete Weight Files: Saves discrete policy head and value head weights alongside checkpoints.

Commands & Modes:
  online        Continuous online self-play loop with adaptive hyperparameter strategy (default)
                Flags passed: --mode online --continuous --search-strategy adaptive --eval-workers 8

   offline       Targeted offline reanalyze grid search (default: 10 iterations)
                 Flags passed: --mode offline --iterations 10 --search-strategy grid --eval-episodes 20 --eval-workers 8
                 Grid sweeps MuZero dials AND PPO microcontroller macrocontroller dials
                 (lr, gamma, clip, entropy, gae_lambda) — every 3rd trial is a PPO trial.

  hybrid        Continuous hybrid loop (interleaved self-play + reanalysis)
                Flags passed: --mode hybrid --continuous --search-strategy adaptive --eval-workers 8

  benchmark     Evaluate current dist/main.py agent against Tier 1-3 baseline packages
                Flags passed to tournament.py benchmark

  custom        Pass raw CLI flags directly to self_improving_loop.py

  help, -h      Display this help message

Examples:
  ./train.sh                                     # Default: Launch continuous online self-play loop
  ./train.sh online --eval-workers 4             # Run online loop with 4 eval workers
  ./train.sh offline --iterations 5              # Run 5 offline experimental sweep trials
  ./train.sh benchmark                           # Run round-robin benchmark tournament
  ./train.sh --use-lora --lora-rank 4            # Train LoRA adapters on frozen champion foundation
EOF
}

# Hardware optimization defaults for Apple Silicon M2 Pro (8 P-Cores, 4 E-Cores, 16GB RAM)
EVAL_WORKERS="${EVAL_WORKERS:-8}"
BASE_ITERATIONS="${BASE_ITERATIONS:-1000}"
SELFPLAY_GAMES="${SELFPLAY_GAMES:-50}"
LOSS_SCHEDULE="${LOSS_SCHEDULE:-low_rank_adaptation}"

# Inspect primary command or flag (defaults to continuous online mode)
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "help" ]]; then
    show_help
    exit 0
fi

if [[ "${1:-}" == -* ]] || [[ $# -eq 0 ]]; then
    MODE="online"
else
    MODE="$1"
    shift 1 2>/dev/null || true
fi

USE_SPATIAL=1
for arg in "$@"; do
    if [[ "$arg" == "macro" ]]; then
        USE_SPATIAL=0
    fi
done

case "$MODE" in
    online)
        echo "======================================================================="
        echo "  Starting Continuous Online Loop (Machine Optimized: 8 P-Cores)"
        echo "  Interpreter   : $PY_BIN"
        echo "  Root          : $PROJECT_ROOT"
        echo "  Architecture  : $(if [[ $USE_SPATIAL -eq 1 ]]; then echo "Spatial Sampled MuZero (Production)"; else echo "Legacy Macro MLP"; fi)"
        echo "  Eval Workers  : $EVAL_WORKERS"
        echo "  Base Iters    : $BASE_ITERATIONS"
        echo "  Self-Play     : $SELFPLAY_GAMES games (Parallel)"
        echo "  Loss Schedule : $LOSS_SCHEDULE (Standard Schedule)"
        echo "  Training Fuel : Top-Tier Replays (Raw Data Fuel -> Loss & Gradients)"
        echo "  Optimizer     : Adam (Applies Updates to Weights / LoRA Adapters)"
        echo "  Pipeline      : BC -> Bootstrapping -> MCTS Distillation"
        echo "======================================================================="
        if [[ $USE_SPATIAL -eq 1 ]]; then
            exec "$PY_BIN" -m pipelines.hybrid_trainer \
                --mode online \
                --continuous \
                --search-strategy adaptive \
                --eval-workers "$EVAL_WORKERS" \
                --base-iterations "$BASE_ITERATIONS" \
                --selfplay-games "$SELFPLAY_GAMES" \
                --loss-schedule "$LOSS_SCHEDULE" \
                "$@"
        else
            exec "$PY_BIN" self_improving_loop.py \
                --mode online \
                --continuous \
                --search-strategy adaptive \
                --eval-workers "$EVAL_WORKERS" \
                --base-iterations "$BASE_ITERATIONS" \
                --selfplay-games "$SELFPLAY_GAMES" \
                --loss-schedule "$LOSS_SCHEDULE" \
                "$@"
        fi
        ;;

    offline)
        echo "======================================================================="
        echo "  Starting Targeted Offline Reanalyse Grid Search"
        echo "  Interpreter   : $PY_BIN"
        echo "  Root          : $PROJECT_ROOT"
        echo "  Architecture  : $(if [[ $USE_SPATIAL -eq 1 ]]; then echo "Spatial Sampled MuZero (Production)"; else echo "Legacy Macro MLP"; fi)"
        echo "  Eval Workers  : $EVAL_WORKERS"
        echo "  Base Iters    : $BASE_ITERATIONS"
        echo "  Loss Schedule : $LOSS_SCHEDULE (Standard Schedule)"
        echo "  Training Fuel : Top-Tier Replays (Raw Data Fuel -> Loss & Gradients)"
        echo "  Optimizer     : Adam (Applies Updates to Weights / LoRA Adapters)"
        echo "  Pipeline      : BC -> Bootstrapping -> MCTS Distillation"
        echo "  PPO Grid      : Every 3rd trial sweeps PPO microcontroller dials"
        echo "======================================================================="
        if [[ $USE_SPATIAL -eq 1 ]]; then
            exec "$PY_BIN" -m pipelines.hybrid_trainer \
                --mode offline \
                --iterations 10 \
                --search-strategy grid \
                --eval-episodes 20 \
                --eval-workers "$EVAL_WORKERS" \
                --base-iterations "$BASE_ITERATIONS" \
                --loss-schedule "$LOSS_SCHEDULE" \
                "$@"
        else
            exec "$PY_BIN" self_improving_loop.py \
                --mode offline \
                --iterations 10 \
                --search-strategy grid \
                --eval-episodes 20 \
                --eval-workers "$EVAL_WORKERS" \
                --base-iterations "$BASE_ITERATIONS" \
                --loss-schedule "$LOSS_SCHEDULE" \
                "$@"
        fi
        ;;

    hybrid)
        echo "======================================================================="
        echo "  Starting Continuous Hybrid Training Loop (Machine Optimized: 8 P-Cores)"
        echo "  Interpreter   : $PY_BIN"
        echo "  Root          : $PROJECT_ROOT"
        echo "  Architecture  : $(if [[ $USE_SPATIAL -eq 1 ]]; then echo "Spatial Sampled MuZero (Production)"; else echo "Legacy Macro MLP"; fi)"
        echo "  Eval Workers  : $EVAL_WORKERS"
        echo "  Base Iters    : $BASE_ITERATIONS"
        echo "  Self-Play     : $SELFPLAY_GAMES games (Parallel)"
        echo "  Loss Schedule : $LOSS_SCHEDULE (Standard Schedule)"
        echo "  Training Fuel : Top-Tier Replays (Raw Data Fuel -> Loss & Gradients)"
        echo "  Optimizer     : Adam (Applies Updates to Weights / LoRA Adapters)"
        echo "  Pipeline      : BC -> Bootstrapping -> MCTS Distillation"
        echo "======================================================================="
        if [[ $USE_SPATIAL -eq 1 ]]; then
            exec "$PY_BIN" -m pipelines.hybrid_trainer \
                --mode hybrid \
                --continuous \
                --search-strategy adaptive \
                --eval-workers "$EVAL_WORKERS" \
                --base-iterations "$BASE_ITERATIONS" \
                --selfplay-games "$SELFPLAY_GAMES" \
                --loss-schedule "$LOSS_SCHEDULE" \
                "$@"
        else
            exec "$PY_BIN" self_improving_loop.py \
                --mode hybrid \
                --continuous \
                --search-strategy adaptive \
                --eval-workers "$EVAL_WORKERS" \
                --base-iterations "$BASE_ITERATIONS" \
                --selfplay-games "$SELFPLAY_GAMES" \
                --loss-schedule "$LOSS_SCHEDULE" \
                "$@"
        fi
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
        echo "[train.sh] Executing self_improving_loop.py with custom arguments: $*"
        exec "$PY_BIN" self_improving_loop.py "$@"
        ;;

    help|-h|--help)
        show_help
        exit 0
        ;;

    *)
        echo "[train.sh] Unknown command: '$MODE'" >&2
        echo "Run './train.sh --help' for available commands." >&2
        exit 1
        ;;
esac