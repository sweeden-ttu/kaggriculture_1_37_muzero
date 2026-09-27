#!/usr/bin/env bash
# =============================================================================
# build.sh — Checkout → competition checkpoint → Kaggle submission
# =============================================================================
# Runs the production Make pipeline end-to-end with competition dials.
#
# Usage:
#   ./build.sh                     # full pipeline; package locally (no Kaggle upload)
#   ./build.sh --submit            # same + kaggle competitions submit
#   ./build.sh --smoke             # short curriculum (NOT for publication)
#   ./build.sh --skip-train        # reuse existing spatial ckpt; package + gate + submit
#   ./build.sh --with-loop         # after weights, run make loop-spatial
#   ./build.sh --with-hybrid       # after weights, run make hybrid
#   ./build.sh --with-tournament   # run make tournament before submit
#   ./build.sh --from distill      # resume from a named stage
#   ./build.sh --only package      # run a single stage
#   ./build.sh --message "msg"     # Kaggle submission message
#   ./build.sh --artifact muzero   # which zip: muzero|hybrid|pl|all (default: muzero)
#
# Stages (in order):
#   env → gold → distill → train → loop → hybrid → package → test
#     → gate → tournament → champion → submit
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DO_SUBMIT=0
SMOKE=0
SKIP_TRAIN=0
WITH_LOOP=0
WITH_HYBRID=0
WITH_TOURNAMENT=0
FROM_STAGE=""
ONLY_STAGE=""
SUBMIT_MESSAGE=""
ARTIFACT="muzero"   # muzero | hybrid | pl | all
KAGGLE_COMP="kaggriculture"
MAKE_EXTRA=()

STAGES=(env gold distill train loop hybrid package test gate tournament champion submit)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\n\033[1;36m==> [%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"; }
warn() { printf '\033[1;33m[build.sh] WARN:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[build.sh] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
build.sh — Checkout → competition checkpoint → Kaggle submission

Usage:
  ./build.sh                     full pipeline; package locally (no Kaggle upload)
  ./build.sh --submit            same + kaggle competitions submit
  ./build.sh --smoke             short curriculum (NOT for publication)
  ./build.sh --skip-train        reuse existing spatial ckpt; package + gate + submit
  ./build.sh --with-loop         after weights, run make loop-spatial
  ./build.sh --with-hybrid       after weights, run make hybrid
  ./build.sh --with-tournament   run make tournament before submit
  ./build.sh --from distill      resume from a named stage
  ./build.sh --only package      run a single stage
  ./build.sh --message "msg"     Kaggle submission message
  ./build.sh --artifact muzero   zip: muzero|hybrid|pl|all (default: muzero)
  ./build.sh PHASE1_STEPS=2000   forward Make VAR=value overrides

Stages:
  env → gold → distill → train → loop → hybrid → package → test
    → gate → tournament → champion → submit
EOF
    exit 0
}

stage_index() {
    local name="$1" i
    for i in "${!STAGES[@]}"; do
        if [[ "${STAGES[$i]}" == "$name" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

should_run() {
    local name="$1"
    local idx want
    if [[ -n "$ONLY_STAGE" ]]; then
        [[ "$name" == "$ONLY_STAGE" ]]
        return
    fi
    idx="$(stage_index "$name")" || return 1
    if [[ -n "$FROM_STAGE" ]]; then
        want="$(stage_index "$FROM_STAGE")" || die "Unknown --from stage: $FROM_STAGE"
        (( idx >= want ))
        return
    fi
    # Conditional stages
    case "$name" in
        train)       [[ "$SKIP_TRAIN" -eq 0 ]]; return ;;
        loop)        [[ "$WITH_LOOP" -eq 1 ]]; return ;;
        hybrid)      [[ "$WITH_HYBRID" -eq 1 ]]; return ;;
        tournament)  [[ "$WITH_TOURNAMENT" -eq 1 ]]; return ;;
        submit)      [[ "$DO_SUBMIT" -eq 1 ]]; return ;;
        *)           return 0 ;;
    esac
}

run_make() {
    log "make $*"
    make "$@" "${MAKE_EXTRA[@]}"
}

resolve_python() {
    if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
        echo "$PYTHON"
        return
    fi
    if [[ -x "/Users/sweeden/miniforge3/envs/kagg/bin/python" ]]; then
        echo "/Users/sweeden/miniforge3/envs/kagg/bin/python"
        return
    fi
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        echo "${CONDA_PREFIX}/bin/python"
        return
    fi
    command -v python3 || command -v python || die "No Python interpreter found"
}

activate_conda_kagg() {
    # Prefer the known miniforge kagg interpreter (matches Makefile) — avoid slow
    # `conda activate` / `conda env list` round-trips on every build.
    local preferred="/Users/sweeden/miniforge3/envs/kagg/bin/python"
    if [[ -x "$preferred" ]]; then
        export PATH="/Users/sweeden/miniforge3/envs/kagg/bin:${PATH}"
        export CONDA_PREFIX="/Users/sweeden/miniforge3/envs/kagg"
        export PYTHON="$preferred"
        log "using kagg python: $PYTHON"
        return
    fi
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        export PYTHON="${CONDA_PREFIX}/bin/python"
        log "using CONDA_PREFIX python: $PYTHON"
        return
    fi
    local conda_base
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]]; then
            # shellcheck source=/dev/null
            source "$conda_base/etc/profile.d/conda.sh"
            if [[ -x "$conda_base/envs/kagg/bin/python" ]]; then
                conda activate kagg
                export PYTHON="$(command -v python)"
                log "conda activate kagg → $PYTHON"
                return
            fi
            if [[ -f environment.yml ]]; then
                warn "conda env 'kagg' missing — creating from environment.yml"
                conda env create -f environment.yml
                conda activate kagg
                export PYTHON="$(command -v python)"
                return
            fi
        fi
    fi
    export PYTHON="$(resolve_python)"
    warn "conda kagg not found; using PYTHON=$PYTHON"
}

artifact_zip() {
    case "$1" in
        muzero)  echo "dist/head_muzero_submission.zip" ;;
        hybrid)  echo "dist/head_hybrid_submission.zip" ;;
        pl)      echo "dist/head_pl_submission.zip" ;;
        *)       die "Unknown artifact: $1 (use muzero|hybrid|pl|all)" ;;
    esac
}

kaggle_submit_one() {
    local zip_path="$1"
    local msg="$2"
    [[ -f "$zip_path" ]] || die "Missing submission zip: $zip_path"
    command -v kaggle >/dev/null 2>&1 || die "kaggle CLI not found (pip/conda install kaggle)"
    log "kaggle competitions submit -c $KAGGLE_COMP -f $zip_path"
    kaggle competitions submit -c "$KAGGLE_COMP" -f "$zip_path" -m "$msg"
}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --submit) DO_SUBMIT=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --skip-train) SKIP_TRAIN=1; shift ;;
        --with-loop) WITH_LOOP=1; shift ;;
        --with-hybrid) WITH_HYBRID=1; shift ;;
        --with-tournament) WITH_TOURNAMENT=1; shift ;;
        --from)
            FROM_STAGE="${2:-}"; [[ -n "$FROM_STAGE" ]] || die "--from needs a stage name"
            shift 2
            ;;
        --only)
            ONLY_STAGE="${2:-}"; [[ -n "$ONLY_STAGE" ]] || die "--only needs a stage name"
            shift 2
            ;;
        --message|-m)
            SUBMIT_MESSAGE="${2:-}"; shift 2
            ;;
        --artifact)
            ARTIFACT="${2:-}"; shift 2
            ;;
        --competition)
            KAGGLE_COMP="${2:-}"; shift 2
            ;;
        --) shift; MAKE_EXTRA+=("$@"); break ;;
        *=*)
            # Pass Make VAR=value overrides through
            MAKE_EXTRA+=("$1"); shift
            ;;
        *)
            die "Unknown flag: $1 (try --help)"
            ;;
    esac
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cleanup() {
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        warn "Aborted with exit code $rc"
    fi
    exit "$rc"
}
trap cleanup INT TERM EXIT

START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
log "Kaggriculture MuZero competition build"
log "root=$PROJECT_ROOT  smoke=$SMOKE  submit=$DO_SUBMIT  artifact=$ARTIFACT"

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

if should_run env; then
    log "STAGE env — conda / interpreter"
    activate_conda_kagg
    export PYTHON="$(resolve_python)"
    "$PYTHON" -c "import sys; print(f'python {sys.version.split()[0]} @ {sys.executable}')"
    if [[ ! -f Makefile ]]; then
        die "Makefile missing — run from repo root after checkout"
    fi
fi

if should_run gold; then
    log "STAGE gold — canonical Kaggle replays"
    run_make gold
fi

if should_run distill; then
    log "STAGE distill — expert policy lessons from replays"
    run_make distill
fi

if should_run train; then
    if [[ "$SMOKE" -eq 1 ]]; then
        log "STAGE train — weights-smoke (short; NOT publication)"
        run_make weights-smoke
    else
        log "STAGE train — competition curriculum (make weights)"
        run_make weights
    fi
fi

if should_run loop; then
    log "STAGE loop — continuous spatial RL (make loop-spatial)"
    run_make loop-spatial
fi

if should_run hybrid; then
    log "STAGE hybrid — hybrid online/offline trainer"
    run_make hybrid
fi

if should_run package; then
    log "STAGE package — promote + head submission zips"
    # promote is a dependency of build; submission also rebuilds weights — use build only
    run_make promote-weights
    run_make build
    ls -lh dist/head_*_submission.zip 2>/dev/null || warn "No head_*_submission.zip in dist/"
fi

if should_run test; then
    log "STAGE test — unit tests"
    run_make test
fi

if should_run gate; then
    log "STAGE gate — buoy H2H gate"
    CANDIDATE="$(artifact_zip muzero)"
    if [[ ! -f "$CANDIDATE" ]]; then
        CANDIDATE="dist/main_mz.py"
    fi
    if [[ -f "$CANDIDATE" ]] || [[ -f dist/head_muzero_submission.zip ]]; then
        run_make gate-buoy GATE_ARGS="--candidate ${CANDIDATE}"
    else
        warn "Skipping gate — no candidate package yet"
    fi
fi

if should_run tournament; then
    log "STAGE tournament — candidates vs baselines"
    run_make tournament
fi

if should_run champion; then
    log "STAGE champion — archive champion snapshot into baselines/"
    run_make champion || warn "champion archive skipped (no zip yet?)"
fi

if should_run submit; then
    log "STAGE submit — upload to Kaggle ($KAGGLE_COMP)"
    if [[ -z "$SUBMIT_MESSAGE" ]]; then
        SUBMIT_MESSAGE="Spatial Sampled MuZero competition build $(date -u '+%Y%m%dT%H%M%SZ')"
    fi
    if [[ "$ARTIFACT" == "all" ]]; then
        for kind in muzero hybrid pl; do
            kaggle_submit_one "$(artifact_zip "$kind")" "$SUBMIT_MESSAGE [$kind]"
        done
    else
        kaggle_submit_one "$(artifact_zip "$ARTIFACT")" "$SUBMIT_MESSAGE"
    fi
    log "Listing recent Kaggle submissions..."
    kaggle competitions submissions -c "$KAGGLE_COMP" 2>/dev/null | head -20 || true
fi

# Always prune stale self-play at the end of a full run (cheap hygiene)
if [[ -z "$ONLY_STAGE" ]] && should_run package; then
    log "Hygiene — prune self-play replays under budget"
    run_make prune || warn "prune skipped"
fi

log "DONE  started=$START_TS  finished=$(date '+%Y-%m-%d %H:%M:%S')"
if [[ "$DO_SUBMIT" -eq 0 && -z "$ONLY_STAGE" ]]; then
    log "Packages ready under dist/. Re-run with --submit to upload to Kaggle."
fi

trap - EXIT
exit 0
