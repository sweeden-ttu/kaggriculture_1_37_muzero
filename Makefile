# Kaggriculture MuZero — build, weights & diagnostics
# Production architecture: spatial Sampled MuZero (ADR 0001)
# Default dials = COMPETITION / publication long-horizon (not smoke)
# Target interpreter: miniforge conda env `kagg` (Python 3.12)
#
# Override examples:
#   make weights                      # full competition curriculum → spatial ckpt
#   make weights-smoke                # short sanity curriculum (opt-in)
#   make smoke-spatial                # 1-step I/O smoke (opt-in)
#   make loop-spatial                 # continuous competition RL loop
#   make gate-buoy
#   make diagnostics DIAGNOSTICS_ARGS='--step 194 --seat 0'

PYTHON ?= /Users/sweeden/miniforge3/envs/kagg/bin/python
export PYTHON
export PYTHONPATH := $(CURDIR):$(PYTHONPATH)
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export PYTHONUNBUFFERED := 1

REPLAYS_DIR := replays
ARTIFACTS := artifacts
DIST := dist
BASELINES := baselines
COMPETITION_DIR := competition
GOLD_THRESHOLD := 35

# Spatial (production) vs legacy macro MLP
ARCH ?= spatial
SPATIAL_CKPT := $(ARTIFACTS)/muzero_spatial_checkpoints.pt
MACRO_CKPT := $(ARTIFACTS)/muzero_checkpoints.pt
MACRO_CHAMP_CKPT := $(ARTIFACTS)/muzero_checkpoints_champion.pt
ifeq ($(ARCH),spatial)
  CKPT_OUT := $(SPATIAL_CKPT)
else
  CKPT_OUT := $(MACRO_CKPT)
endif

# Pass-through argument bags
DIAGNOSTICS_ARGS ?=
BENCHMARK_ARGS ?=
TOURNAMENT_ARGS ?=
BUILD_ARGS ?=
EMBED_ARGS ?=
PRUNE_ARGS ?=
PPO_ARGS ?=
WEIGHTS_ARGS ?=
LOOP_ARGS ?=
GATE_ARGS ?=

# ---------------------------------------------------------------------------
# Production / competition / publication long-horizon dials (DEFAULT)
# Docs: MCTS 50 sims (eval), k=5, buffer 1000, train-steps ~1000/phase
# ---------------------------------------------------------------------------
PHASE1_STEPS ?= 1000
PHASE2_STEPS ?= 400
PHASE3_STEPS ?= 1000
PHASE4_STEPS ?= 400
SELFPLAY_EPISODES ?= 50
BATCH ?= 64
SIMS ?= 50
LR ?= 1e-3
K_STEPS ?= 5
BUFFER_MAX ?= 1000

# Back-compat aliases (same as production defaults)
COMP_PHASE1_STEPS ?= $(PHASE1_STEPS)
COMP_PHASE2_STEPS ?= $(PHASE2_STEPS)
COMP_PHASE3_STEPS ?= $(PHASE3_STEPS)
COMP_PHASE4_STEPS ?= $(PHASE4_STEPS)
COMP_SELFPLAY ?= $(SELFPLAY_EPISODES)
COMP_BATCH ?= $(BATCH)
COMP_SIMS ?= $(SIMS)
COMP_K_STEPS ?= $(K_STEPS)

# Opt-in short curriculum (make weights-smoke only)
SMOKE_PHASE1_STEPS ?= 120
SMOKE_PHASE2_STEPS ?= 30
SMOKE_PHASE3_STEPS ?= 40
SMOKE_PHASE4_STEPS ?= 20
SMOKE_SELFPLAY ?= 4
SMOKE_BATCH ?= 32
SMOKE_SIMS ?= 16
SMOKE_K_STEPS ?= 3

# Spatial continuous loop (competition)
ITERATIONS ?= 50
PROMOTE_EVERY ?= 1

# Hybrid online/offline trainer dials (ADR 0001) — production
HYBRID_ITERATIONS ?= 48
HYBRID_BASE_ITERS ?= 1000
HYBRID_SELFPLAY ?= 50
HYBRID_EVAL_EPISODES ?= 30
HYBRID_STRATEGY ?= adaptive
HYBRID_ARGS ?=

# Legacy macro-only dials (--arch macro)
PHASE3_EPOCHS ?= $(PHASE4_STEPS)
LORA_RANK ?= 16
LORA_ALPHA ?= 16.0

PPO_TRIALS ?= 12
PPO_ROUNDS ?= 20
GOLD_TARGET ?= 50
PRUNE_MAX_GB ?= 10.0

# Inference MCTS depth for packaged agents
export MUZERO_SPATIAL_SIMS ?= $(SIMS)
export MUZERO_SPATIAL_SAMPLES ?= 8

# Shared args for isolated spatial phase recipes
SPATIAL_TRAIN_COMMON := --arch spatial \
	--batch $(BATCH) \
	--sims $(SIMS) \
	--lr $(LR) \
	--k-steps $(K_STEPS) \
	--buffer-max $(BUFFER_MAX) \
	--out $(SPATIAL_CKPT) \
	$(WEIGHTS_ARGS)

# Canonical Kaggle replays: ^[0-9]+\.json$ (selfplay_*.json excluded)
CANONICAL_COUNT := $(shell ls $(REPLAYS_DIR) 2>/dev/null | grep -E '^[0-9]+\.json$$' | wc -l | tr -d ' ')

.PHONY: help \
	clean gold replays prune distill \
	phase-0 phase-1 phase-2 phase-3 phase-4 phases \
	refactor-plan refactor-apply \
	from-checkout pipeline \
	weights weights-spatial weights-competition weights-smoke weights-macro train-weights promote-weights embed-weights \
	loop-spatial \
	hybrid sweep-offline online-spatial \
	ppo-grid ppo-train \
	test build submission champion \
	benchmark tournament tounament \
	league league-rr league-pfsp league-sample league-status \
	dashboard gate-buoy \
	diagnostics concurrency \
	smoke-spatial smoke-macro

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:
	@echo "Kaggriculture MuZero Make targets (production ARCH=spatial)"
	@echo ""
	@echo "Data / replays"
	@echo "  clean / gold / prune / distill"
	@echo ""
	@echo "End-to-end competition pipeline"
	@echo "  from-checkout / pipeline   ./build.sh (checkout → weights → package → gate)"
	@echo "                             Add BUILD_SH_ARGS='--submit' to upload to Kaggle"
	@echo ""
	@echo "Spatial curriculum (isolated slices)"
	@echo "  phase-0   Download/bootstrap gold Kaggle replays into $(REPLAYS_DIR)/"
	@echo "  phase-1   Expert BC (PHASE1_STEPS=$(PHASE1_STEPS))"
	@echo "  phase-2   SimSiam / dynamics bootstrap (PHASE2_STEPS=$(PHASE2_STEPS))"
	@echo "  phase-3   Sampled-MCTS distill + self-play (PHASE3_STEPS=$(PHASE3_STEPS))"
	@echo "  phase-4   Reanalyze refresh (PHASE4_STEPS=$(PHASE4_STEPS))"
	@echo "  phases    Run phase-0 → phase-4 in order"
	@echo ""
	@echo "Repo housekeeping (LangGraph + Ollama)"
	@echo "  refactor-plan         Dry-run inventory / gaps / IDE / docs plan"
	@echo "  refactor-apply        Write env.yml, .vscode, .cursor rules, gitignore"
	@echo ""
	@echo "Build weights — COMPETITION defaults (CKPT_OUT=$(CKPT_OUT))"
	@echo "  weights / weights-spatial / weights-competition"
	@echo "                        BC=$(PHASE1_STEPS) boot=$(PHASE2_STEPS) distill=$(PHASE3_STEPS)"
	@echo "                        reanalyze=$(PHASE4_STEPS) selfplay=$(SELFPLAY_EPISODES)"
	@echo "                        batch=$(BATCH) sims=$(SIMS) k=$(K_STEPS) buffer=$(BUFFER_MAX)"
	@echo "                        → $(SPATIAL_CKPT)"
	@echo "  loop-spatial          Continuous RL loop (ITERATIONS=$(ITERATIONS))"
	@echo "  hybrid                Hybrid online/offline (iters=$(HYBRID_ITERATIONS) base=$(HYBRID_BASE_ITERS))"
	@echo "  promote-weights       Validate + copy spatial → packaging aliases"
	@echo "  weights-smoke         Opt-in short curriculum (NOT for publication)"
	@echo "  smoke-spatial         Opt-in 1-step I/O smoke"
	@echo ""
	@echo "Package / evaluate"
	@echo "  build / submission / gate-buoy / tournament / test"
	@echo ""
	@echo "Leagues (PFSP — docs/League_Tracker.md)"
	@echo "  league / league-rr    Round-robin all-vs-all (+ optional tracker)"
	@echo "  league-pfsp           Prioritized fictitious self-play matches"
	@echo "  league-sample         Dry-run PFSP matchmaking samples (no games)"
	@echo "  league-status         Print saved LeagueTracker leaderboard"
	@echo "  Override dirs/games:  make league-pfsp TOURNAMENT_ARGS='--dirs dist baselines --games 16'"
	@echo ""
	@echo "Diagnostics: diagnostics | concurrency"

# ---------------------------------------------------------------------------
# Data / replays
# ---------------------------------------------------------------------------

clean:
	rm -rf $(REPLAYS_DIR)/*.json
	@echo "[make] Cleared replay folder ($(REPLAYS_DIR))."

gold:
	@if [ "$(CANONICAL_COUNT)" -le "$(GOLD_THRESHOLD)" ]; then \
		echo "[make] Found $(CANONICAL_COUNT) canonical replays (threshold $(GOLD_THRESHOLD)) — downloading..."; \
		$(PYTHON) scripts/download_top_gold_replays.py --target-count $(GOLD_TARGET); \
	else \
		echo "[make] Found $(CANONICAL_COUNT) canonical replays (threshold $(GOLD_THRESHOLD)) — skipping download."; \
	fi

replays: gold

prune:
	$(PYTHON) scripts/prune_replays.py --replays-dir $(REPLAYS_DIR) --max-size-gb $(PRUNE_MAX_GB) $(PRUNE_ARGS)

distill: gold
	$(PYTHON) learn_from_replay.py --all

# ---------------------------------------------------------------------------
# Spatial curriculum phases (0 = data, 1–4 = train slices)
# ---------------------------------------------------------------------------

# Phase 0: bootstrap gold Kaggle replay JSON into replays/
phase-0: gold
	@echo "[make] phase-0 complete — canonical count=$$(ls $(REPLAYS_DIR) 2>/dev/null | grep -E '^[0-9]+\.json$$' | wc -l | tr -d ' ')"

# Phase 1: expert behavioral cloning into spatial chassis
phase-1: phase-0
	@echo "[make] phase-1 BC steps=$(PHASE1_STEPS) → $(SPATIAL_CKPT)"
	$(PYTHON) -m pipelines.train_muzero \
		$(SPATIAL_TRAIN_COMMON) \
		--spatial-phase bc \
		--phase1-steps $(PHASE1_STEPS) \
		--phase2-steps 0 \
		--phase4-steps 0 \
		--selfplay-episodes 0 \
		--summary-out $(ARTIFACTS)/train_summary_phase1.json

# Phase 2: SimSiam / dynamics bootstrap on shared (or synthetic) buffer
phase-2:
	@echo "[make] phase-2 bootstrap steps=$(PHASE2_STEPS) → $(SPATIAL_CKPT)"
	$(PYTHON) -m pipelines.train_muzero \
		$(SPATIAL_TRAIN_COMMON) \
		--spatial-phase bootstrap \
		--phase1-steps 0 \
		--phase2-steps $(PHASE2_STEPS) \
		--phase4-steps 0 \
		--selfplay-episodes 0 \
		--summary-out $(ARTIFACTS)/train_summary_phase2.json

# Phase 3: Sampled-MCTS self-play harvest + policy/value distill
phase-3:
	@echo "[make] phase-3 MCTS distill steps=$(PHASE3_STEPS) selfplay=$(SELFPLAY_EPISODES) → $(SPATIAL_CKPT)"
	$(PYTHON) -m pipelines.train_muzero \
		$(SPATIAL_TRAIN_COMMON) \
		--spatial-phase distill \
		--phase1-steps 0 \
		--phase2-steps 0 \
		--phase4-steps $(PHASE3_STEPS) \
		--selfplay-episodes $(SELFPLAY_EPISODES) \
		--summary-out $(ARTIFACTS)/train_summary_phase3.json

# Phase 4: reanalyze — refresh MCTS π/v targets, continue Adam
phase-4:
	@echo "[make] phase-4 reanalyze steps=$(PHASE4_STEPS) → $(SPATIAL_CKPT)"
	$(PYTHON) -m pipelines.train_muzero \
		$(SPATIAL_TRAIN_COMMON) \
		--spatial-phase reanalyze \
		--phase1-steps 0 \
		--phase2-steps 0 \
		--phase3-epochs $(PHASE4_STEPS) \
		--phase4-steps 0 \
		--selfplay-episodes 0 \
		--summary-out $(ARTIFACTS)/train_summary_phase4.json

# Full sequential curriculum via isolated targets
phases: phase-0 phase-1 phase-2 phase-3 phase-4
	@$(MAKE) promote-weights
	@echo "[make] phases 0–4 complete → $(SPATIAL_CKPT)"

# ---------------------------------------------------------------------------
# End-to-end: checkout → competition weights → package → gate → (optional) Kaggle
# ---------------------------------------------------------------------------

BUILD_SH_ARGS ?=

from-checkout pipeline:
	./build.sh $(BUILD_SH_ARGS)

# Repo refactor / housekeeping agent (docs/vsa)
# ---------------------------------------------------------------------------

refactor-plan:
	$(PYTHON) docs/vsa/kaggriculture_muzero_refactor_agent.py all --offline

refactor-apply:
	$(PYTHON) docs/vsa/kaggriculture_muzero_refactor_agent.py all --apply --offline

# ---------------------------------------------------------------------------
# Build weights (spatial production path)
# ---------------------------------------------------------------------------

weights: weights-spatial

# Curriculum: BC (phase1) → bootstrap (phase2) → MCTS distill (phase3) → reanalyze (phase4)
# Defaults are COMPETITION / publication long-horizon.
weights-spatial: gold
	@echo "[make] Building COMPETITION spatial Sampled MuZero weights → $(SPATIAL_CKPT)"
	@echo "[make]   bc=$(PHASE1_STEPS) boot=$(PHASE2_STEPS) distill=$(PHASE3_STEPS) reanalyze=$(PHASE4_STEPS)"
	@echo "[make]   selfplay=$(SELFPLAY_EPISODES) batch=$(BATCH) sims=$(SIMS) k=$(K_STEPS) buffer=$(BUFFER_MAX)"
	$(PYTHON) -m pipelines.train_muzero \
		--arch spatial \
		--spatial-phase all \
		--phase1-steps $(PHASE1_STEPS) \
		--phase2-steps $(PHASE2_STEPS) \
		--phase3-epochs $(PHASE4_STEPS) \
		--phase4-steps $(PHASE3_STEPS) \
		--batch $(BATCH) \
		--sims $(SIMS) \
		--lr $(LR) \
		--k-steps $(K_STEPS) \
		--buffer-max $(BUFFER_MAX) \
		--selfplay-episodes $(SELFPLAY_EPISODES) \
		--out $(SPATIAL_CKPT) \
		--summary-out $(ARTIFACTS)/train_summary_spatial.json \
		$(WEIGHTS_ARGS)
	@$(MAKE) promote-weights

# Alias: defaults already are competition / publication dials
weights-competition: weights-spatial

# Opt-in short sanity curriculum (NOT for publication checkpoints)
weights-smoke:
	$(MAKE) weights-spatial \
		PHASE1_STEPS=$(SMOKE_PHASE1_STEPS) \
		PHASE2_STEPS=$(SMOKE_PHASE2_STEPS) \
		PHASE3_STEPS=$(SMOKE_PHASE3_STEPS) \
		PHASE4_STEPS=$(SMOKE_PHASE4_STEPS) \
		SELFPLAY_EPISODES=$(SMOKE_SELFPLAY) \
		BATCH=$(SMOKE_BATCH) \
		SIMS=$(SMOKE_SIMS) \
		K_STEPS=$(SMOKE_K_STEPS) \
		BUFFER_MAX=256

weights-macro: gold
	@echo "[make] Building legacy macro MLP weights → $(MACRO_CKPT)"
	$(PYTHON) -m pipelines.train_muzero \
		--arch macro \
		--use-lora \
		--lora-rank $(LORA_RANK) \
		--lora-alpha $(LORA_ALPHA) \
		--phase1-steps $(PHASE1_STEPS) \
		--phase2-steps $(PHASE2_STEPS) \
		--phase3-epochs $(PHASE3_EPOCHS) \
		--phase4-steps $(PHASE4_STEPS) \
		--batch $(BATCH) \
		--sims $(SIMS) \
		--lr $(LR) \
		--k-steps $(K_STEPS) \
		--buffer-max $(BUFFER_MAX) \
		--out $(MACRO_CKPT) \
		--summary-out $(ARTIFACTS)/train_summary_macro.json \
		$(WEIGHTS_ARGS)

train-weights:
	@if [ "$(ARCH)" = "macro" ]; then $(MAKE) weights-macro; else $(MAKE) weights-spatial; fi

# Validate spatial loadability then copy to packaging aliases.
promote-weights:
	@if [ ! -f "$(SPATIAL_CKPT)" ]; then \
		echo "[make] No spatial checkpoint at $(SPATIAL_CKPT); skip promote."; \
		exit 0; \
	fi
	@echo "[make] Promoting $(SPATIAL_CKPT) → packaging checkpoint names (validated)"
	$(PYTHON) -c "from muzero.spatial_checkpoint import promote_spatial_champion; \
		print(promote_spatial_champion('$(SPATIAL_CKPT)'))"

embed-weights: promote-weights
	$(PYTHON) scripts/embed_muzero_weights.py \
		--ckpt $(CKPT_OUT) \
		$(EMBED_ARGS)

# Tiny smoke: skip BC + distill; 1 bootstrap step on synthetic buffer
smoke-spatial:
	$(PYTHON) -m pipelines.train_muzero \
		--arch spatial \
		--phase1-steps 0 \
		--phase2-steps 1 \
		--phase4-steps 0 \
		--skip-phase3 --skip-phase4 \
		--selfplay-episodes 0 \
		--scratch \
		--batch 4 \
		--sims 4 \
		--k-steps 2 \
		--out $(SPATIAL_CKPT) \
		$(WEIGHTS_ARGS)

smoke-macro:
	$(PYTHON) -m pipelines.train_muzero \
		--arch macro \
		--phase1-steps 1 \
		--phase2-steps 0 \
		--phase3-epochs 0 \
		--phase4-steps 0 \
		--skip-phase2 --skip-phase3 --skip-phase4 \
		--scratch \
		--out $(MACRO_CKPT) \
		$(WEIGHTS_ARGS)

# ---------------------------------------------------------------------------
# Continuous spatial RL loop
# ---------------------------------------------------------------------------

loop-spatial:
	@echo "[make] COMPETITION spatial loop iterations=$(ITERATIONS) sims=$(SIMS) batch=$(BATCH) selfplay=$(SELFPLAY_EPISODES)"
	$(PYTHON) -m pipelines.phases.phase_spatial_loop \
		--iterations $(ITERATIONS) \
		--phase1-steps $(PHASE1_STEPS) \
		--phase2-steps $(PHASE2_STEPS) \
		--phase3-epochs $(PHASE4_STEPS) \
		--phase4-steps $(PHASE3_STEPS) \
		--selfplay-episodes $(SELFPLAY_EPISODES) \
		--batch $(BATCH) \
		--sims $(SIMS) \
		--lr $(LR) \
		--k-steps $(K_STEPS) \
		--buffer-max $(BUFFER_MAX) \
		--promote-every $(PROMOTE_EVERY) \
		$(LOOP_ARGS)

# ---------------------------------------------------------------------------
# Unified Hybrid Online/Offline Trainer & Hyperparameter Sweep (Spatial ADR 0001)
# ---------------------------------------------------------------------------

hybrid: gold
	$(PYTHON) -m pipelines.hybrid_trainer \
		--mode hybrid \
		--iterations $(HYBRID_ITERATIONS) \
		--base-iterations $(HYBRID_BASE_ITERS) \
		--selfplay-episodes $(HYBRID_SELFPLAY) \
		--eval-episodes $(HYBRID_EVAL_EPISODES) \
		--sweep-strategy $(HYBRID_STRATEGY) \
		$(HYBRID_ARGS)

sweep-offline: gold
	$(PYTHON) -m pipelines.hybrid_trainer \
		--mode offline \
		--iterations $(HYBRID_ITERATIONS) \
		--base-iterations $(HYBRID_BASE_ITERS) \
		--eval-episodes $(HYBRID_EVAL_EPISODES) \
		--sweep-strategy $(HYBRID_STRATEGY) \
		$(HYBRID_ARGS)

online-spatial: gold
	$(PYTHON) -m pipelines.hybrid_trainer \
		--mode online \
		--iterations $(HYBRID_ITERATIONS) \
		--base-iterations $(HYBRID_BASE_ITERS) \
		--selfplay-episodes $(HYBRID_SELFPLAY) \
		--eval-episodes $(HYBRID_EVAL_EPISODES) \
		$(HYBRID_ARGS)

# ---------------------------------------------------------------------------
# PPO tuning
# ---------------------------------------------------------------------------

ppo-grid:
	$(PYTHON) pipelines/ppo_tuning.py grid --trials $(PPO_TRIALS) $(PPO_ARGS)

ppo-train:
	$(PYTHON) pipelines/ppo_tuning.py train --rounds $(PPO_ROUNDS) $(PPO_ARGS)

# ---------------------------------------------------------------------------
# Package / evaluate
# ---------------------------------------------------------------------------

build: promote-weights
	$(PYTHON) scripts/build_head_submissions.py $(BUILD_ARGS)

test:
	$(PYTHON) -m pytest tests/ -q

benchmark:
	./train.sh benchmark $(BENCHMARK_ARGS)

champion:
	@STAMP=$$(date +%Y%m%d_%H%M%S); \
	echo "[make] Moving champion to $(BASELINES)/ (stamp: $$STAMP)"; \
	cp $(ARTIFACTS)/champion_submission.zip $(BASELINES)/submission_champion_$$STAMP.zip; \
	cp $(ARTIFACTS)/champion_main.py $(BASELINES)/champion_main_$$STAMP.py; \
	echo "[make] Champion archived: $(BASELINES)/submission_champion_$$STAMP.zip"

tournament:
	@echo "[make] Moving candidates into $(COMPETITION_DIR)/ (head-to-competition)"; \
	mkdir -p $(COMPETITION_DIR); \
	cp $(DIST)/champion_cand*.zip $(COMPETITION_DIR)/ 2>/dev/null || true; \
	cp $(ARTIFACTS)/candidate_submission.zip $(COMPETITION_DIR)/ 2>/dev/null || true; \
	cp $(ARTIFACTS)/champion_submission.zip $(COMPETITION_DIR)/ 2>/dev/null || true; \
	echo "[make] Running head-to-competition tournament..."; \
	$(PYTHON) scripts/tournament_candidates_vs_baselines.py \
		--candidates-dir $(COMPETITION_DIR) \
		--workers 8 \
		--games-per-matchup 2 \
		$(TOURNAMENT_ARGS)

tounament: tournament

# Round-robin all-vs-all (also persists LeagueTracker unless --no-tracker)
league league-rr:
	$(PYTHON) tournament.py league $(TOURNAMENT_ARGS)

# Prioritized Fictitious Self-Play league (hard/var + AlphaStar roles)
league-pfsp:
	$(PYTHON) tournament.py league-pfsp $(TOURNAMENT_ARGS)

# Sample PFSP matchmaking without playing env games
league-sample:
	$(PYTHON) tournament.py league-sample $(TOURNAMENT_ARGS)

# Print persisted LeagueTracker win matrices / leaderboard
league-status:
	$(PYTHON) tournament.py league-status $(TOURNAMENT_ARGS)

dashboard:
	$(PYTHON) scripts/render_tournament_dashboard.py $(TOURNAMENT_ARGS)

gate-buoy:
	$(PYTHON) scripts/gate_buoy.py $(GATE_ARGS)

# Full submission pipeline (spatial production):
#   distill → spatial weights → promote → head packages (with load validation)
submission: gold distill
	@echo ""
	@echo "================================================================================"
	@echo "  SUBMISSION BUILD PIPELINE (COMPETITION spatial Sampled MuZero)"
	@echo "  bc=$(PHASE1_STEPS) boot=$(PHASE2_STEPS) distill=$(PHASE3_STEPS) re=$(PHASE4_STEPS)"
	@echo "  selfplay=$(SELFPLAY_EPISODES) batch=$(BATCH) sims=$(SIMS) → $(SPATIAL_CKPT)"
	@echo "================================================================================"
	@echo "[make] Step 1/3: Spatial weight build..."
	$(MAKE) weights-spatial \
		PHASE1_STEPS=$(PHASE1_STEPS) \
		PHASE2_STEPS=$(PHASE2_STEPS) \
		PHASE3_STEPS=$(PHASE3_STEPS) \
		PHASE4_STEPS=$(PHASE4_STEPS) \
		SELFPLAY_EPISODES=$(SELFPLAY_EPISODES)
	@echo "[make] Step 2/3: Building head submission packages..."
	$(PYTHON) scripts/build_head_submissions.py $(BUILD_ARGS)
	@echo "[make] Step 3/3: Validating submission artifacts..."
	@$(PYTHON) -c "from muzero.spatial_checkpoint import assert_spatial_checkpoint_loadable; \
		import os,zipfile; \
		p='dist/head_muzero_submission.zip'; \
		assert os.path.isfile(p), p+' missing'; \
		z=zipfile.ZipFile(p); names=set(z.namelist()); \
		assert any(n=='main.py' or n.endswith('/main.py') for n in names), 'main.py missing'; \
		assert 'muzero/chassis.py' in names or any(n.endswith('chassis.py') for n in names), 'muzero package missing'; \
		assert_spatial_checkpoint_loadable('$(SPATIAL_CKPT)'); \
		print('  head_muzero_submission.zip OK ('+str(len(names))+' files); spatial ckpt loadable')"
	@echo "[make] === SUBMISSION BUILD COMPLETE ==="

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

diagnostics:
	@echo "[make] Running economic diagnostics..."
	$(PYTHON) scripts/economic_diagnostics.py $(DIAGNOSTICS_ARGS)

concurrency:
	@echo "[make] Running system concurrency benchmark..."
	$(PYTHON) scripts/system_concurrency_benchmark.py $(DIAGNOSTICS_ARGS)
