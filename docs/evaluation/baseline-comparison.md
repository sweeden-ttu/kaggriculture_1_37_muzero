# Baseline Comparison Strategy

## Two-Tier Validation System

The system uses complementary baseline sets for different validation purposes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BASELINE VALIDATION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

  CANDIDATE DISCOVERED (boost.sh / train.sh)
           │
           ▼
  ┌─────────────────────┐
  │  LOCAL TOP-8 GATE   │  ← Rapid validation (minutes)
  │  baselines/         │     4 games each
  │  8 candidate zips   │
  └──────────┬──────────┘
           │ Pass threshold?
           ▼
  ┌─────────────────────┐
  │  KAGGLE GOLD TEST   │  ← Certification (hours)
  │  /Volumes/BASELINES/│     8-16 games each
  │  5 scored zips      │
  └──────────┬──────────┘
           │
           ▼
  LEADERBOARD SUBMISSION
```

## Local Candidates (Rapid Gating)

**Location**: `baselines/` (this repo + external READ+WRITE)

| Candidate | Score | Iteration | Type |
|-----------|-------|-----------|------|
| `submission_train_cand042.zip` | $110,301.60 | 42 | Champion snapshot |
| `submission_train_cand048.zip` | $107,477.65 | 48 | Champion snapshot |
| `submission_train_cand056.zip` | $106,908.15 | 56 | Champion snapshot |
| `submission_train_cand012.zip` | $106,755.65 | 12 | Champion snapshot |
| `submission_boost_cand016.zip` | $108,092.65 | 16 | Champion snapshot |
| `submission_boost_cand003.zip` | $106,236.50 | 3 | Population pool |
| `submission_boost_cand014.zip` | $105,679.50 | 14 | Population pool |
| `submission_boost_cand034.zip` | $104,067.65 | 34 | Final champion |

**Evaluation Command**:
```bash
python tournament.py benchmark \
    --agent dist/main.py \
    --baselines baselines \
    --games-per-baseline 4
```

**Gate Threshold**: Must beat **majority** of local top-8 to proceed to gold test.

## Official Kaggle Baselines (Certification)

**Location**: `/Volumes/BASELINES/muzero/baselines/` (READONLY)

| File | Kaggle Score | Notes |
|------|--------------|-------|
| `submission_2033.zip` | 2033 | Highest scored |
| `submission_2028.zip` | 2028 | |
| `submission_2025.zip` | 2025 | |
| `submission_2021.zip` | 2021 | |
| `submission_0771.tar.gz` | 771 | Early baseline |
| `main.py` | — | Reference chassis |

**Evaluation Command**:
```bash
python tournament.py benchmark \
    --agent dist/main.py \
    --baselines /Volumes/BASELINES/muzero/baselines \
    --games-per-baseline 8
```

**Certification Threshold**: Must demonstrate **statistically significant** improvement over highest scored baseline (2033).

## Score Interpretation

### Kaggle Score vs Dollar Score

| Metric | Description |
|--------|-------------|
| **Kaggle Score** | Official leaderboard metric (2021-2033 range) |
| **Dollar Score** | Internal reward proxy ($100k-$150k range) |
| **Conversion** | Non-linear; Dollar score ≈ Kaggle Score × 50 + offset |

### Why Two Metrics?

- **Kaggle Score**: Official, public, sparse, delayed
- **Dollar Score**: Internal, dense, immediate, correlates with Kaggle

**Validation Principle**: Candidates optimized on Dollar Score must transfer to Kaggle Score gains.

## Comparison Matrix Generation

```python
# From evaluation/league.py:56-174 (run_round_robin_league)
# Produces full cross-product matrix

results = {
    "candidate_a": {
        "vs_candidate_b": {"win_rate": 0.62, "avg_score_a": 115000, "avg_score_b": 108000},
        "vs_baseline_2033": {"win_rate": 0.55, "avg_score_a": 112000, "avg_score_b": 105000},
        ...
    },
    ...
}
```

## Reporting

### Automated Report (from `evaluation/league.py:163-172`)

```json
{
  "timestamp": "2026-09-27T03:45:00Z",
  "total_games": 56,
  "elapsed_seconds": 342.5,
  "rankings": [
    {
      "rank": 1,
      "name": "submission_train_cand042",
      "points": 14.0,
      "wins": 7, "losses": 0, "ties": 0,
      "win_rate_pct": 100.0,
      "avg_score": 125430.0
    },
    ...
  ],
  "matches": [ ... ]
}
```

### Console Output
```
================================================================================
  ALL-VS-ALL ROUND-ROBIN LEAGUE (8 agents, 56 total games)
================================================================================
  Match  1/56: submission_train_cand042 vs submission_2033      → Winner: submission_train_cand042
  Match  2/56: submission_train_cand042 vs submission_boost_cand016 → Winner: submission_train_cand042
  ...

================================================================================
  FINAL LEAGUE RANKINGS
================================================================================
    Rank | Agent                          | Points | W-L-T    | Win %  | Avg Score
--------------------------------------------------------------------------------
       1 | submission_train_cand042       |   14.0 | 7-0-0    | 100.0% |  $125,430
       2 | submission_2033                |   11.5 | 5-1-1    |  78.6% |  $118,200
       ...
================================================================================
```

## Historical Progression Tracking

### Score Evolution
```
Iteration    Train Best      Boost Best       Kaggle Best
─────────────────────────────────────────────────────────
12           $106,755        —                —
16           —               $108,092         —
34           —               $104,067         —
42           $110,301        —                —
48           $107,477        —                —
56           $106,908        —                —
```

### Promotion History
Each champion promotion logged with:
- Timestamp
- Candidate path
- Eval results vs previous champion
- Eval results vs baselines
- Git commit hash

## Code References

- `evaluation/league.py:56-174` - Round-robin tournament
- `evaluation/league.py:181-227` - LeagueTracker (win-rate tracking)
- `tournament.py:1` - CLI benchmark
- `tournament_candidates_vs_baselines.py:1` - Cross-tournament
- `tournament_sub_vs_base.py:1` - Submission vs baselines
- `AGENTS.md:45-85` - Baselines comparison documentation