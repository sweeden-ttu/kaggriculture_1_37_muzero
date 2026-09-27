# Tournament Engine & Benchmarking

## Overview

The evaluation system runs head-to-head tournaments between agents, supporting both local candidate validation and Kaggle baseline certification.

## Tournament Engine

### Core Function (from `evaluation/league.py:56-174`)

```python
def run_round_robin_league(
    agent_paths: Sequence[str],
    *,
    games_per_pair: int = 2,
    base_seed: int = 10101,
    episode_steps: int = 720,
    save_replay_dir: Optional[str] = None,
    min_replay_score: float = 80000.0,
    quiet: bool = False,
) -> Dict[str, Any]:
```

### Execution Flow

```
1. Discover agent packages (zip files with main.py)
2. Generate all pair combinations (round-robin)
3. For each pair, play N games with alternating seats
4. Record: winner, scores, seed, game index
5. Aggregate: wins, losses, ties, points, avg score
6. Rank by points → avg score tiebreaker
7. Output leaderboard + match log
```

### Scoring System
- Win: +1.0 points
- Tie: +0.5 points
- Loss: +0.0 points
- Rank by: (points, avg_score)

### Seat Alternation
```python
# From evaluation/league.py:96-97
a_is_p0 = (g % 2 == 0)  # Alternate player 0 assignment
seed = base_seed + game_idx
```

## Benchmark Command

```bash
# Evaluate against local top-8 baselines
python tournament.py benchmark \
    --agent dist/main.py \
    --baselines baselines \
    --games-per-baseline 4

# Evaluate against Kaggle gold standard
python tournament.py benchmark \
    --agent dist/main.py \
    --baselines /Volumes/BASELINES/muzero/baselines \
    --games-per-baseline 4

# Cross-tournament: Official baselines vs Candidate league
python tournament_candidates_vs_baselines.py \
    --workers 8 \
    --games-per-matchup 2
```

## Tournament Output

### Leaderboard Format
```
  Rank | Agent                    | Points | W-L-T   | Win %  | Avg Score
──────┼──────────────────────────┼────────┼─────────┼────────┼───────────
   1  | submission_train_cand042 |  14.0  | 7-0-0   | 100.0% | $125,430
   2  | submission_2033          |  11.5  | 5-1-1   |  78.6% | $118,200
   ...
```

### Match Log (per game)
```json
{
  "game_idx": 1,
  "agent_a": "submission_train_cand042",
  "agent_b": "submission_2033",
  "score_a": 124500,
  "score_b": 112300,
  "winner": "A",
  "seed": 10102
}
```

## Baseline Comparison Strategy

### Two-Tier Validation

| Tier | Baselines | Purpose | Frequency |
|------|-----------|---------|-----------|
| **Local** | `baselines/` (top-8 candidates) | Rapid gating, iteration | Every eval |
| **Gold** | `/Volumes/BASELINES/muzero/baselines/` (Kaggle-scored) | Leaderboard certification | Periodic |

### Local Candidates (this repo)
| File | Score | Source |
|------|-------|--------|
| `submission_train_cand042.zip` | $110,301.60 | Iter 42 champion |
| `submission_train_cand048.zip` | $107,477.65 | Iter 48 champion |
| `submission_train_cand056.zip` | $106,908.15 | Iter 56 champion |
| `submission_train_cand012.zip` | $106,755.65 | Iter 12 champion |
| `submission_boost_cand016.zip` | $108,092.65 | Iter 16 champion |
| `submission_boost_cand003.zip` | $106,236.50 | Iter 3 pool |
| `submission_boost_cand014.zip` | $105,679.50 | Iter 14 pool |
| `submission_boost_cand034.zip` | $104,067.65 | Iter 34 champion |

### Official Kaggle Baselines (read-only)
| File | Kaggle Score |
|------|--------------|
| `submission_2033.zip` | 2033 |
| `submission_2028.zip` | 2028 |
| `submission_2025.zip` | 2025 |
| `submission_2021.zip` | 2021 |
| `submission_0771.tar.gz` | 771 |

## Replay Collection

```python
# From evaluation/league.py:56-174
save_replay_dir="replays/eval"
min_replay_score=80000.0  # Only save high-scoring games
```

High-scoring replays feed back into `learn_from_replay.py` for distillation.

## League Integration (PFSP)

### Opponent Selection During Evaluation
When evaluating in league context:
```python
# From evaluation/league.py:278-310
def select_matchmaking_opponent(agent_role, active_agent_id, pfsp):
    if agent_role == "main":
        # 35% self-play, 50% hard PFSP, 15% var PFSP
    elif agent_role == "main_exploiter":
        # 50% vs current main, 50% var PFSP past mains
    elif agent_role == "league_exploiter":
        # 100% hard PFSP (p=3.0)
```

### Win-Rate Tracking
```python
# From evaluation/league.py:181-227
LeagueTracker:
    checkpoints: List[str]
    wins_matrix: np.ndarray    # wins[i,j] = i beats j
    games_matrix: np.ndarray
    smoothing = 1.0  # Laplace

get_win_rate(agent, opponent) → (wins + 1) / (games + 2)
```

## Statistical Significance

### Minimum Games for Confidence
| Win Rate Gap | Games Needed (95% CI) |
|--------------|----------------------|
| 10% (55% vs 45%) | ~400 |
| 20% (60% vs 40%) | ~100 |
| 30% (65% vs 35%) | ~50 |

**Default**: 4 games per baseline (fast gating), 8-16 for certification.

## Code References

- `evaluation/league.py:56-174` - `run_round_robin_league`
- `evaluation/league.py:278-310` - `select_matchmaking_opponent`
- `evaluation/harness.py:1` - `play_single_game`
- `tournament.py:1` - CLI benchmark command
- `tournament_candidates_vs_baselines.py:1` - Cross-tournament
- `tournament_sub_vs_base.py:1` - Submission vs baselines
- `tournament_zip_round_robin.py:1` - Zip round-robin