# League Training & Prioritized Fictitious Self-Play (PFSP)

## Overview

The league system implements **AlphaStar-style PFSP** for diverse opponent sampling during self-play, creating an adaptive curriculum that targets agent weaknesses.

## League Architecture

### LeagueTracker (from `evaluation/league.py:181-227`)

```python
class LeagueTracker:
    checkpoints: List[str]              # Frozen model IDs
    wins_matrix: np.ndarray [N, N]      # wins[i,j] = i beats j
    games_matrix: np.ndarray [N, N]     # games played
    smoothing: float = 1.0              # Laplace prior

add_checkpoint(checkpoint_id)           # Expand matrices
record_match_outcome(a, b, outcome)     # outcome: 1.0/0.5/0.0
get_win_rate(a, b) → (wins+1)/(games+2) # Smoothed win rate
```

### PFSPSampler (from `evaluation/league.py:229-276`)

```python
class PFSPSampler:
    def sample_opponent(active_id, mode="hard", power=2.0):
        win_rates = [league.get_win_rate(active_id, c) for c in candidates]
        
        if mode == "hard":
            weights = (1 - win_rates) ** power      # Focus on difficult
        elif mode == "var":
            weights = win_rates * (1 - win_rates)   # Focus on ~50%
        elif mode == "uniform":
            weights = 1.0
        
        return weighted_random_choice(candidates, weights)
```

## PFSP Weighting Functions

| Mode | Formula | Focus | Use Case |
|------|---------|-------|----------|
| **hard** | (1 - x)^p | Low win rate opponents | Exploit weaknesses |
| **var** | x(1 - x) | Win rate ≈ 0.5 | Competitive matches |
| **uniform** | 1.0 | All equal | Exploration |

**Power p**: Higher = sharper focus (typically 2.0 or 3.0)

### Visualization
```
Win Rate (x) → Weight
     0.0  ──►  1.0 (hard, p=2)
     0.2  ──►  0.64
     0.5  ──►  0.25 (hard)  |  0.25 (var peak)
     0.8  ──►  0.04
     1.0  ──►  0.0
```

## Agent Roles & Matchmaking Ratios

(from `evaluation/league.py:278-310`)

### Main Agent (Primary Learner)
```
35% Self-Play
50% Hard PFSP (p=2.0)  ← Focus on losses
15% Var PFSP           ← Focus on ~50% matches
```

### Main Exploiter (Targets Current Main)
```
50% vs Current Main Agent (direct exploit)
50% Var PFSP vs Past Mains
```

### League Exploiter (Broad Weakness Hunter)
```
100% Hard PFSP across ALL league checkpoints (p=3.0)
```

### Code
```python
def select_matchmaking_opponent(agent_role, active_id, pfsp):
    rand = random()
    
    if agent_role == "main":
        if rand < 0.35:      return active_id, "self_play"
        elif rand < 0.85:    return pfsp.sample_opponent(active_id, "hard", 2.0)
        else:                return pfsp.sample_opponent(active_id, "var")
    
    elif agent_role == "main_exploiter":
        if rand < 0.50:      return "current_main_agent", "direct_exploit"
        else:                return pfsp.sample_opponent(active_id, "var")
    
    elif agent_role == "league_exploiter":
        return pfsp.sample_opponent(active_id, "hard", 3.0)
```

## Integration with Self-Play Loop

```python
# From pipelines/self_improving_loop.py
class SelfPlayManager:
    def __init__(self):
        self.league = LeagueTracker()
        self.pfsp = PFSPSampler(self.league)
        self.current_main = "champion_latest"
    
    def get_opponent_for_worker(self, worker_role):
        opponent, mode = select_matchmaking_opponent(
            worker_role, 
            self.current_main, 
            self.pfsp
        )
        return load_agent(opponent), mode
    
    def record_game_result(self, agent_id, opponent_id, outcome):
        self.league.record_match_outcome(agent_id, opponent_id, outcome)
        
        # Check for population collapse
        if outcome == 1.0 and self._is_exploiter(agent_id):
            if self._exploiter_win_rate(agent_id) > 0.70:
                self._reset_exploiter(agent_id)  # Force new strategy
```

## Population Collapse Prevention

### Detection
- Exploiter wins > 70% vs main agents
- League diversity (entropy of win-rate matrix) drops below threshold

### Remediation
```python
def _reset_exploiter(self, exploiter_id):
    # Re-initialize exploiter from random or champion + noise
    new_weights = self.current_main.clone()
    new_weights += torch.randn_like(new_weights) * 0.1
    save_checkpoint(new_weights, f"{exploiter_id}_reset")
    self.league.add_checkpoint(f"{exploiter_id}_reset")
```

## Laplace Smoothing

```python
# From evaluation/league.py:81-82
def get_win_rate(self, agent_id, opponent_id):
    games = self.games_matrix[i, j]
    wins = self.wins_matrix[i, j]
    return (wins + self.smoothing) / (games + 2.0 * self.smoothing)
```

**Effect**: 
- 0 games → 0.5 win rate (max uncertainty)
- Encourages exploration of new checkpoints
- Prevents premature convergence to "easy" opponents

## League Diversity Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Win-rate entropy** | -Σ p log p | High (diverse) |
| **Exploiter win rate** | vs main agents | < 70% |
| **Checkpoint count** | len(checkpoints) | 10-50 |
| **Match coverage** | played / possible | > 80% |

## Integration with Tournament Evaluation

```python
# When new champion promoted
def on_champion_promoted(new_champion_path):
    # 1. Add to league
    league.add_checkpoint(new_champion_path)
    
    # 2. Run calibration matches vs existing league
    for existing in league.checkpoints[:-1]:
        for _ in range(4):  # 4 games each
            result = play_match(new_champion_path, existing)
            league.record_match_outcome(new_champion_path, existing, result)
    
    # 3. Update current_main
    self.current_main = new_champion_path
```

## Code References

- `evaluation/league.py:181-227` - `LeagueTracker`
- `evaluation/league.py:229-276` - `PFSPSampler`
- `evaluation/league.py:278-310` - `select_matchmaking_opponent`
- `pipelines/self_improving_loop.py:1` - Integration point
- `docs/League_Tracker.md:1` - Full theoretical documentation