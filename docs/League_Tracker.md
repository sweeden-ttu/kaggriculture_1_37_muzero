# League Tracker

**Prioritized Fictitious Self-Play (PFSP)** is a multi-agent matchmaking algorithm introduced in AlphaStar to train agents against a diverse league of past policies. 

Standard fictitious play samples opponents uniformly from the league. PFSP instead prioritizes opponent policies based on the learning agent's **win rate** against them. This creates a focused, adaptive curriculum that targets an agent's specific tactical weaknesses or matches it against evenly paired rivals.

---

### 1. Mathematical Formulation

Given an active learning agent policy \\(\pi'\\) and a pool of historical league opponent policies \\(\{\pi_1, \pi_2, \dots, \pi_N\}\\):

1. Compute the empirical win probability \\(x_i = \text{WinRate}(\pi' \text{ vs } \pi_i) \in\\).
2. Evaluate a weighting function \\(f(x_i)\\) over the win rate:
   * **Hard Opponents (\\(f_{\text{hard}}\\))**: Focuses on opponents that the agent struggles against (low win rate \\(x\\)):
     \\[f_{\text{hard}}(x) = (1 - x)^p \quad (p \ge 1)\\]
   * **Matched Opponents (\\(f_{\text{var}}\\))**: Focuses on opponents with competitive, high-variance games (\\(x \approx 0.5\\)):
     \\[f_{\text{var}}(x) = x(1 - x)\\]
3. Normalize weights into a multinomial sampling distribution:
   \\[P(\text{Match } \pi_i) = \frac{f(x_i)}{\sum_{j=1}^N f(x_j)}\\]

---

### 2. Python / PyTorch Implementation

Below is a modular, standalone implementation of the `PFSPSampler` and `LeagueTracker` class ready for integration into a training pipeline:

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

class LeagueTracker:
    """
    Tracks policy checkpoints and empirical win rates between league members.
    """
    def __init__(self, smoothing: float = 1.0):
        self.checkpoints: List[str] = []  # List of checkpoint IDs
        # Store win/loss/draw tally: wins_matrix[i, j] = wins of policy i against policy j
        self.wins_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.games_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.smoothing = smoothing  # Laplace smoothing to prevent division by zero

    def add_checkpoint(self, checkpoint_id: str):
        """Registers a new frozen model checkpoint into the league."""
        self.checkpoints.append(checkpoint_id)
        n = len(self.checkpoints)
        
        # Expand matrices
        new_wins = np.zeros((n, n), dtype=np.float32)
        new_games = np.zeros((n, n), dtype=np.float32)
        
        if n > 1:
            new_wins[:n-1, :n-1] = self.wins_matrix
            new_games[:n-1, :n-1] = self.games_matrix
            
        self.wins_matrix = new_wins
        self.games_matrix = new_games

    def record_match_outcome(self, agent_id: str, opponent_id: str, outcome: float):
        """
        Records game result: outcome = 1.0 (win), 0.5 (draw), 0.0 (loss)
        """
        i = self.checkpoints.index(agent_id)
        j = self.checkpoints.index(opponent_id)
        
        self.wins_matrix[i, j] += outcome
        self.wins_matrix[j, i] += (1.0 - outcome)
        self.games_matrix[i, j] += 1.0
        self.games_matrix[j, i] += 1.0

    def get_win_rate(self, agent_id: str, opponent_id: str) -> float:
        """Computes empirical win rate with Laplace smoothing."""
        i = self.checkpoints.index(agent_id)
        j = self.checkpoints.index(opponent_id)
        
        games = self.games_matrix[i, j]
        wins = self.wins_matrix[i, j]
        
        # Laplace-smoothed win rate estimate
        return (wins + self.smoothing) / (games + 2.0 * self.smoothing)

class PFSPSampler:
    """
    Prioritized Fictitious Self-Play Sampler.
    Selects opponent checkpoints from the league based on priority weighting functions.
    """
    def __init__(self, league: LeagueTracker):
        self.league = league

    def sample_opponent(
        self, 
        active_agent_id: str, 
        mode: str = "hard", 
        power: float = 2.0
    ) -> str:
        """
        Samples an opponent checkpoint for active_agent_id.
        
        Args:
            active_agent_id: ID of the currently learning model.
            mode: "hard" (focus on losses), "var" (focus on ~50% win rates), or "uniform".
            power: Exponent p for f_hard(x) = (1 - x)^p.
        """
        checkpoints = self.league.checkpoints
        if not checkpoints or (len(checkpoints) == 1 and checkpoints == active_agent_id):
            return active_agent_id  # Fallback to self-play if league is empty

        # Filter candidates (excluding self if desired)
        candidates = [c for c in checkpoints if c != active_agent_id]
        if not candidates:
            return active_agent_id

        win_rates = np.array([
            self.league.get_win_rate(active_agent_id, c) for c in candidates
        ], dtype=np.float32)

        # Apply weighting function f(x)
        if mode == "hard":
            # f_hard(x) = (1 - x)^p : high weight for low win rates
            weights = np.power(np.maximum(0.0, 1.0 - win_rates), power)
        elif mode == "var":
            # f_var(x) = x * (1 - x) : high weight for win rates near 0.5
            weights = win_rates * (1.0 - win_rates)
        elif mode == "uniform":
            weights = np.ones_like(win_rates)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Avoid all-zero weights
        total_weight = np.sum(weights)
        if total_weight <= 1e-8:
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / total_weight

        # Sample opponent index
        sampled_idx = np.random.choice(len(candidates), p=probs)
        return candidates[sampled_idx]
```

---

### 3. Integrating PFSP into League Agent Roles

Following the AlphaStar league architecture, different agent roles sample opponents using specific PFSP modes:

```python
def select_matchmaking_opponent(
    agent_role: str, 
    active_agent_id: str, 
    pfsp: PFSPSampler
) -> Tuple[str, str]:
    """
    Applies role-specific matchmaking ratios from AlphaStar league rules.
    """
    rand = np.random.rand()

    if agent_role == "main":
        # Main Agent: 35% Self-Play, 50% Hard PFSP, 15% Main Exploiter PFSP
        if rand < 0.35:
            return active_agent_id, "self_play"
        elif rand < 0.85:
            opponent = pfsp.sample_opponent(active_agent_id, mode="hard", power=2.0)
            return opponent, "pfsp_hard"
        else:
            opponent = pfsp.sample_opponent(active_agent_id, mode="var")
            return opponent, "pfsp_var"

    elif agent_role == "main_exploiter":
        # Main Exploiter: 50% vs Current Main, 50% vs Past Mains via f_var
        if rand < 0.50:
            return "current_main_agent", "direct_exploit"
        else:
            opponent = pfsp.sample_opponent(active_agent_id, mode="var")
            return opponent, "pfsp_var"

    elif agent_role == "league_exploiter":
        # League Exploiter: 100% PFSP across all league checkpoints using f_hard
        opponent = pfsp.sample_opponent(active_agent_id, mode="hard", power=3.0)
        return opponent, "pfsp_hard"
    
    return active_agent_id, "fallback"
```

---

### Key Theoretical & Practical Tips
1. **Laplace Smoothing**: Initialize unplayed matchups with pseudo-counts (e.g., \\(1\text{ win} / 2\text{ games} = 0.5\\)) so newly added checkpoints are explored before priorities collapse.
2. **Temperature / Power Tuning (\\(p\\))**: Setting \\(p=2.0\\) or \\(p=3.0\\) in \\(f_{\text{hard}}(x) = (1-x)^p\\) sharpens search priorities onto strong counter-strategies, preventing agents from wasting compute against easily defeated historical checkpoints.
3. **Preventing Population Collapse**: Resetting exploiter parameters periodically (e.g., when defeating main agents in \\(\ge 70\%\\) of matches) forces the discovery of genuinely new counter-strategies rather than over-tuning old ones.