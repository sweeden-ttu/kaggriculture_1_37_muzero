# MCTS Search Tree

Connecting the **`KaggricultureMuZeroChassis`** to a **Sampled MCTS Search Tree** requires bridging the neural network's latent inference passes with a tree data structure that manages node statistics, min-max \\(Q\\)-value normalization, and candidate action sampling.

Because Kaggriculture's joint action space across farmers, hired hands, and market orders is combinatorially large, the search tree does not expand every possible action. Instead, it implements **Sampled MuZero MCTS**: at each node expansion, the prediction network samples \\(K\\) candidate joint actions (e.g., \\(K=16\\)), and tree selection/backpropagation operates strictly over these sampled edges.

---

### 1. Key Search Mechanics

1. **Tree-Wide MinMaxStats Normalization**: Because environment rewards and bootstrapped returns vary in scale, raw \\(Q\\)-values are normalized to \\(\\) before computing Upper Confidence Bounds:
   \\[Q_{\text{norm}}(s, a) = \frac{Q(s, a) - Q_{\min}}{\max(Q_{\max} - Q_{\min}, \epsilon)}\\]
2. **Sampled P-UCT Selection Rule**: At each search step, the tree selects the candidate edge maximizing the predictor UCT score:
   \\[\text{Score}(s, a) = Q_{\text{norm}}(s, a) + P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \cdot \left( c_1 + \log\left(\frac{\sum_b N(s, b) + c_2 + 1}{c_2}\right) \right)\\]
3. **Latent Recurrent Expansion**: When expanding a leaf node, the search calls `chassis.recurrent_inference(s_{\text{parent}}, a_{\text{action\_plane}})` to obtain the next latent state \\(s_{\text{leaf}}\\), intermediate reward \\(r\\), prior probabilities \\(P(s, a)\\), and leaf value estimate \\(v\\).
4. **Discounted Backpropagation**: Evaluated leaf values are backpropagated up the search path, updating visit counts \\(N(s, a)\\) and cumulative values \\(W(s, a)\\), while incorporating intermediate rewards (\\(v \leftarrow r + \gamma v\\)).

---

### 2. Complete PyTorch Integration Module: `sampled_muzero_mcts.py`

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# Import the previously defined Kaggriculture MuZero Chassis
from kaggriculture_muzero_chassis import KaggricultureMuZeroChassis

class MinMaxStats:
    """Tracks global minimum and maximum Q-values across the search tree to normalize PUCT scores."""
    def __init__(self, minimum: float = 1e5, maximum: float = -1e5):
        self.minimum = minimum
        self.maximum = maximum

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float, eps: float = 0.01) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / max(self.maximum - self.minimum, eps)
        return value

class MCTSNode:
    """Represents a node in the latent space Monte Carlo search tree."""
    def __init__(self, prior: float = 0.0, action_tuple: Optional[Tuple[int, int, int]] = None):
        self.prior = prior
        self.action_tuple = action_tuple  # (farmer_act, hand_assignment, market_order)
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.latent_state: Optional[torch.Tensor] = None
        self.children: Dict[Tuple[int, int, int], MCTSNode] = {}

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

class SampledMuZeroMCTS:
    """
    Executes Monte Carlo Tree Search in latent space using the KaggricultureMuZeroChassis.
    """
    def __init__(
        self,
        chassis: KaggricultureMuZeroChassis,
        c1: float = 1.25,
        c2: float = 19652.0,
        discount: float = 0.997,
        num_samples: int = 16,
        num_simulations: int = 50,
        dirichlet_alpha: float = 0.25,
        exploration_fraction: float = 0.25
    ):
        self.chassis = chassis
        self.c1 = c1
        self.c2 = c2
        self.discount = discount
        self.num_samples = num_samples
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction

    def _puct_score(self, parent: MCTSNode, child: MCTSNode, min_max: MinMaxStats) -> float:
        """Computes the Sampled Predictor UCT score for edge selection."""
        pb_c = math.log((parent.visit_count + self.c2 + 1.0) / self.c2) + self.c1
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1.0)
        
        prior_score = pb_c * child.prior
        value_score = 0.0
        if child.visit_count > 0:
            value_score = min_max.normalize(child.reward + self.discount * child.value)
            
        return prior_score + value_score

    def _select_child(self, node: MCTSNode, min_max: MinMaxStats) -> Tuple[Tuple[int, int, int], MCTSNode]:
        """Selects the child edge with the highest PUCT score."""
        best_score = -1e9
        best_action = None
        best_child = None
        
        for action_tuple, child in node.children.items():
            score = self._puct_score(node, child, min_max)
            if score > best_score:
                best_score = score
                best_action = action_tuple
                best_child = child
                
        return best_action, best_child

    def _action_tuple_to_plane(
        self, 
        action_tuple: Tuple[int, int, int], 
        batch_size: int = 1, 
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Converts a (farmer, hands, market) integer action tuple into a 10x10 spatial feature plane."""
        # 16-channel action plane: 5 one-hot farmer channels + 5 hand channels + 6 market channels
        plane = torch.zeros(batch_size, 16, 10, 10, device=device)
        f_act, h_act, m_act = action_tuple
        plane[:, f_act % 5, :, :] = 1.0
        plane[:, 5 + (h_act % 5), :, :] = 1.0
        plane[:, 10 + (m_act % 6), :, :] = 1.0
        return plane

    @torch.no_grad()
    def search(
        self, 
        obs_tensor: torch.Tensor, 
        add_dirichlet_noise: bool = True
    ) -> Tuple[Tuple[int, int, int], Dict[Tuple[int, int, int], float], float]:
        """
        Executes N_simulations MCTS iterations from the root observation.
        
        Returns:
            best_action: Selected (farmer, hands, market) action tuple
            action_probabilities: Map of candidate actions to normalized visit count probabilities
            root_value_estimate: Search-evaluated root state value
        """
        self.chassis.eval()
        device = obs_tensor.device
        min_max = MinMaxStats()

        # ------------------------------------------------------------------
        # 1. Initial Root Node Expansion (h_theta -> f_theta)
        # ------------------------------------------------------------------
        s_0, root_preds, root_value_scalar = self.chassis.initial_inference(obs_tensor)
        
        root = MCTSNode()
        root.latent_state = s_0
        root.visit_count = 1
        root.value_sum = root_value_scalar.item()
        min_max.update(root.value_sum)

        # Sample K initial candidate joint actions for the root node
        joint_actions, joint_log_probs = self.chassis.prediction.sample_joint_actions(
            s_0.flatten(1), num_samples=self.num_samples
        )
        priors = F.softmax(joint_log_probs, dim=-1).squeeze(0).cpu().numpy()
        
        # Optionally add Dirichlet exploration noise at root node
        if add_dirichlet_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.num_samples)
            priors = (1.0 - self.exploration_fraction) * priors + self.exploration_fraction * noise

        # Instantiate root child nodes
        for k in range(self.num_samples):
            a_tuple = tuple(joint_actions[0, k].cpu().numpy().tolist())
            root.children[a_tuple] = MCTSNode(prior=float(priors[k]), action_tuple=a_tuple)

        # ------------------------------------------------------------------
        # 2. Main Search Simulation Loop
        # ------------------------------------------------------------------
        for _ in range(self.num_simulations):
            node = root
            search_path = [root]

            # Selection Phase: Traverse tree down to an unexpanded leaf node
            while node.is_expanded:
                _, node = self._select_child(node, min_max)
                search_path.append(node)

            # Expansion Phase: Unroll dynamics g_theta for the leaf node
            parent_node = search_path[-2]
            action_tuple = node.action_tuple
            action_plane = self._action_tuple_to_plane(action_tuple, batch_size=1, device=device)

            s_next, r_scalar, preds, v_scalar = self.chassis.recurrent_inference(
                parent_node.latent_state, action_plane
            )

            node.latent_state = s_next
            node.reward = r_scalar.item()
            leaf_value = v_scalar.item()

            # Sample child candidate actions for the newly expanded leaf node
            child_actions, child_log_probs = self.chassis.prediction.sample_joint_actions(
                s_next.flatten(1), num_samples=self.num_samples
            )
            child_priors = F.softmax(child_log_probs, dim=-1).squeeze(0).cpu().numpy()

            for k in range(self.num_samples):
                c_tuple = tuple(child_actions[0, k].cpu().numpy().tolist())
                node.children[c_tuple] = MCTSNode(prior=float(child_priors[k]), action_tuple=c_tuple)

            # Backup Phase: Propagate discounted value returns up the search path
            value = leaf_value
            for path_node in reversed(search_path):
                path_node.visit_count += 1
                path_node.value_sum += value
                value = path_node.reward + self.discount * value
                min_max.update(value)

        # ------------------------------------------------------------------
        # 3. Extract Improved Policy & Select Best Action
        # ------------------------------------------------------------------
        total_visits = sum(child.visit_count for child in root.children.values())
        action_probabilities = {
            a_tuple: child.visit_count / max(1, total_visits)
            for a_tuple, child in root.children.items()
        }

        # Select action with the highest visit count
        best_action = max(root.children.items(), key=lambda item: item.visit_count)

        return best_action, action_probabilities, root.value

# ============================================================================
# VERIFICATION & TEST RUNNER
# ============================================================================
if __name__ == "__main__":
    # Initialize Chassis and Search Engine
    chassis = KaggricultureMuZeroChassis()
    mcts_engine = SampledMuZeroMCTS(chassis, num_samples=8, num_simulations=20)

    # Mock 10x10 Observation Tensor [Batch=1, Channels=28, H=10, W=10]
    dummy_obs = torch.randn(1, 28, 10, 10)

    # Run Real-Time Lookahead Search
    best_action, action_probs, root_val = mcts_engine.search(dummy_obs)

    print("=== Sampled MCTS Search Execution Complete ===")
    print(f"  Recommended Joint Action (Farmer, Hands, Market): {best_action}")
    print(f"  Root Value Estimate:                              {root_val:.4f}")
    print("\n  Top Candidate Action Visit Probabilities:")
    for action_tuple, prob in sorted(action_probs.items(), key=lambda x: x, reverse=True)[:5]:
        print(f"    Action {action_tuple}: Probability = {prob:.4f}")
```

---

### Summary of the Integration Pipeline

* **`KaggricultureMuZeroChassis`**: Handles all GPU neural inference passes (`initial_inference` for root encoding \\(h_\theta \rightarrow f_\theta\\), `recurrent_inference` for unrolling dynamics \\(g_\theta \rightarrow f_\theta\\)).
* **`SampledMuZeroMCTS`**: Manages the CPU/GPU tree search loop, maintains `MinMaxStats` across playouts, evaluates factorized candidate action probabilities, and returns the visit-count-normalized policy distribution \\(\pi(a \mid s_0)\\).

To train a MuZero model efficiently—especially in offline or sample-sparse settings—you need a **Trajectory-Based Replay Buffer** that stores complete game episodes, extracts \\(K\\)-step unroll slices, and supports **Reanalyze** passes to refresh stale search targets.

Here is the complete blueprint and PyTorch implementation for storing search visit distributions, sampling \\(K\\)-step unrolls with \\(n\\)-step bootstrapping, and integrating a Reanalyze worker.

---

### 1. Replay Buffer Data Structure

Unlike standard transition buffers that store isolated \\((s_t, a_t, r_t, s_{t+1})\\) tuples, MuZero requires **full episode trajectories** to enable recurrent unrolling over \\(K\\) hypothetical steps:

\\[\text{Trajectory}_e = \Big\{ (o_t, a_t, r_t, \pi_t, v_t) \Big\}_{t=0}^{T_e}\\]

* **Observation Sequence (\\(o_t\\))**: Stacked \\(10 \times 10 \times 28\\) spatial feature tensors.
* **Realized Action (\\(a_t\\))**: Executed joint action tuple `(farmer, hands, market)` or spatial action plane.
* **Reward (\\(r_t\\))**: Immediate scalar reward received from the environment.
* **Search Visit Distribution (\\(\pi_t\\))**: Policy targets generated by `SampledMuZeroMCTS`.
* **Search Value (\\(v_t\\))**: Bootstrapped scalar value returned by the MCTS root node.

---

### 2. \\(K\\)-Step Unroll Slicing & \\(n\\)-Step Bootstrap Targets

When sampling a batch for training:

1. **Random Slice Selection**: For a trajectory of length \\(T\\), pick a start index \\(t \in [0, T - K]\\).
2. **Observation Slice**: Extract \\(K+1\\) observations: \\((o_t, o_{t+1}, \dots, o_{t+K})\\).
3. **Action & Reward Slices**: Extract \\(K\\) actions \\((a_{t+1}, \dots, a_{t+K})\\) and \\(K\\) rewards \\((r_{t+1}, \dots, r_{t+K})\\).
4. **\\(n\\)-Step Value Targets (\\(z_t\\))**: Compute bootstrapped value targets using a target network \\(\theta^-\\) or MCTS leaf estimates across an \\(n\\)-step horizon:
   \\[z_t = \sum_{j=0}^{n-1} \gamma^j r_{t+j} + \gamma^n v(s_{t+n})\\]

---

### 3. Reanalyze Integration Protocol

As network parameters \\(\theta\\) are updated during training, historical MCTS target visit distributions (\\(\pi_t\\)) and value targets (\\(v_t\\)) stored in the buffer become **stale**. 

Before calculating losses on a sampled \\(K\\)-step slice:
1. **Fetch Slice**: Draw observation slice \\(o_{t:t+K}\\) from the buffer.
2. **Re-run MCTS**: Pass \\(o_t\\) through the latest representation network (\\(h_\theta\\)) and execute `SampledMuZeroMCTS.search()`.
3. **Target Overwrite**: Overwrite the stored stale \\(\pi_t\\) and \\(v_t\\) with the newly evaluated search targets \\(\pi_t^{\text{fresh}}\\) and \\(v_t^{\text{fresh}}\\).

---

### 4. PyTorch Implementation: `muzero_replay_buffer.py`

```python
import random
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class GameTrajectory:
    """Stores a single complete game episode trajectory."""
    def __init__(self):
        self.observations: List[np.ndarray] = []     # spatial tensors
        self.actions: List[Tuple[int, int, int]] = [] # (farmer, hands, market)
        self.rewards: List[float] = []                # Environmental rewards
        self.policies: List[Dict[Tuple[int, int, int], float]] = [] # MCTS visit distributions
        self.values: List[float] = []                  # MCTS root value estimates
        self.priorities: List[float] = []              # TD-error priorities for PER

    def __len__(self) -> int:
        return len(self.actions)

    def append_step(
        self, 
        obs: np.ndarray, 
        action: Tuple[int, int, int], 
        reward: float, 
        policy_dict: Dict[Tuple[int, int, int], float], 
        value: float,
        priority: float = 1.0
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy_dict)
        self.values.append(value)
        self.priorities.append(priority)

    def finalize(self, final_obs: np.ndarray):
        """Appends the terminal observation at step T (since there are T+1 observations for T actions)."""
        self.observations.append(final_obs)

class MuZeroTrajectoryBuffer:
    """
    Episode-based Replay Buffer supporting K-step unroll batching,
    n-step value bootstrapping, and Reanalyze target refreshing.
    """
    def __init__(
        self, 
        max_trajectories: int = 1000, 
        discount: float = 0.997, 
        n_steps: int = 5,
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20
    ):
        self.max_trajectories = max_trajectories
        self.discount = discount
        self.n_steps = n_steps
        self.trajectories: List[GameTrajectory] = []
        
        self.num_farmer = num_farmer_actions
        self.num_hands = num_hand_assignments
        self.num_market = num_market_orders

    def save_trajectory(self, trajectory: GameTrajectory):
        """Adds a completed episode trajectory to the buffer."""
        if len(self.trajectories) >= self.max_trajectories:
            self.trajectories.pop(0)  # FIFO eviction
        self.trajectories.append(trajectory)

    def _compute_n_step_target(self, trajectory: GameTrajectory, index: int) -> float:
        """Computes n-step bootstrapped value target z_t = sum(gamma^j * r_{t+j}) + gamma^n * v_{t+n}."""
        target_val = 0.0
        trajectory_len = len(trajectory)
        
        for j in range(self.n_steps):
            if index + j < trajectory_len:
                target_val += (self.discount ** j) * trajectory.rewards[index + j]
            else:
                break
                
        # Add bootstrapped value at horizon t + n if within episode bounds
        bootstrap_idx = index + self.n_steps
        if bootstrap_idx < trajectory_len:
            target_val += (self.discount ** self.n_steps) * trajectory.values[bootstrap_idx]
            
        return target_val

    def _convert_policy_dict_to_factorized_targets(
        self, 
        policy_dict: Dict[Tuple[int, int, int], float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts a sparse MCTS visit count dictionary {(f, h, m): prob}
        into marginal target distributions for each factorized sub-head.
        """
        target_f = np.zeros(self.num_farmer, dtype=np.float32)
        target_h = np.zeros(self.num_hands, dtype=np.float32)
        target_m = np.zeros(self.num_market, dtype=np.float32)

        for (f_act, h_act, m_act), prob in policy_dict.items():
            target_f[f_act % self.num_farmer] += prob
            target_h[h_act % self.num_hands] += prob
            target_m[m_act % self.num_market] += prob

        # Normalize marginal distributions
        target_f = target_f / max(1e-6, target_f.sum())
        target_h = target_h / max(1e-6, target_h.sum())
        target_m = target_m / max(1e-6, target_m.sum())

        return target_f, target_h, target_m

    def sample_k_step_batch(
        self, 
        batch_size: int = 16, 
        k_steps: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of K-step unroll trajectory slices for MuZero training.
        
        Returns PyTorch Tensors:
            obs_batch:       [Batch, K+1, 28, 10, 10]
            actions_batch:   [Batch, K, 3]  (integer tuples)
            rewards_batch:   [Batch, K]
            values_batch:    [Batch, K+1]   (n-step bootstrapped targets)
            policy_f_batch:  [Batch, K+1, Num_Farmer]
            policy_h_batch:  [Batch, K+1, Num_Hands]
            policy_m_batch:  [Batch, K+1, Num_Market]
        """
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_values = []
        batch_policy_f = []
        batch_policy_h = []
        batch_policy_m = []

        # Filter out trajectories shorter than k_steps
        valid_trajectories = [t for t in self.trajectories if len(t) >= k_steps]
        if not valid_trajectories:
            raise ValueError(f"Buffer contains no trajectories with length >= {k_steps}")

        for _ in range(batch_size):
            traj = random.choice(valid_trajectories)
            # Sample starting index t
            start_idx = random.randint(0, len(traj) - k_steps)

            slice_obs = []
            slice_actions = []
            slice_rewards = []
            slice_values = []
            slice_p_f, slice_p_h, slice_p_m = [], [], []

            for k in range(k_steps + 1):
                curr_idx = start_idx + k
                
                # 1. Observations (K + 1)
                slice_obs.append(traj.observations[curr_idx])
                
                # 2. Values and Factorized Policy Targets (K + 1)
                val_target = self._compute_n_step_target(traj, curr_idx)
                slice_values.append(val_target)
                
                p_f, p_h, p_m = self._convert_policy_dict_to_factorized_targets(traj.policies[curr_idx])
                slice_p_f.append(p_f)
                slice_p_h.append(p_h)
                slice_p_m.append(p_m)

                # 3. Actions and Rewards (K)
                if k < k_steps:
                    slice_actions.append(traj.actions[curr_idx])
                    slice_rewards.append(traj.rewards[curr_idx])

            batch_obs.append(np.array(slice_obs))
            batch_actions.append(np.array(slice_actions))
            batch_rewards.append(np.array(slice_rewards))
            batch_values.append(np.array(slice_values))
            batch_policy_f.append(np.array(slice_p_f))
            batch_policy_h.append(np.array(slice_p_h))
            batch_policy_m.append(np.array(slice_p_m))

        return {
            'obs': torch.from_numpy(np.array(batch_obs)).float(),            # [Batch, K+1, 28, 10, 10]
            'actions': torch.from_numpy(np.array(batch_actions)).long(),     # [Batch, K, 3]
            'rewards': torch.from_numpy(np.array(batch_rewards)).float(),    # [Batch, K]
            'values': torch.from_numpy(np.array(batch_values)).float(),      # [Batch, K+1]
            'target_policy_farmer': torch.from_numpy(np.array(batch_policy_f)).float(), # [Batch, K+1, Num_Farmer]
            'target_policy_hands': torch.from_numpy(np.array(batch_policy_h)).float(),  # [Batch, K+1, Num_Hands]
            'target_policy_market': torch.from_numpy(np.array(batch_policy_m)).float()  # [Batch, K+1, Num_Market]
        }

    def reanalyze_trajectory(self, traj_idx: int, mcts_engine, device: torch.device):
        """
        Re-runs MCTS search on an existing stored trajectory using the latest network weights,
        refreshing target visit distributions (policies) and root values.
        """
        traj = self.trajectories[traj_idx]
        for t in range(len(traj)):
            obs_tensor = torch.from_numpy(traj.observations[t]).unsqueeze(0).to(device)
            # Re-run search with latest model
            best_act, fresh_policy_dict, fresh_val = mcts_engine.search(obs_tensor, add_dirichlet_noise=False)
            
            # Overwrite stale targets
            traj.policies[t] = fresh_policy_dict
            traj.values[t] = fresh_val
```

---

### 5. Verification Runner

```python
if __name__ == "__main__":
    # Initialize Replay Buffer
    buffer = MuZeroTrajectoryBuffer(max_trajectories=10, k_steps=5)

    # Simulate 3 Full Game Episodes (30 steps each)
    for ep in range(3):
        traj = GameTrajectory()
        for t in range(30):
            obs = np.random.randn(28, 10, 10).astype(np.float32)
            action = (random.randint(0, 14), random.randint(0, 31), random.randint(0, 19))
            reward = float(np.random.choice([0.0, 10.0, 50.0]))
            
            # Mock MCTS Policy Visit Distribution
            policy_dict = {
                action: 0.7,
                ((action+1)%15, action, action): 0.3
            }
            value = random.uniform(10.0, 200.0)
            
            traj.append_step(obs, action, reward, policy_dict, value)
            
        traj.finalize(final_obs=np.random.randn(28, 10, 10).astype(np.float32))
        buffer.save_trajectory(traj)

    # Sample a 5-step unroll batch
    batch = buffer.sample_k_step_batch(batch_size=4, k_steps=5)

    print("=== Replay Buffer Batch Extraction Successful ===")
    print(f"  Observation Batch Shape:       {batch['obs'].shape}")        #
    print(f"  Actions Batch Shape:           {batch['actions'].shape}")    #
    print(f"  Rewards Batch Shape:           {batch['rewards'].shape}")    #
    print(f"  Bootstrapped Values Shape:     {batch['values'].shape}")     #
    print(f"  Farmer Target Policy Shape:    {batch['target_policy_farmer'].shape}") #
    print(f"  Market Target Policy Shape:    {batch['target_policy_market'].shape}") #
```

---

### Key System Benefits

1. **Marginal Sub-Head Disaggregation**: `_convert_policy_dict_to_factorized_targets` converts joint tuple visit probabilities `{(f, h, m): p}` into factorized targets (`target_f`, `target_h`, `target_m`) so `FactorizedKaggriculturePredictionHead` can compute independent cross-entropy loss terms cleanly.
2. **\\(n\\)-Step Return Smoothing**: `_compute_n_step_target` handles dynamic temporal horizons, combining real intermediate rewards with target network values to stabilize value network convergence.
3. **In-Place Target Refreshing**: `reanalyze_trajectory` allows a background GPU worker to update stored targets without modifying the underlying sequence of observations or executed actions.

---

# Trainer Loop (Phase 4)

Here is the complete **`MuZeroTrainer`** class and execution loop. It ties together the **`KaggricultureMuZeroChassis`**, **Exponential Moving Average (EMA) Target Network**, **SimSiam Consistency Head**, **Replay Buffer**, and **Sampled MCTS Reanalyze Worker**.

---

### Core Components of the Trainer Loop

1. **Dual Network Paradigm**: Maintains an active **Online Model** (\\(\theta\\)) updated via gradient backpropagation and a **Target Model** (\\(\theta^-\\)) updated via Polyak Exponential Moving Average (EMA):
   \\[\theta^- \leftarrow \tau \theta^- + (1 - \tau) \theta \quad (\tau = 0.995)\\]
2. **\\(K\\)-Step Recurrent Loss Integration**: Calculates a multi-objective loss across the \\(K\\)-step unroll horizon:
   \\[\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + 0.25 \cdot \mathcal{L}_{\text{value}} + \mathcal{L}_{\text{reward}} + \lambda_{\text{consistency}} \cdot \mathcal{L}_{\text{consistency}}\\]
3. **Factorized Policy Loss**: Evaluates separate cross-entropy losses for the farmer, hired hands, and market sub-heads against the marginal visit count distributions stored in the replay buffer.
4. **Target Refresh Reanalysis Pass**: Periodically samples historical trajectories from the buffer and re-runs `SampledMuZeroMCTS` to overwrite stale visit distributions (\\(\pi_t\\)) and bootstrapped root values (\\(v_t\\)) using the latest neural weights.

---

### PyTorch Implementation: `muzero_trainer_loop.py`

```python
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple, List, Optional

# Import the previously defined chassis, buffer, and MCTS modules
from kaggriculture_muzero_chassis import KaggricultureMuZeroChassis
from simsiam_consistency import SimSiamProjectionPredictionHead, SimSiamConsistencyLoss
from sampled_muzero_mcts import SampledMuZeroMCTS
from muzero_replay_buffer import MuZeroTrajectoryBuffer, GameTrajectory

class MuZeroTrainer:
    """
    Complete Trainer Engine coordinating gradient updates, target network EMA,
    multi-head factorized losses, and asynchronous MCTS Reanalysis passes.
    """
    def __init__(
        self,
        online_model: KaggricultureMuZeroChassis,
        simsiam_head: SimSiamProjectionPredictionHead,
        buffer: MuZeroTrajectoryBuffer,
        mcts_engine: SampledMuZeroMCTS,
        lr: float = 3e-4,
        ema_tau: float = 0.995,
        consistency_weight: float = 0.25,
        grad_clip: float = 5.0,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.online_model = online_model.to(self.device)
        
        # 1. Instantiate Target Network (EMA copy)
        self.target_model = copy.deepcopy(online_model).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False  # Target model receives no direct gradients
            
        self.simsiam_head = simsiam_head.to(self.device)
        self.simsiam_loss_fn = SimSiamConsistencyLoss().to(self.device)
        self.buffer = buffer
        self.mcts_engine = mcts_engine

        self.ema_tau = ema_tau
        self.consistency_weight = consistency_weight
        self.grad_clip = grad_clip

        # 2. Joint Optimizer over Online Chassis & SimSiam Heads
        joint_params = list(self.online_model.parameters()) + list(self.simsiam_head.parameters())
        self.optimizer = AdamW(joint_params, lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-5)

    def update_target_network(self):
        """Polyaks EMA Target Network Update: theta^- <- tau * theta^- + (1 - tau) * theta"""
        with torch.no_grad():
            for p_online, p_target in zip(self.online_model.parameters(), self.target_model.parameters()):
                p_target.data.copy_(self.ema_tau * p_target.data + (1.0 - self.ema_tau) * p_online.data)

    def _action_tuple_to_plane(self, action_tuples: torch.Tensor) -> torch.Tensor:
        """
        Converts integer action tuples [Batch, 3] into 10x10 spatial feature planes [Batch, 16, 10, 10].
        """
        batch_size = action_tuples.shape
        plane = torch.zeros(batch_size, 16, 10, 10, device=self.device)
        for i in range(batch_size):
            f_act, h_act, m_act = action_tuples[i].tolist()
            plane[i, f_act % 5, :, :] = 1.0
            plane[i, 5 + (h_act % 5), :, :] = 1.0
            plane[i, 10 + (m_act % 6), :, :] = 1.0
        return plane

    def train_step(self, batch_size: int = 8, k_steps: int = 5) -> Dict[str, float]:
        """
        Executes a single optimization step over a K-step unroll batch sampled from the buffer.
        """
        self.online_model.train()
        self.simsiam_head.train()

        # Sample K-step trajectory slice batch
        batch = self.buffer.sample_k_step_batch(batch_size=batch_size, k_steps=k_steps)
        
        obs_batch = batch['obs'].to(self.device)                      # [Batch, K+1, 28, 10, 10]
        actions_batch = batch['actions'].to(self.device)              # [Batch, K, 3]
        rewards_batch = batch['rewards'].to(self.device)              # [Batch, K]
        target_values = batch['values'].to(self.device)               # [Batch, K+1]
        target_p_f = batch['target_policy_farmer'].to(self.device)    # [Batch, K+1, Num_Farmer]
        target_p_h = batch['target_policy_hands'].to(self.device)     # [Batch, K+1, Num_Hands]
        target_p_m = batch['target_policy_market'].to(self.device)    # [Batch, K+1, Num_Market]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward_loss = 0.0
        total_consistency_loss = 0.0

        # ------------------------------------------------------------------
        # Step 0: Root Observation Encoding (t = 0)
        # ------------------------------------------------------------------
        s_current, root_preds, root_val_scalar = self.online_model.initial_inference(obs_batch[:, 0])

        # Loss at Root (k = 0)
        total_policy_loss += (
            F.cross_entropy(root_preds['logits_farmer'], target_p_f[:, 0]) +
            F.cross_entropy(root_preds['logits_hands'], target_p_h[:, 0]) +
            F.cross_entropy(root_preds['logits_market'], target_p_m[:, 0])
        )
        total_value_loss += F.mse_loss(root_val_scalar, target_values[:, 0])

        # ------------------------------------------------------------------
        # Recurrent K-Step Unroll Loop (k = 1 ... K)
        # ------------------------------------------------------------------
        for k in range(1, k_steps + 1):
            action_plane = self._action_tuple_to_plane(actions_batch[:, k - 1])
            
            # 1. Unroll Dynamics: s_hat_k, r_hat_k = g_theta(s_hat_{k-1}, a_k)
            s_unrolled, r_predicted, preds_k, v_predicted = self.online_model.recurrent_inference(s_current, action_plane)
            
            # 2. Standard Task Prediction Losses
            total_policy_loss += (
                F.cross_entropy(preds_k['logits_farmer'], target_p_f[:, k]) +
                F.cross_entropy(preds_k['logits_hands'], target_p_h[:, k]) +
                F.cross_entropy(preds_k['logits_market'], target_p_m[:, k])
            )
            total_value_loss += F.mse_loss(v_predicted, target_values[:, k])
            total_reward_loss += F.mse_loss(r_predicted, rewards_batch[:, k - 1])

            # 3. Ground-Truth Target Embedding from Target Network: z_k = h_target(o_k)
            with torch.no_grad():
                target_z_k = self.target_model.representation(obs_batch[:, k])

            # 4. SimSiam Self-Supervised Consistency Loss
            c_loss_k = self.simsiam_loss_fn(s_unrolled, target_z_k, self.simsiam_head)
            total_consistency_loss += c_loss_k

            s_current = s_unrolled

        # Normalize across unroll horizon
        mean_policy_loss = total_policy_loss / (k_steps + 1)
        mean_value_loss = total_value_loss / (k_steps + 1)
        mean_reward_loss = total_reward_loss / k_steps
        mean_consistency_loss = total_consistency_loss / k_steps

        # Composite Loss Function
        total_loss = (
            mean_policy_loss + 
            0.25 * mean_value_loss + 
            mean_reward_loss + 
            self.consistency_weight * mean_consistency_loss
        )

        # ------------------------------------------------------------------
        # Gradient Backpropagation & Parameter Update
        # ------------------------------------------------------------------
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.online_model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        # Update EMA Target Network
        self.update_target_network()

        return {
            'loss_total': total_loss.item(),
            'loss_policy': mean_policy_loss.item(),
            'loss_value': mean_value_loss.item(),
            'loss_reward': mean_reward_loss.item(),
            'loss_consistency': mean_consistency_loss.item(),
            'lr': self.scheduler.get_last_lr()
        }

    def run_reanalyze_pass(self, num_trajectories: int = 1):
        """
        Refreshes stale MCTS visit distributions and bootstrapped value targets
        in the replay buffer using the latest network parameters.
        """
        if not self.buffer.trajectories:
            return

        sampled_indices = random.sample(
            range(len(self.buffer.trajectories)), 
            min(num_trajectories, len(self.buffer.trajectories))
        )
        for idx in sampled_indices:
            self.buffer.reanalyze_trajectory(idx, self.mcts_engine, self.device)

# ============================================================================
# VERIFICATION & TEST RUNNER
# ============================================================================
if __name__ == "__main__":
    device = "cpu"
    
    # 1. Initialize Chassis, SimSiam Head, Buffer, and MCTS
    online_chassis = KaggricultureMuZeroChassis()
    simsiam_head = SimSiamProjectionPredictionHead(latent_dim=64, proj_dim=64, pred_dim=32)
    mcts_engine = SampledMuZeroMCTS(online_chassis, num_samples=4, num_simulations=10)
    replay_buffer = MuZeroTrajectoryBuffer(max_trajectories=10)

    # 2. Populate Replay Buffer with Mock Game Trajectories
    for ep in range(2):
        traj = GameTrajectory()
        for t in range(15):
            obs = torch.randn(28, 10, 10).numpy()
            action = (random.randint(0, 14), random.randint(0, 31), random.randint(0, 19))
            reward = float(random.choice([0.0, 10.0, 25.0]))
            policy_dict = {action: 0.8, ((action+1)%15, action, action): 0.2}
            value = random.uniform(5.0, 100.0)
            traj.append_step(obs, action, reward, policy_dict, value)
            
        traj.finalize(final_obs=torch.randn(28, 10, 10).numpy())
        replay_buffer.save_trajectory(traj)

    # 3. Instantiate Trainer
    trainer = MuZeroTrainer(
        online_model=online_chassis,
        simsiam_head=simsiam_head,
        buffer=replay_buffer,
        mcts_engine=mcts_engine,
        lr=1e-3,
        device=device
    )

    print("=== Starting MuZero Training & Reanalyze Verification Loop ===")
    
    # Run 3 Training Steps
    for step in range(1, 4):
        metrics = trainer.train_step(batch_size=2, k_steps=3)
        print(f"\nTraining Iteration {step}:")
        print(f"  Total Loss:       {metrics['loss_total']:.4f}")
        print(f"  Policy Loss:      {metrics['loss_policy']:.4f}")
        print(f"  Value Loss:       {metrics['loss_value']:.4f}")
        print(f"  Consistency Loss: {metrics['loss_consistency']:.4f}")
        print(f"  Learning Rate:    {metrics['lr']:.6f}")

    # Run Reanalyze Pass
    print("\nExecuting Buffer Reanalyze Pass...")
    trainer.run_reanalyze_pass(num_trajectories=1)
    print("Reanalyze Pass completed successfully! Stale targets updated in-place.")
```

---

### Pipeline Architecture Summary

```
   [ Replay Buffer ] ──(Sample Batch)──► [ Representation Network h_θ ]
          │                                        │
          │ (Reanalyze Refresh)                    ▼
          ▼                              Root Latent State s_0
   [ Sampled MCTS ]                                │
   (Target Refresh) ◄──────────────────────────────┤
                                                   ▼
                                     [ Recurrent Loop k=1..K ]
                                        • Dynamics g_theta
                                        • Prediction f_theta
                                                   │
                                                   ▼
                                     [ SimSiam Consistency ]
                                      • vs h_target(o_k)
```

1. **`train_step`**: Handles sampling \\(K\\)-step unrolls, forward recurrent prediction through `online_model`, target generation via `target_model` (EMA), gradient clipping, and optimizer stepping.
2. **`run_reanalyze_pass`**: Keeps the replay buffer targets aligned with the latest policy network, ensuring high sample efficiency during offline or low-data training.

---
Integrating **Prioritized Experience Replay (PER)** into a MuZero training pipeline replaces uniform trajectory sampling with a data-driven curriculum. By prioritizing trajectory slices where the network's value predictions exhibit high temporal-difference (TD) error, the agent focuses its gradient updates on complex economic transitions, accelerating convergence.

---

### 1. Mathematical Formulation for MuZero PER

In standard DQN, TD-error is calculated across single-step Bellman transitions. In **MuZero**, because the network is unrolled over \\(K\\) recurrent steps, priority is defined across the unroll sequence.

#### A. Trajectory Priority Calculation
For a trajectory slice starting at time \\(t\\) unrolled over \\(K\\) steps, the priority \\(p_t\\) is determined by the **maximum absolute value error** between the search value estimate \\(\nu\\) and the \\(n\\)-step bootstrapped target \\(z\\):

\\[p_t = \max_{k=0 \dots K} \left| v_{t+k} - z_{t+k} \right| + \epsilon\\]

*(where \\(\epsilon \approx 10^{-5}\\) is a small positive constant preventing zero-priority starvation).*

#### B. Sampling Probability
The probability \\(P(i)\\) of sampling trajectory slice \\(i\\) from a buffer of \\(N\\) available transitions is:

\\[P(i) = \frac{p_i^\alpha}{\sum_{k=1}^N p_k^\alpha}\\]

* \\(\alpha \in [0.6, 1.0]\\) dictates how aggressively high-error slices are prioritized (\\(\alpha = 0\\) yields uniform sampling).

#### C. Importance Sampling (IS) Weights
Because non-uniform sampling introduces estimation bias into stochastic gradient descent, each sampled transition is scaled by an **Importance Sampling (IS) weight** \\(w_i\\):

\\[w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta \bigg/ \max_j w_j\\]

* \\(\beta\\) is annealed linearly from \\(\beta_{\text{start}} = 0.4\\) to \\(1.0\\) across training, ensuring full bias correction as network parameters stabilize.

---

### 2. PyTorch Implementation: `prioritized_muzero_buffer.py`

Below is the complete PyTorch implementation extending `MuZeroTrajectoryBuffer` and `MuZeroTrainer` with **PER sampling weights** and dynamic priority updates.

```python
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import chassis and buffer base structures
from kaggriculture_muzero_chassis import KaggricultureMuZeroChassis
from muzero_replay_buffer import GameTrajectory, MuZeroTrajectoryBuffer

class PrioritizedMuZeroBuffer(MuZeroTrajectoryBuffer):
    """
    Trajectory-Based Replay Buffer supporting TD-Error Prioritized Experience Replay (PER)
    and Importance Sampling (IS) weight computation for MuZero K-step unrolls.
    """
    def __init__(
        self,
        max_trajectories: int = 1000,
        discount: float = 0.997,
        n_steps: int = 5,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        eps: float = 1e-5
    ):
        super().__init__(max_trajectories=max_trajectories, discount=discount, n_steps=n_steps)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame_count = 0

    def update_beta(self, current_frame: int) -> float:
        """Anneals beta linearly from beta_start -> 1.0."""
        self.frame_count = current_frame
        fraction = min(1.0, current_frame / float(self.beta_frames))
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)
        return self.beta

    def sample_prioritized_k_step_batch(
        self, 
        batch_size: int = 16, 
        k_steps: int = 5
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]], torch.Tensor]:
        """
        Samples a batch of K-step unroll slices proportional to TD-error priorities P(i)^alpha.
        
        Returns:
            batch_tensors: Dictionary containing K-step observation, action, value, and policy targets
            batch_slice_keys: List of (trajectory_idx, start_step_idx) for priority updating
            is_weights: Importance sampling weights tensor [Batch, 1]
        """
        valid_slices: List[Tuple[int, int, float]] = [] # (traj_idx, start_idx, priority)
        
        for t_idx, traj in enumerate(self.trajectories):
            if len(traj) >= k_steps:
                for s_idx in range(len(traj) - k_steps):
                    # Fetch stored max TD priority for this slice
                    prio = traj.priorities[s_idx] if s_idx < len(traj.priorities) else 1.0
                    valid_slices.append((t_idx, s_idx, prio))

        if not valid_slices:
            raise ValueError(f"No valid trajectory slices of length >= {k_steps}")

        N = len(valid_slices)
        priorities = np.array([p for _, _, p in valid_slices], dtype=np.float32)
        
        # Calculate Sampling Probabilities P(i) = p_i^alpha / sum(p^alpha)
        probs = np.power(priorities + self.eps, self.alpha)
        probs /= probs.sum()

        # Sample batch indices according to P(i)
        sampled_indices = np.random.choice(N, size=batch_size, p=probs, replace=True)
        sampled_slices = [valid_slices[idx] for idx in sampled_indices]

        # Calculate Importance Sampling Weights w_i = (N * P(i))^(-beta) / max(w)
        sampled_probs = probs[sampled_indices]
        weights = np.power(N * sampled_probs, -self.beta)
        weights /= weights.max() # Normalize for numerical stability
        is_weights_tensor = torch.from_numpy(weights).float().unsqueeze(1) # [Batch, 1]

        # Extract Batch Tensors
        batch_obs, batch_actions, batch_rewards, batch_values = [], [], [], []
        batch_p_f, batch_p_h, batch_p_m = [], [], []
        batch_slice_keys = []

        for t_idx, s_idx, _ in sampled_slices:
            traj = self.trajectories[t_idx]
            batch_slice_keys.append((t_idx, s_idx))

            slice_obs, slice_actions, slice_rewards, slice_values = [], [], [], []
            slice_pf, slice_ph, slice_pm = [], [], []

            for k in range(k_steps + 1):
                curr_idx = s_idx + k
                slice_obs.append(traj.observations[curr_idx])
                
                # Compute n-step value target
                val_target = self._compute_n_step_target(traj, curr_idx)
                slice_values.append(val_target)
                
                pf, ph, pm = self._convert_policy_dict_to_factorized_targets(traj.policies[curr_idx])
                slice_pf.append(pf)
                slice_ph.append(ph)
                slice_pm.append(pm)

                if k < k_steps:
                    slice_actions.append(traj.actions[curr_idx])
                    slice_rewards.append(traj.rewards[curr_idx])

            batch_obs.append(np.array(slice_obs))
            batch_actions.append(np.array(slice_actions))
            batch_rewards.append(np.array(slice_rewards))
            batch_values.append(np.array(slice_values))
            batch_p_f.append(np.array(slice_pf))
            batch_p_h.append(np.array(slice_ph))
            batch_policy_m = batch_p_m.append(np.array(slice_pm))

        batch_dict = {
            'obs': torch.from_numpy(np.array(batch_obs)).float(),
            'actions': torch.from_numpy(np.array(batch_actions)).long(),
            'rewards': torch.from_numpy(np.array(batch_rewards)).float(),
            'values': torch.from_numpy(np.array(batch_values)).float(),
            'target_policy_farmer': torch.from_numpy(np.array(batch_p_f)).float(),
            'target_policy_hands': torch.from_numpy(np.array(batch_p_h)).float(),
            'target_policy_market': torch.from_numpy(np.array(batch_p_m)).float()
        }

        return batch_dict, batch_slice_keys, is_weights_tensor

    def update_slice_priorities(
        self, 
        slice_keys: List[Tuple[int, int]], 
        td_errors: np.ndarray
    ):
        """Updates stored priorities in the buffer using fresh absolute TD-errors."""
        for (t_idx, s_idx), error in zip(slice_keys, td_errors):
            if t_idx < len(self.trajectories):
                traj = self.trajectories[t_idx]
                if s_idx < len(traj.priorities):
                    traj.priorities[s_idx] = float(error) + self.eps
```

---

### 3. Integrating PER into the Trainer Loop

To apply PER during gradient backpropagation, the loss computation scales each sequence's element-wise loss by its corresponding Importance Sampling weight \\(w_i\\):

\\[\mathcal{L}_{\text{weighted}} = \frac{1}{B} \sum_{i=1}^B w_i \cdot \ell_i\\]

```python
class PrioritizedMuZeroTrainer:
    """
    Trainer engine integrating Prioritized Experience Replay (PER),
    IS-weighted loss updates, and dynamic priority refreshes.
    """
    def __init__(
        self,
        model: KaggricultureMuZeroChassis,
        buffer: PrioritizedMuZeroBuffer,
        lr: float = 1e-3,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.buffer = buffer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_step_per(self, batch_size: int = 8, k_steps: int = 5, global_step: int = 0) -> Dict[str, float]:
        self.model.train()
        
        # 1. Update beta schedule
        current_beta = self.buffer.update_beta(global_step)

        # 2. Sample PER batch with IS weights
        batch, slice_keys, is_weights = self.buffer.sample_prioritized_k_step_batch(
            batch_size=batch_size, k_steps=k_steps
        )

        obs = batch['obs'].to(self.device)             # [Batch, K+1, 28, 10, 10]
        actions = batch['actions'].to(self.device)     # [Batch, K, 3]
        target_values = batch['values'].to(self.device)# [Batch, K+1]
        is_weights = is_weights.to(self.device)        # [Batch, 1]

        # 3. Compute Per-Sample Unroll Losses
        s_curr, root_preds, root_val = self.model.initial_inference(obs[:, 0])
        
        # Per-sample value error at root (k = 0)
        value_errors = torch.abs(root_val - target_values[:, 0])  # [Batch]
        total_value_loss = F.mse_loss(root_val, target_values[:, 0], reduction='none') # [Batch]

        # Recurrent unroll
        for k in range(1, k_steps + 1):
            # (Action plane conversion omitted for brevity)
            action_plane = torch.zeros(batch_size, 16, 10, 10, device=self.device)
            s_next, r_pred, preds_k, v_pred = self.model.recurrent_inference(s_curr, action_plane)
            
            step_val_error = torch.abs(v_pred - target_values[:, k])
            value_errors = torch.max(value_errors, step_val_error) # Track max TD-error across horizon
            total_value_loss += F.mse_loss(v_pred, target_values[:, k], reduction='none')
            s_curr = s_next

        # 4. Apply IS Weights to Loss
        weighted_loss = (is_weights * total_value_loss.unsqueeze(1)).mean()

        # 5. Backpropagation
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()

        # 6. Update Priorities in Replay Buffer with fresh TD-errors
        fresh_td_errors = value_errors.detach().cpu().numpy()
        self.buffer.update_slice_priorities(slice_keys, fresh_td_errors)

        return {
            'loss': weighted_loss.item(),
            'beta': current_beta,
            'mean_td_error': float(np.mean(fresh_td_errors))
        }
```

---

### Key Advantages of PER in MuZero

1. **Eliminates Manual Sampling Heuristics**: Replaces static re-sampling rules with a dynamic curriculum focused on states where value or policy predictions struggle.
2. **Accelerated Value Convergence**: Focuses replay bandwidth on high-yield economic transitions (such as land expansion steps or price crashes).
3. **Unbiased Optimization**: Normalizing gradients with Importance Sampling weights \\(w_i\\) prevents model instability and gradient explosion on high-priority trajectories.

---

🌾 **Next Step**: Would you like to generate a downloadable **Python Script** or **Technical Report** artifact compiling this complete Prioritized MuZero system for your codebase?