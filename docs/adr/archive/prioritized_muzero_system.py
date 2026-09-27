"""
NOT CANONICAL — ARCHIVED REFERENCE ONLY.

See docs/adr/0001-spatial-sampled-muzero-deferred.md.
Do NOT wire this module into muzero/, pipelines/, train.sh, or packaging.
The production chassis is muzero/_core.py (32-dim MLP, 8 MacroOptions, Gumbel MCTS).

================================================================================
HISTORICAL: Spatial Sampled MuZero sketch for Kaggriculture (deferred)
================================================================================
Features (not implemented in production):
1. KaggricultureMuZeroChassis: spatial ResNet h/g/f + factorized farmer/hands/market.
2. SimSiam Consistency with projector/predictor heads.
3. Sampled MCTS over joint action tuples.
4. Trajectory PER + reanalyze + EMA target network.
================================================================================
"""

import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# ============================================================================
# 1. RESNET & CHASSIS NETWORKS (h_theta, g_theta, f_theta)
# ============================================================================

class ResNetBlock2D(nn.Module):
    """Pre-activation ResNet block preserving 10x10 spatial grid dimensions."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class RepresentationNetwork(nn.Module):
    """
    h_theta: Encodes raw 10x10 spatial observation tensor o_t into root latent state s_0.
    Input:  [Batch, C_obs, 10, 10]
    Output: Latent state tensor s_0 [Batch, C_latent, 10, 10]
    """
    def __init__(self, in_channels: int = 28, latent_channels: int = 64, num_blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([ResNetBlock2D(latent_channels) for _ in range(num_blocks)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        s_0 = self.stem(obs)
        for block in self.res_blocks:
            s_0 = block(s_0)
        s_0 = F.normalize(s_0, p=2, dim=1)
        return s_0


class DynamicsNetwork(nn.Module):
    """
    g_theta: Predicts next latent state s^k and immediate reward r^k given (s^{k-1}, a^k).
    Inputs:
        s_prev: Latent state tensor [Batch, C_latent, 10, 10]
        action_plane: One-hot / spatial action encoding plane [Batch, C_action, 10, 10]
    Outputs:
        s_next: Transitioned latent state [Batch, C_latent, 10, 10]
        reward_logits: Categorical reward distribution logits [Batch, 601]
    """
    def __init__(
        self, 
        latent_channels: int = 64, 
        action_channels: int = 16, 
        num_blocks: int = 3,
        support_size: int = 601
    ):
        super().__init__()
        in_channels = latent_channels + action_channels
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([ResNetBlock2D(latent_channels) for _ in range(num_blocks)])
        
        self.reward_head = nn.Sequential(
            nn.Conv2d(latent_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 10 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, support_size)
        )

    def forward(self, s_prev: torch.Tensor, action_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s_prev, action_plane], dim=1)
        s_next = self.stem(x)
        for block in self.res_blocks:
            s_next = block(s_next)
            
        s_next = F.normalize(s_next, p=2, dim=1)
        reward_logits = self.reward_head(s_next)
        
        return s_next, reward_logits


class PredictionNetwork(nn.Module):
    """
    f_theta: Predicts factorized policy priors p^k and value v^k from latent state s^k.
    Supports candidate joint action sampling for Sampled MCTS.
    """
    def __init__(
        self, 
        latent_channels: int = 64, 
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20,
        support_size: int = 601
    ):
        super().__init__()
        
        self.trunk = nn.Sequential(
            nn.Conv2d(latent_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 10 * 10, 256),
            nn.ReLU(inplace=True)
        )
        
        self.farmer_head = nn.Linear(256, num_farmer_actions)
        self.hands_head = nn.Linear(256, num_hand_assignments)
        self.market_head = nn.Linear(256, num_market_orders)
        self.value_head = nn.Linear(256, support_size)

    def forward(self, s_k: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.trunk(s_k)
        return {
            'logits_farmer': self.farmer_head(features),
            'logits_hands': self.hands_head(features),
            'logits_market': self.market_head(features),
            'value_logits': self.value_head(features)
        }

    def sample_joint_actions(
        self, 
        s_k: torch.Tensor, 
        num_samples: int = 16,
        masks: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.forward(s_k)
        
        logits_f = preds['logits_farmer']
        logits_h = preds['logits_hands']
        logits_m = preds['logits_market']
        
        if masks is not None:
            logits_f = logits_f.masked_fill(masks['farmer'] == 0, -1e9)
            logits_h = logits_h.masked_fill(masks['hands'] == 0, -1e9)
            logits_m = logits_m.masked_fill(masks['market'] == 0, -1e9)

        dist_f = torch.distributions.Categorical(logits=logits_f)
        dist_h = torch.distributions.Categorical(logits=logits_h)
        dist_m = torch.distributions.Categorical(logits=logits_m)
        
        a_f = dist_f.sample((num_samples,)).transpose(0, 1)
        a_h = dist_h.sample((num_samples,)).transpose(0, 1)
        a_m = dist_m.sample((num_samples,)).transpose(0, 1)
        
        joint_actions = torch.stack([a_f, a_h, a_m], dim=-1)
        
        log_prob = (
            dist_f.log_prob(a_f) + 
            dist_h.log_prob(a_h) + 
            dist_m.log_prob(a_m)
        )
        
        return joint_actions, log_prob


class KaggricultureMuZeroChassis(nn.Module):
    """Unified container for Representation (h), Dynamics (g), and Prediction (f)."""
    def __init__(
        self,
        obs_channels: int = 28,
        latent_channels: int = 64,
        action_channels: int = 16,
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20,
        support_size: int = 601
    ):
        super().__init__()
        self.support_size = support_size
        self.register_buffer("support", torch.linspace(-300, 300, support_size))
        
        self.representation = RepresentationNetwork(obs_channels, latent_channels)
        self.dynamics = DynamicsNetwork(latent_channels, action_channels, support_size=support_size)
        self.prediction = PredictionNetwork(
            latent_channels, num_farmer_actions, num_hand_assignments, num_market_orders, support_size
        )

    def initial_inference(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        s_0 = self.representation(obs)
        preds = self.prediction(s_0)
        value_scalar = self._logits_to_scalar(preds['value_logits'])
        return s_0, preds, value_scalar

    def recurrent_inference(
        self, 
        s_prev: torch.Tensor, 
        action_plane: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        s_next, reward_logits = self.dynamics(s_prev, action_plane)
        preds = self.prediction(s_next)
        reward_scalar = self._logits_to_scalar(reward_logits)
        value_scalar = self._logits_to_scalar(preds['value_logits'])
        return s_next, reward_scalar, preds, value_scalar

    def _logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.support, dim=-1)


# ============================================================================
# 2. SIMSIAM SELF-SUPERVISED CONSISTENCY MODULE
# ============================================================================

class SimSiamProjectionPredictionHead(nn.Module):
    """Asymmetric Projection and Prediction heads for SimSiam-style latent consistency."""
    def __init__(self, latent_dim: int = 64, proj_dim: int = 64, pred_dim: int = 32):
        super().__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(latent_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim)
        )
        self.pred_head = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim)
        )

    def project(self, latent_state: torch.Tensor) -> torch.Tensor:
        if latent_state.dim() > 2:
            latent_state = latent_state.mean(dim=[-2, -1]) # Global average pool if spatial
        return self.proj_head(latent_state)

    def predict(self, projected_state: torch.Tensor) -> torch.Tensor:
        return self.pred_head(projected_state)


class SimSiamConsistencyLoss(nn.Module):
    """Computes Negative Cosine Similarity with explicit Stop-Gradient on target."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self, 
        unrolled_latent: torch.Tensor, 
        target_observation_embedding: torch.Tensor,
        simsiam_head: SimSiamProjectionPredictionHead
    ) -> torch.Tensor:
        p_k = simsiam_head.predict(simsiam_head.project(unrolled_latent))
        
        with torch.no_grad():
            z_k_target = simsiam_head.project(target_observation_embedding).detach()

        p_k_norm = F.normalize(p_k, dim=-1, eps=self.eps)
        z_k_norm = F.normalize(z_k_target, dim=-1, eps=self.eps)

        cosine_sim = (p_k_norm * z_k_norm).sum(dim=-1)
        return -cosine_sim.mean()


# ============================================================================
# 3. SAMPLED MCTS SEARCH ENGINE
# ============================================================================

class MinMaxStats:
    """Tracks global minimum and maximum Q-values across the search tree."""
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
    """Node in the latent space Monte Carlo search tree."""
    def __init__(self, prior: float = 0.0, action_tuple: Optional[Tuple[int, int, int]] = None):
        self.prior = prior
        self.action_tuple = action_tuple
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
    """Executes Monte Carlo Tree Search in latent space using KaggricultureMuZeroChassis."""
    def __init__(
        self,
        chassis: KaggricultureMuZeroChassis,
        c1: float = 1.25,
        c2: float = 19652.0,
        discount: float = 0.997,
        num_samples: int = 8,
        num_simulations: int = 25,
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
        pb_c = math.log((parent.visit_count + self.c2 + 1.0) / self.c2) + self.c1
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1.0)
        
        prior_score = pb_c * child.prior
        value_score = 0.0
        if child.visit_count > 0:
            value_score = min_max.normalize(child.reward + self.discount * child.value)
            
        return prior_score + value_score

    def _select_child(self, node: MCTSNode, min_max: MinMaxStats) -> Tuple[Tuple[int, int, int], MCTSNode]:
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
        self.chassis.eval()
        device = obs_tensor.device
        min_max = MinMaxStats()

        s_0, root_preds, root_value_scalar = self.chassis.initial_inference(obs_tensor)
        
        root = MCTSNode()
        root.latent_state = s_0
        root.visit_count = 1
        root.value_sum = root_value_scalar.item()
        min_max.update(root.value_sum)

        joint_actions, joint_log_probs = self.chassis.prediction.sample_joint_actions(
            s_0, num_samples=self.num_samples
        )
        priors = F.softmax(joint_log_probs, dim=-1).squeeze(0).cpu().numpy()
        
        if add_dirichlet_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * self.num_samples)
            priors = (1.0 - self.exploration_fraction) * priors + self.exploration_fraction * noise

        for k in range(self.num_samples):
            a_tuple = tuple(joint_actions[0, k].cpu().numpy().tolist())
            root.children[a_tuple] = MCTSNode(prior=float(priors[k]), action_tuple=a_tuple)

        for _ in range(self.num_simulations):
            node = root
            search_path = [root]

            while node.is_expanded:
                _, node = self._select_child(node, min_max)
                search_path.append(node)

            parent_node = search_path[-2]
            action_tuple = node.action_tuple
            action_plane = self._action_tuple_to_plane(action_tuple, batch_size=1, device=device)

            s_next, r_scalar, preds, v_scalar = self.chassis.recurrent_inference(
                parent_node.latent_state, action_plane
            )

            node.latent_state = s_next
            node.reward = r_scalar.item()
            leaf_value = v_scalar.item()

            child_actions, child_log_probs = self.chassis.prediction.sample_joint_actions(
                s_next, num_samples=self.num_samples
            )
            child_priors = F.softmax(child_log_probs, dim=-1).squeeze(0).cpu().numpy()

            for k in range(self.num_samples):
                c_tuple = tuple(child_actions[0, k].cpu().numpy().tolist())
                node.children[c_tuple] = MCTSNode(prior=float(child_priors[k]), action_tuple=c_tuple)

            value = leaf_value
            for path_node in reversed(search_path):
                path_node.visit_count += 1
                path_node.value_sum += value
                value = path_node.reward + self.discount * value
                min_max.update(value)

        total_visits = sum(child.visit_count for child in root.children.values())
        action_probabilities = {
            a_tuple: child.visit_count / max(1, total_visits)
            for a_tuple, child in root.children.items()
        }

        best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]

        return best_action, action_probabilities, root.value


# ============================================================================
# 4. PRIORITIZED EXPERIENCE REPLAY (PER) TRAJECTORY BUFFER
# ============================================================================

class GameTrajectory:
    """Stores a single complete game episode trajectory."""
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[Tuple[int, int, int]] = []
        self.rewards: List[float] = []
        self.policies: List[Dict[Tuple[int, int, int], float]] = []
        self.values: List[float] = []
        self.priorities: List[float] = []

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
        self.observations.append(final_obs)


class PrioritizedMuZeroBuffer:
    """Trajectory-Based Replay Buffer supporting TD-Error PER & IS Weights."""
    def __init__(
        self,
        max_trajectories: int = 1000,
        discount: float = 0.997,
        n_steps: int = 5,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        eps: float = 1e-5,
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20
    ):
        self.max_trajectories = max_trajectories
        self.discount = discount
        self.n_steps = n_steps
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        
        self.num_farmer = num_farmer_actions
        self.num_hands = num_hand_assignments
        self.num_market = num_market_orders
        
        self.trajectories: List[GameTrajectory] = []

    def save_trajectory(self, trajectory: GameTrajectory):
        if len(self.trajectories) >= self.max_trajectories:
            self.trajectories.pop(0)
        self.trajectories.append(trajectory)

    def update_beta(self, current_frame: int) -> float:
        fraction = min(1.0, current_frame / float(self.beta_frames))
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)
        return self.beta

    def _compute_n_step_target(self, trajectory: GameTrajectory, index: int) -> float:
        target_val = 0.0
        trajectory_len = len(trajectory)
        
        for j in range(self.n_steps):
            if index + j < trajectory_len:
                target_val += (self.discount ** j) * trajectory.rewards[index + j]
            else:
                break
                
        bootstrap_idx = index + self.n_steps
        if bootstrap_idx < trajectory_len:
            target_val += (self.discount ** self.n_steps) * trajectory.values[bootstrap_idx]
            
        return target_val

    def _convert_policy_dict_to_factorized_targets(
        self, 
        policy_dict: Dict[Tuple[int, int, int], float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_f = np.zeros(self.num_farmer, dtype=np.float32)
        target_h = np.zeros(self.num_hands, dtype=np.float32)
        target_m = np.zeros(self.num_market, dtype=np.float32)

        for (f_act, h_act, m_act), prob in policy_dict.items():
            target_f[f_act % self.num_farmer] += prob
            target_h[h_act % self.num_hands] += prob
            target_m[m_act % self.num_market] += prob

        target_f /= max(1e-6, target_f.sum())
        target_h /= max(1e-6, target_h.sum())
        target_m /= max(1e-6, target_m.sum())

        return target_f, target_h, target_m

    def sample_prioritized_k_step_batch(
        self, 
        batch_size: int = 8, 
        k_steps: int = 5
    ) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]], torch.Tensor]:
        valid_slices: List[Tuple[int, int, float]] = []
        
        for t_idx, traj in enumerate(self.trajectories):
            if len(traj) >= k_steps:
                for s_idx in range(len(traj) - k_steps):
                    prio = traj.priorities[s_idx] if s_idx < len(traj.priorities) else 1.0
                    valid_slices.append((t_idx, s_idx, prio))

        if not valid_slices:
            raise ValueError(f"No valid trajectory slices of length >= {k_steps}")

        N = len(valid_slices)
        priorities = np.array([p for _, _, p in valid_slices], dtype=np.float32)
        
        probs = np.power(priorities + self.eps, self.alpha)
        probs /= probs.sum()

        sampled_indices = np.random.choice(N, size=batch_size, p=probs, replace=True)
        sampled_slices = [valid_slices[idx] for idx in sampled_indices]

        sampled_probs = probs[sampled_indices]
        weights = np.power(N * sampled_probs, -self.beta)
        weights /= weights.max()
        is_weights_tensor = torch.from_numpy(weights).float().unsqueeze(1)

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
            batch_p_m.append(np.array(slice_pm))

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

    def update_slice_priorities(self, slice_keys: List[Tuple[int, int]], td_errors: np.ndarray):
        for (t_idx, s_idx), error in zip(slice_keys, td_errors):
            if t_idx < len(self.trajectories):
                traj = self.trajectories[t_idx]
                if s_idx < len(traj.priorities):
                    traj.priorities[s_idx] = float(error) + self.eps

    def reanalyze_trajectory(self, traj_idx: int, mcts_engine: SampledMuZeroMCTS, device: torch.device):
        traj = self.trajectories[traj_idx]
        for t in range(len(traj)):
            obs_tensor = torch.from_numpy(traj.observations[t]).unsqueeze(0).to(device)
            best_act, fresh_policy_dict, fresh_val = mcts_engine.search(obs_tensor, add_dirichlet_noise=False)
            traj.policies[t] = fresh_policy_dict
            traj.values[t] = fresh_val


# ============================================================================
# 5. COMPLETE MUZERO TRAINER ENGINE WITH PER & EMA TARGET NETWORK
# ============================================================================

class PrioritizedMuZeroTrainer:
    """Unified Trainer coordinating PER, IS-weighted loss, EMA Target Model, and Reanalyze."""
    def __init__(
        self,
        online_model: KaggricultureMuZeroChassis,
        simsiam_head: SimSiamProjectionPredictionHead,
        buffer: PrioritizedMuZeroBuffer,
        mcts_engine: SampledMuZeroMCTS,
        lr: float = 1e-3,
        ema_tau: float = 0.995,
        consistency_weight: float = 0.25,
        grad_clip: float = 5.0,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.online_model = online_model.to(self.device)
        
        self.target_model = copy.deepcopy(online_model).to(self.device)
        for param in self.target_model.parameters():
            param.requires_grad = False
            
        self.simsiam_head = simsiam_head.to(self.device)
        self.simsiam_loss_fn = SimSiamConsistencyLoss().to(self.device)
        self.buffer = buffer
        self.mcts_engine = mcts_engine

        self.ema_tau = ema_tau
        self.consistency_weight = consistency_weight
        self.grad_clip = grad_clip

        joint_params = list(self.online_model.parameters()) + list(self.simsiam_head.parameters())
        self.optimizer = torch.optim.AdamW(joint_params, lr=lr, weight_decay=1e-4)

    def update_target_network(self):
        with torch.no_grad():
            for p_online, p_target in zip(self.online_model.parameters(), self.target_model.parameters()):
                p_target.data.copy_(self.ema_tau * p_target.data + (1.0 - self.ema_tau) * p_online.data)

    def _action_tuple_to_plane(self, action_tuples: torch.Tensor) -> torch.Tensor:
        batch_size = action_tuples.shape[0]
        plane = torch.zeros(batch_size, 16, 10, 10, device=self.device)
        for i in range(batch_size):
            f_act, h_act, m_act = action_tuples[i].tolist()
            plane[i, f_act % 5, :, :] = 1.0
            plane[i, 5 + (h_act % 5), :, :] = 1.0
            plane[i, 10 + (m_act % 6), :, :] = 1.0
        return plane

    def train_step(self, batch_size: int = 4, k_steps: int = 3, global_step: int = 0) -> Dict[str, float]:
        self.online_model.train()
        self.simsiam_head.train()

        current_beta = self.buffer.update_beta(global_step)

        # 1. Sample PER Batch with IS Weights
        batch, slice_keys, is_weights = self.buffer.sample_prioritized_k_step_batch(
            batch_size=batch_size, k_steps=k_steps
        )

        obs_batch = batch['obs'].to(self.device)
        actions_batch = batch['actions'].to(self.device)
        rewards_batch = batch['rewards'].to(self.device)
        target_values = batch['values'].to(self.device)
        target_p_f = batch['target_policy_farmer'].to(self.device)
        target_p_h = batch['target_policy_hands'].to(self.device)
        target_p_m = batch['target_policy_market'].to(self.device)
        is_weights = is_weights.to(self.device)

        total_policy_loss = torch.zeros(batch_size, device=self.device)
        total_value_loss = torch.zeros(batch_size, device=self.device)
        total_reward_loss = torch.zeros(batch_size, device=self.device)
        total_consistency_loss = torch.zeros(batch_size, device=self.device)

        value_errors = torch.zeros(batch_size, device=self.device)

        # 2. Root Pass (k = 0)
        s_current, root_preds, root_val_scalar = self.online_model.initial_inference(obs_batch[:, 0])

        total_policy_loss += (
            F.cross_entropy(root_preds['logits_farmer'], target_p_f[:, 0], reduction='none') +
            F.cross_entropy(root_preds['logits_hands'], target_p_h[:, 0], reduction='none') +
            F.cross_entropy(root_preds['logits_market'], target_p_m[:, 0], reduction='none')
        )
        total_value_loss += F.mse_loss(root_val_scalar, target_values[:, 0], reduction='none')
        value_errors = torch.abs(root_val_scalar - target_values[:, 0])

        # 3. Recurrent Unroll Loop (k = 1 ... K)
        for k in range(1, k_steps + 1):
            action_plane = self._action_tuple_to_plane(actions_batch[:, k - 1])
            s_unrolled, r_predicted, preds_k, v_predicted = self.online_model.recurrent_inference(s_current, action_plane)

            total_policy_loss += (
                F.cross_entropy(preds_k['logits_farmer'], target_p_f[:, k], reduction='none') +
                F.cross_entropy(preds_k['logits_hands'], target_p_h[:, k], reduction='none') +
                F.cross_entropy(preds_k['logits_market'], target_p_m[:, k], reduction='none')
            )
            total_value_loss += F.mse_loss(v_predicted, target_values[:, k], reduction='none')
            total_reward_loss += F.mse_loss(r_predicted, rewards_batch[:, k - 1], reduction='none')

            # Track Max TD-Error across horizon for priority updating
            step_val_error = torch.abs(v_predicted - target_values[:, k])
            value_errors = torch.max(value_errors, step_val_error)

            with torch.no_grad():
                target_z_k = self.target_model.representation(obs_batch[:, k])

            c_loss_k = self.simsiam_loss_fn(s_unrolled, target_z_k, self.simsiam_head)
            total_consistency_loss += c_loss_k

            s_current = s_unrolled

        # Normalize per-sample losses
        mean_p_loss = total_policy_loss / (k_steps + 1)
        mean_v_loss = total_value_loss / (k_steps + 1)
        mean_r_loss = total_reward_loss / k_steps
        mean_c_loss = total_consistency_loss / k_steps

        # Composite Per-Sample Loss
        per_sample_loss = mean_p_loss + 0.25 * mean_v_loss + mean_r_loss + self.consistency_weight * mean_c_loss

        # Apply Importance Sampling Weights [Batch, 1]
        weighted_loss = (is_weights * per_sample_loss.unsqueeze(1)).mean()

        # 4. Backward & Step
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.update_target_network()

        # 5. Update Priorities in Replay Buffer
        fresh_td_errors = value_errors.detach().cpu().numpy()
        self.buffer.update_slice_priorities(slice_keys, fresh_td_errors)

        return {
            'loss_total': weighted_loss.item(),
            'loss_policy': mean_p_loss.mean().item(),
            'loss_value': mean_v_loss.mean().item(),
            'loss_consistency': mean_c_loss.mean().item(),
            'beta': current_beta,
            'mean_td_error': float(np.mean(fresh_td_errors))
        }

    def run_reanalyze_pass(self, num_trajectories: int = 1):
        if not self.buffer.trajectories:
            return
        sampled_indices = random.sample(
            range(len(self.buffer.trajectories)), 
            min(num_trajectories, len(self.buffer.trajectories))
        )
        for idx in sampled_indices:
            self.buffer.reanalyze_trajectory(idx, self.mcts_engine, self.device)


# ============================================================================
# 6. VERIFICATION & TEST EXECUTABLE RUNNER
# ============================================================================

if __name__ == "__main__":
    print("================================================================================")
    print("INITIALIZING COMPLETE PRIORITIZED MUZERO SYSTEM TEST")
    print("================================================================================")

    device = "cpu"
    
    # 1. Instantiate Core Models
    online_chassis = KaggricultureMuZeroChassis()
    simsiam_head = SimSiamProjectionPredictionHead(latent_dim=64, proj_dim=64, pred_dim=32)
    mcts_engine = SampledMuZeroMCTS(online_chassis, num_samples=4, num_simulations=10)
    per_buffer = PrioritizedMuZeroBuffer(max_trajectories=10)

    # 2. Populate Buffer with Mock Trajectories
    print("Populating PER Trajectory Buffer with synthetic games...")
    for ep in range(3):
        traj = GameTrajectory()
        for t in range(12):
            obs = np.random.randn(28, 10, 10).astype(np.float32)
            action = (random.randint(0, 14), random.randint(0, 31), random.randint(0, 19))
            reward = float(random.choice([0.0, 10.0, 50.0]))
            
            # Construct valid action tuple dictionary for policy visit probabilities
            alt_action = ((action[0] + 1) % 15, action[1], action[2])
            policy_dict = {action: 0.8, alt_action: 0.2}
            value = random.uniform(10.0, 150.0)
            priority = random.uniform(0.5, 5.0)  # TD-error priority
            traj.append_step(obs, action, reward, policy_dict, value, priority=priority)
            
        traj.finalize(final_obs=np.random.randn(28, 10, 10).astype(np.float32))
        per_buffer.save_trajectory(traj)

    print(f"Stored {len(per_buffer.trajectories)} trajectories in PER Buffer.")

    # 3. Instantiate Trainer
    trainer = PrioritizedMuZeroTrainer(
        online_model=online_chassis,
        simsiam_head=simsiam_head,
        buffer=per_buffer,
        mcts_engine=mcts_engine,
        lr=1e-3,
        device=device
    )

    # 4. Run Training Steps
    print("\nExecuting PER Training Steps with Importance Sampling Weighting:")
    for step in range(1, 4):
        metrics = trainer.train_step(batch_size=2, k_steps=3, global_step=step * 100)
        print(f"  Step {step} | Total Loss: {metrics['loss_total']:.4f} | "
              f"Policy Loss: {metrics['loss_policy']:.4f} | Value Loss: {metrics['loss_value']:.4f} | "
              f"Consistency: {metrics['loss_consistency']:.4f} | Beta: {metrics['beta']:.3f} | "
              f"Mean TD Error: {metrics['mean_td_error']:.4f}")

    # 5. Run Reanalyze Pass
    print("\nExecuting Trajectory Reanalyze Pass...")
    trainer.run_reanalyze_pass(num_trajectories=1)
    print("Reanalyze Pass Completed successfully!")

    print("\n================================================================================")
    print("ALL PRIORITIZED MUZERO SYSTEM TESTS EXECUTED WITH ZERO ERRORS!")
    print("================================================================================")
