# API Reference

## Core Modules

### `muzero.observation_encoder`

#### `KaggricultureObservationEncoder`
```python
class KaggricultureObservationEncoder:
    def __init__(self, board_size: int = 10, num_channels: int = 28)
    
    def encode(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Encode observation dict to [1, 28, 10, 10] tensor."""
    
    def get_action_mask(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Return legal action masks: {'farmer': [15], 'hands': [32], 'market': [20]}."""
```

#### `encode_spatial_observation(obs: Dict[str, Any]) -> torch.Tensor`
Convenience function returning `[28, 10, 10]` (no batch dim).

---

### `muzero.chassis`

#### `ResNetBlock2D(channels: int)`
Pre-activation ResNet block preserving 10×10 spatial dims.

#### `SpatialRepresentationNetwork` (h_θ)
```python
class SpatialRepresentationNetwork:
    def __init__(self, in_channels: int = 28, latent_channels: int = 64, num_blocks: int = 3)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Input: [B, 28, 10, 10] → Output: [B, 64, 10, 10] (L2 normalized)"""
```

#### `SpatialDynamicsNetwork` (g_θ)
```python
class SpatialDynamicsNetwork:
    def __init__(self, latent_channels: int = 64, action_channels: int = 16, 
                 num_blocks: int = 3, support_size: int = 601)
    
    def forward(self, s_prev: torch.Tensor, action_plane: torch.Tensor) 
        -> Tuple[torch.Tensor, torch.Tensor]:
        """Input: s_prev [B, 64, 10, 10], action [B, 16, 10, 10]
           Output: s_next [B, 64, 10, 10], reward_logits [B, 601]"""
```

#### `SpatialPredictionNetwork` (f_θ) = `FactorizedKaggriculturePredictionHead`
```python
class SpatialPredictionNetwork:
    def __init__(self, latent_channels: int = 64, num_farmer_actions: int = 15,
                 num_hand_assignments: int = 32, num_market_orders: int = 20,
                 support_size: int = 601)
    
    def forward(self, s_k: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns: {
            'logits_farmer': [B, 15],
            'logits_hands': [B, 32],
            'logits_market': [B, 20],
            'value_logits': [B, 601]
        }"""
    
    def sample_joint_actions(self, s_k: torch.Tensor, num_samples: int = 16,
                            masks: Optional[Dict[str, torch.Tensor]] = None)
        -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: joint_actions [B, K, 3], log_probs [B, K]"""
```

#### `KaggricultureMuZeroChassis`
```python
class KaggricultureMuZeroChassis:
    def __init__(self, obs_channels: int = 28, latent_channels: int = 64,
                 action_channels: int = 16, num_farmer_actions: int = 15,
                 num_hand_assignments: int = 32, num_market_orders: int = 20,
                 support_size: int = 601)
    
    def initial_inference(self, obs: torch.Tensor) 
        -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Root inference: obs → (s_0, preds_dict, value_scalar)"""
    
    def recurrent_inference(self, s_prev: torch.Tensor, action_plane: torch.Tensor)
        -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Tree inference: (s, action) → (s_next, reward, preds_dict, value_scalar)"""
    
    def _logits_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert support logits to scalar via softmax expectation."""
```

---

### `muzero.chassis` — SimSiam Consistency

#### `SimSiamProjectionPredictionHead`
```python
class SimSiamProjectionPredictionHead:
    def __init__(self, latent_dim: int = 64, proj_dim: int = 64, pred_dim: int = 32)
    
    def project(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Global avg pool if spatial → proj_head → [B, proj_dim]"""
    
    def predict(self, projected_state: torch.Tensor) -> torch.Tensor:
        """pred_head → [B, proj_dim]"""
```

#### `SimSiamConsistencyLoss`
```python
class SimSiamConsistencyLoss:
    def __init__(self, eps: float = 1e-8)
    
    def forward(self, unrolled_latent: torch.Tensor, 
                target_observation_embedding: torch.Tensor,
                simsiam_head: SimSiamProjectionPredictionHead) -> torch.Tensor:
        """Negative cosine similarity with stop-gradient on target.
           Returns scalar loss ∈ [-1, 0]."""
```

#### `compute_muzero_unroll_loss_with_consistency(...)`
Full K-step loss computation with policy, value, reward, consistency.

---

### `muzero.mcts`

#### `MinMaxStats`
```python
class MinMaxStats:
    def __init__(self, minimum: Optional[float] = None, maximum: Optional[float] = None,
                 min_value: Optional[float] = None, max_value: Optional[float] = None)
    
    def update(self, value: float) -> None
    def normalize(self, value: float, eps: float = 0.0) -> float
    def clear(self) -> None
    def is_valid(self) -> bool
```

#### `MCTSNode`
```python
class MCTSNode:
    def __init__(self, prior: float = 0.0, action_tuple: Optional[Tuple[int, int, int]] = None)
    
    prior: float
    action_tuple: Optional[Tuple[int, int, int]]
    visit_count: int
    value_sum: float
    reward: float
    latent_state: Optional[torch.Tensor]
    children: Dict[Tuple[int, int, int], 'MCTSNode']
    
    @property
    def is_expanded(self) -> bool
    @property
    def value(self) -> float  # value_sum / visit_count
```

#### `SampledMuZeroMCTS`
```python
class SampledMuZeroMCTS:
    def __init__(self, chassis: KaggricultureMuZeroChassis,
                 c1: float = 1.25, c2: float = 19652.0,
                 discount: float = 0.997, num_samples: int = 8,
                 num_simulations: int = 25, dirichlet_alpha: float = 0.25,
                 exploration_fraction: float = 0.25)
    
    @torch.no_grad()
    def search(self, obs_tensor: torch.Tensor, add_dirichlet_noise: bool = True,
               masks: Optional[Dict[str, torch.Tensor]] = None)
        -> Tuple[Tuple[int, int, int], Dict[Tuple[int, int, int], float], float]:
        """Returns: (best_action, visit_probabilities, root_value)"""
```

---

### `muzero.buffer`

#### `GameTrajectory`
```python
class GameTrajectory:
    observations: List[np.ndarray]      # [28, 10, 10]
    actions: List[Tuple[int, int, int]] # (farmer, hands, market)
    rewards: List[float]
    policies: List[Dict[Tuple, float]]  # MCTS visit distributions
    values: List[float]                 # MCTS root values
    priorities: List[float]             # TD-error for PER
    
    def __len__(self) -> int
    def append_step(self, obs, action, reward, policy_dict, value, priority=1.0)
    def finalize(self, final_obs: np.ndarray)
```

#### `MuZeroTrajectoryBuffer`
```python
class MuZeroTrajectoryBuffer:
    def __init__(self, max_trajectories: int = 1000, discount: float = 0.997,
                 n_steps: int = 5, k_steps: int = 5,
                 num_farmer_actions: int = 15, num_hand_assignments: int = 32,
                 num_market_orders: int = 20)
    
    def save_trajectory(self, trajectory: GameTrajectory)
    def sample_k_step_batch(self, batch_size: int = 16, k_steps: int = 5)
        -> Dict[str, torch.Tensor]
    def reanalyze_trajectory(self, traj_idx: int, mcts_engine, device: torch.device)
```

#### `PrioritizedMuZeroBuffer` (extends MuZeroTrajectoryBuffer)
```python
class PrioritizedMuZeroBuffer(MuZeroTrajectoryBuffer):
    def __init__(self, ..., alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 100000, eps: float = 1e-5)
    
    def update_beta(self, current_frame: int) -> float
    def sample_prioritized_k_step_batch(self, batch_size: int = 8, k_steps: int = 5)
        -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]], torch.Tensor]
    def update_slice_priorities(self, slice_keys: List[Tuple[int, int]], td_errors: np.ndarray)
```

---

### `muzero.trainer`

#### `PrioritizedMuZeroTrainer`
```python
class PrioritizedMuZeroTrainer:
    def __init__(self, online_model: KaggricultureMuZeroChassis,
                 simsiam_head: SimSiamProjectionPredictionHead,
                 buffer: PrioritizedMuZeroBuffer,
                 mcts_engine: SampledMuZeroMCTS,
                 lr: float = 1e-3, ema_tau: float = 0.995,
                 consistency_weight: float = 0.25, grad_clip: float = 5.0,
                 device: str = "cpu")
    
    def train_step(self, batch_size: int = 4, k_steps: int = 3, 
                   global_step: int = 0) -> Dict[str, float]:
        """Returns: {loss_total, loss_policy, loss_value, loss_reward, 
                      loss_consistency, beta, mean_td_error}"""
    
    def run_reanalyze_pass(self, num_trajectories: int = 1)
    def update_target_network(self)
```

#### `MuZeroTrainer` (Uniform sampling)
```python
class MuZeroTrainer:
    def __init__(self, online_model: KaggricultureMuZeroChassis,
                 simsiam_head: SimSiamProjectionPredictionHead,
                 buffer: MuZeroTrajectoryBuffer,
                 mcts_engine: SampledMuZeroMCTS,
                 lr: float = 3e-4, ema_tau: float = 0.995,
                 consistency_weight: float = 0.25, grad_clip: float = 5.0,
                 device: str = "cpu")
    
    def train_step(self, batch_size: int = 4, k_steps: int = 3) -> Dict[str, float]
    def run_reanalyze_pass(self, num_trajectories: int = 1)
```

---

### `muzero.lora`

```python
def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32, dropout: float = 0.1)
def merge_lora(model: nn.Module)
def save_lora_adapters(model: nn.Module, path: str)
def load_lora_adapters(model: nn.Module, path: str)
def lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]
def has_lora(model: nn.Module) -> bool
def export_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]

DEFAULT_LORA_RANK = 16
```

---

### `muzero.ppo`

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256)

class PPOBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, gamma: float = 0.99, lam: float = 0.95)

def ppo_rollout_episode(env, actor_critic, max_steps: int = 720) -> PPOBuffer
def ppo_rollout_batch(env, actor_critic, num_episodes: int) -> PPOBuffer
def train_ppo(actor_critic, optimizer, buffer, epochs: int = 4, 
              clip_eps: float = 0.2, ...) -> Dict[str, float]

def save_ppo_checkpoint(actor_critic, optimizer, path: str)
def load_ppo_checkpoint(path: str) -> Tuple[nn.Module, optim.Optimizer]
def resolve_ppo_checkpoint() -> str

class PPOHyperParams: ...
DEFAULT_PPO_CHECKPOINT = "ppo_checkpoint.pt"
```

---

### `muzero.legacy_macro` (MLP Macro MuZero)

```python
class MuZeroNetwork(nn.Module): ...
class RepresentationNetwork(nn.Module): ...
class DynamicsNetwork(nn.Module): ...
class PredictionNetwork(nn.Module): ...
class LatentMacroOptionMCTS: ...
class MuZeroPlanner: ...
class PrioritizedReplayBuffer: ...

def muzero_bootstrap_loss(...) -> loss
def muzero_consistency_loss(...) -> loss
def muzero_k_step_unroll_loss(...) -> loss
def muzero_mcts_distill_loss(...) -> loss
def compute_target_values(...) -> targets
def reanalyze_trajectory_slice(...) -> targets
def create_target_network(model) -> model
def update_target_network(online, target, tau=0.995)
def scale_gradient(tensor, scale)
def load_muzero_checkpoint(path) -> state_dict
def save_muzero_checkpoint(model, path)
def resolve_muzero_checkpoint() -> path

DEFAULT_MUZERO_CHECKPOINT = "muzero_checkpoints.pt"
```

---

### `evaluation.league`

#### `LeagueTracker`
```python
class LeagueTracker:
    def __init__(self, smoothing: float = 1.0)
    def add_checkpoint(self, checkpoint_id: str)
    def record_match_outcome(self, agent_id: str, opponent_id: str, outcome: float)
    def get_win_rate(self, agent_id: str, opponent_id: str) -> float
```

#### `PFSPSampler`
```python
class PFSPSampler:
    def __init__(self, league: LeagueTracker)
    def sample_opponent(self, active_agent_id: str, 
                        mode: str = "hard", power: float = 2.0) -> str
```

#### `select_matchmaking_opponent(agent_role, active_agent_id, pfsp) -> Tuple[str, str]`
Returns (opponent_id, mode_string) based on role ratios.

---

### `pipelines.train_muzero`

Main training entry point. Run with:
```bash
python pipelines/train_muzero.py --help
```

---

### `hybrid_chassis`

#### `Chassis` (Reactive)
```python
class Chassis:
    def __init__(self, routes, router=None, settings=None, opponent_plan=None)
    def act(self, observation, configuration=None) -> Dict
```

#### `make_agent(routes, router=None, opponent_plan=None, **settings) -> Callable`
Returns Kaggle-compatible agent function.

---

## Constants (`muzero.spatial_constants`)

```python
OBS_CHANNELS = 28
LATENT_CHANNELS = 64
ACTION_CHANNELS = 16
NUM_FARMER_ACTIONS = 15
NUM_HAND_ASSIGNMENTS = 32
NUM_MARKET_ORDERS = 20
SUPPORT_SIZE = 601
SUPPORT_B = 300
DISCOUNT = 0.997

DEFAULT_SPATIAL_CHECKPOINT = "spatial_muzero.pt"

def joint_action_to_plane(action_tuple, batch_size=1, device='cpu') -> torch.Tensor
def soft_cross_entropy(logits, targets) -> torch.Tensor
```

---

## Types (`muzero.types`)

```python
class DiscreteSupport:
    def __init__(self, size: int = 601, bound: int = 300)
    def scalar_to_support(self, x: torch.Tensor) -> torch.Tensor  # Target projection
    def support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor  # Expectation

class MacroOption(Enum): ...
class MacroAction: ...
class MacroState: ...

OBS_DIM = 32
HIDDEN_DIM = 256
NUM_MACRO_OPTIONS = 8
REWARD_SCALE = 1/1000
POLICY_FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
LAND_PRICES = [0, 1000, 2000, 4000]
LAND_ORDER = ["NW", "NE", "SW", "SE"]
LAND_BUFFERS = {"NW": 0, "NE": 1000, "SW": 2500, "SE": 5500}
DEFAULT_PRICES = {...}
MACRO_OPTION_NAMES = [...]
```

---

## Encoding (`muzero.encoding`)

```python
def encode_observation(obs) -> np.ndarray  # [32] macro
def encode_macro_state(state) -> np.ndarray
def decode_macro_state(vec) -> MacroState
def legal_macro_options(state) -> List[MacroOption]
def wealth(state) -> float
```