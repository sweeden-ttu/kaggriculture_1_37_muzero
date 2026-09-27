
# Factorized Prediction Head

In environments like **Kaggriculture**, a single turn action \\(a_t\\) requires simultaneous decisions across three distinct economic and operational subsystems:

1. **Farmer Actions** (\\(a_{\text{farmer}}\\)): Directional movement, tile interactions (`PLANT`, `WATER`, `HARVEST`, `BUILD`).
2. **Hired Hand Dispatching** (\\(a_{\text{hands}}\\)): Spatial allocations or chore queue priority weighting (e.g., Rescue vs. Harvest vs. Plant).
3. **Market Orders** (\\(a_{\text{market}}\\)): Crop liquidations, seed purchases, and land expansion decisions.

If you attempt to model this joint action space with a single flat Softmax layer, the cross-product combinations explode (\\(|A_{\text{joint}}| = |A_{\text{farmer}}| \times |A_{\text{hands}}| \times |A_{\text{market}}| \approx 10^4\text{--}10^5\\)). This causes severe MCTS search tree sparsity and policy loss instability.

To solve this, the **Prediction Network (\\(f_\theta\\))** is structured with a **Factorized Policy Head**.

---

## 1. Architectural Strategy: Independent vs. Autoregressive Factorization

There are two primary ways to factorize the joint policy distribution \\(p(a \mid s)\\) given the latent state \\(s = g_\theta(s_{k-1}, a_{k-1})\\):

#### Option A: Independent Sub-Head Factorization (Sampled MuZero Paradigm)
Assumes conditional independence between the subsystems given the latent representation \\(s\\):

\\[p(a_{\text{joint}} \mid s) = p_{\text{farmer}}(a_{\text{farmer}} \mid s) \times p_{\text{hands}}(a_{\text{hands}} \mid s) \times p_{\text{market}}(a_{\text{market}} \mid s)\\]

* **Pros**: Highly parallelizable, minimal compute overhead, easy loss logging per subsystem.
* **Cons**: Cannot express intra-turn correlations directly (e.g., selecting `SELL_WHEAT` in the market head when the farmer head just harvested wheat on the exact same turn) except through the shared latent state \\(s\\).

#### Option B: Autoregressive Sequential Factorization (AlphaStar / UniZero Style)
Models intra-turn dependencies sequentially using cross-subsystem conditioning:

\\[p(a_{\text{joint}} \mid s) = p_{\text{market}}(a_{\text{market}} \mid s) \times p_{\text{farmer}}(a_{\text{farmer}} \mid s, \text{emb}(a_{\text{market}})) \times p_{\text{hands}}(a_{\text{hands}} \mid s, \text{emb}(a_{\text{market}}), \text{emb}(a_{\text{farmer}}))\\]

* **Pros**: Captures fine-grained intra-step dependencies (e.g., market expansion dictates hand dispatch allocation on the same frame).
* **Cons**: Sequential forward passes required for sampling during MCTS node expansion.

---

## 2. Integrating Factorized Heads into MCTS (Sampled MuZero)

Because MCTS cannot expand \\(10,000+\\) leaves at every tree node, you integrate factorized heads using **Sampled MCTS**:

1. **Node Expansion**: When expanding a leaf latent state \\(s\\), do not instantiate all actions. Instead, sample a small set of \\(K\\) candidate joint tuples (\\(K \approx 10\text{--}20\\)) by drawing independently or autoregressively from each factorized policy head:
   \\[a^{(k)} = \left(a_{\text{farmer}}^{(k)}, a_{\text{hands}}^{(k)}, a_{\text{market}}^{(k)}\right) \sim p_{\text{factorized}}(\cdot \mid s)\\]
2. **PUCT Search**: MCTS selection and backpropagation execute strictly over the \\(K\\) sampled candidate actions.
3. **Loss Computation**: During network training, the policy loss sums the cross-entropy / KL-divergence losses between each policy head and its corresponding marginal visitation counts (\\(\pi_{\text{MCTS}}\\)) derived from the search tree:

\\[\mathcal{L}_{\text{policy}} = \mathcal{L}_{\text{CE}}\left(\pi_{\text{farmer}}, p_{\text{farmer}}\right) + \mathcal{L}_{\text{CE}}\left(\pi_{\text{hands}}, p_{\text{hands}}\right) + \mathcal{L}_{\text{CE}}\left(\pi_{\text{market}}, p_{\text{market}}\right)\\]

---

## 3. PyTorch Factorized Policy Head Implementation

Here is a clean PyTorch blueprint for \\(f_\theta\\) using independent sub-heads with spatial action masking and candidate joint sampling:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class FactorizedKaggriculturePredictionHead(nn.Module):
    """
    MuZero Prediction Network (f_theta) with Factorized Policy Heads
    and a Scalar/Distributional Value Head.
    """
    def __init__(
        self, 
        latent_dim: int = 256, 
        num_farmer_actions: int = 15,
        num_hand_assignments: int = 32,
        num_market_orders: int = 20,
        value_support_size: int = 601 # Discrete support [-300, 300]
    ):
        super().__init__()
        
        # Shared Residual Back-Bone for the Prediction Head
        self.shared_trunk = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU()
        )
        
        # 1. Factorized Policy Sub-Heads
        self.farmer_head = nn.Linear(latent_dim, num_farmer_actions)
        self.hands_head = nn.Linear(latent_dim, num_hand_assignments)
        self.market_head = nn.Linear(latent_dim, num_market_orders)
        
        # 2. Distributional Value Head (Categorical cross-entropy)
        self.value_head = nn.Linear(latent_dim, value_support_size)

    def forward(self, latent_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: latent_state [Batch, Latent_Dim]
        Output: Logits for each factorized policy sub-head + Value Distribution
        """
        features = self.shared_trunk(latent_state)
        
        return {
            'logits_farmer': self.farmer_head(features),
            'logits_hands': self.hands_head(features),
            'logits_market': self.market_head(features),
            'value_logits': self.value_head(features)
        }

    def sample_joint_actions(
        self, 
        latent_state: torch.Tensor, 
        num_samples: int = 16,
        masks: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples K joint action candidates for Sampled MuZero MCTS expansion.
        Returns:
            sampled_joint_actions: [Batch, K, 3] integer tensor
            joint_log_probs: [Batch, K] prior log probabilities
        """
        preds = self.forward(latent_state)
        
        # Apply action masks if provided
        logits_farmer = preds['logits_farmer']
        logits_hands = preds['logits_hands']
        logits_market = preds['logits_market']
        
        if masks is not None:
            logits_farmer = logits_farmer.masked_fill(masks['farmer'] == 0, -1e9)
            logits_hands = logits_hands.masked_fill(masks['hands'] == 0, -1e9)
            logits_market = logits_market.masked_fill(masks['market'] == 0, -1e9)

        # Categorical distributions for each head
        dist_farmer = torch.distributions.Categorical(logits=logits_farmer)
        dist_hands = torch.distributions.Categorical(logits=logits_hands)
        dist_market = torch.distributions.Categorical(logits=logits_market)
        
        # Sample K candidates independently
        # Shapes: [K, Batch] -> transpose to [Batch, K]
        a_farmer = dist_farmer.sample((num_samples,)).transpose(0, 1)
        a_hands = dist_hands.sample((num_samples,)).transpose(0, 1)
        a_market = dist_market.sample((num_samples,)).transpose(0, 1)
        
        # Combine into Joint Action Candidate Tensor [Batch, K, 3]
        joint_actions = torch.stack([a_farmer, a_hands, a_market], dim=-1)
        
        # Compute joint log prior: log p_joint = log p1 + log p2 + log p3
        log_prob = (
            dist_farmer.log_prob(a_farmer) + 
            dist_hands.log_prob(a_hands) + 
            dist_market.log_prob(a_market)
        )
        
        return joint_actions, log_prob
```

---

## 4. Key Implementation Recommendations

1. **Action Masking**: Apply subsystem masks directly to the logits prior to sampling during MCTS expansion. For example, if no market slots are free, mask out all `SELL` market order logits to \\(-\infty\\).
2. **Entropy Regularization**: Add entropy bonuses to each sub-head loss term (\\(\mathcal{H}(p_m)\\)) during training to prevent one subsystem (e.g., market orders) from collapsing into premature deterministic behavior before the farmer sub-head finishes learning spatial movement routes.
3. **Autoregressive Conditioning**: If market orders heavily dictate farmer actions within the exact same turn, pass an embedding of the selected \\(a_{\text{market}}\\) into a small GRU/MLP layer before computing \\(a_{\text{farmer}}\\) logits.

