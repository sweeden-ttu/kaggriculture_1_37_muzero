# Kaggriculture Observation Encoder

To handle progressive land unlocks cleanly in a MuZero policy without breaking neural network architectures or causing tensor dimension mismatches, the standard approach is **Fixed Maximum-Grid Feature Stacking with Spatial Masking**.

In **Kaggriculture**, the full farm map is at most a **10×10 grid** composed of four 5×5 quadrants (`NW`, `NE`, `SW`, `SE`), starting with only `NW` unlocked. Changing tensor shapes dynamically during an episode as land is purchased causes severe issues for Convolutional / ResNet backbones and MCTS rollouts. 

Here is the blueprint for structuring the observation space, spatial channels, action masks, and representation network (\\(h_\theta\\)).

---

### 1. Fixed Maximum-Grid Spatial Tensor (\\(10 \times 10 \times C\\))

Always initialize observation tensors at the maximum possible map resolution (**10×10** spatial dimensions) from turn 0. Rather than resizing the grid when a new quadrant is bought, use explicit **mask channels** to encode quadrant availability.

```
Observation Tensor: [Batch, Channels, Height=10, Width=10]
```

#### Recommended Channel Layout (\\(C \approx 25\text{--}35\\) channels)

| Channel Group | Channel Count | Feature Description |
| :--- | :--- | :--- |
| **Land Mask** | 1 | **Quadrant Availability**: `1.0` for unlocked tiles, `0.0` for `LOCKED` tiles. |
| **Tile Entities (One-Hot)** | 8 | One-hot binary planes indicating tile state: `Empty`, `Weed`, `Plant` (Wheat, Carrot, Tomato, Strawberry, Melon), `Structure` (Coop/Pasture). |
| **Crop Growth & Health** | 5 | Continuous/Normalized features per tile: `Watered Today` (0 or 1), `Consecutive Unwatered` (\\(n/2\\)), `Yield Available` (\\(y/\text{max\_yield}\\)), `Lifespan Decay` (\\(t/\text{max\_lifespan}\\)), `Fertilized Until`. |
| **Livestock States** | 4 | `Animal Present` (Goose/Cow/Sheep one-hot), `Fed Today` (0 or 1), `Consecutive Unfed` (\\(n/2\\)), `Pending Care Bonus`. |
| **Unit Positions** | 2 | Spatial density/one-hot maps: `Farmer Position` (\\(1\\) at farmer \\(x, y\\)), `Hired Hands Positions` (count of hands on tile). |
| **Global Context (Broadcast)** | 8–12 | Global environment states tiled across the entire 10×10 grid: `Normalized Day` (\\(d/30\\)), `Normalized Hour` (\\(h/24\\)), `Log Cash` (\\(\log_{10}(\text{Cash})\\)), `Shed Capacity Ratio` (\\(S/100\\)), and `Crop Market Prices` (\\(P_{\text{crop}} / P_0\\)). |

---

### 2. Action Masking in MCTS & Policy Prediction

Because the observation tensor encompasses the entire 10×10 map, the policy head (\\(f_\theta\\)) will output action logits across all spatial positions. To prevent MuZero's MCTS search tree from exploring invalid actions on locked land:

1. **Construct a Legal Action Mask**: At each real environment step \\(t\\), compute a boolean mask \\(M(a) \in \{0, 1\}^{|A|}\\).
2. **Mask Out Locked Quadrants**: Set \\(M(a) = 0\\) for any spatial tile action (`PLANT`, `WATER`, `DIG`, `BUILD_COOP`, etc.) targeted at a tile where the `Land Mask` channel is `0.0`.
3. **Logit Suppression**: Prior to running Softmax in the prediction network \\(f_\theta\\) and during MCTS P-UCT expansion, force illegal action logits to \\(-\infty\\):
   \\[p(a \mid s) = \text{Softmax}(\text{Logits}(a) + \log M(a))\\]

---

### 3. Representation Network Integration (\\(h_\theta\\))

The representation network maps raw observation stacks into the abstract latent state \\(s^0 = h_\theta(o_{1:t})\\). 

* **Preserving Resolution**: For small grid sizes like \\(10 \times 10\\), avoid heavy downsampling convolutions (such as stride-2 layers used in Atari). Use \\(3 \times 3\\) convolutions with `padding=1` and ResNet blocks to preserve the \\(10 \times 10\\) spatial layout in the hidden state tensor \\(s^0 \in \mathbb{R}^{10 \times 10 \times C_{\text{hidden}}}\\).
* **Dynamics Network Unrolling (\\(g_\theta\\))**: When unrolling \\(g_\theta(s^{k-1}, a^k)\\), encode the selected action \\(a^k\\) as a \\(10 \times 10 \times C_{\text{action}}\\) spatial feature plane (e.g., a one-hot grid showing where a plant or water action was applied) and concatenate it with the latent state \\(s^{k-1}\\).

---

### 4. Preventing Latent Shift When Quadrants Unlock

When an agent purchases a new land quadrant (e.g., unlocking `NE` for \$1,000):

1. **Abrupt Distribution Shift**: The `Land Mask` channel for those 25 tiles flips from `0.0` to `1.0` in a single step.
2. **Self-Supervised Regularization (SSL)**: To prevent the representation network \\(h_\theta\\) and dynamics network \\(g_\theta\\) from diverging or suffering value instability after a land unlock, train with **Self-Supervised Consistency Loss** (e.g., as implemented in UniZero or EfficientZero). 
3. **Consistency Objective**: Penalize the cosine distance between the predicted latent state \\(s^k\\) from \\(g_\theta\\) and the directly encoded observation embedding \\(z_{t+k} = h_\theta(o_{t+k})\\):
   \\[\mathcal{L}_{\text{consistency}} = 1 - \frac{s^k \cdot z_{t+k}}{\|s^k\|_2 \|z_{t+k}\|_2}\\]
   This keeps latent dynamics aligned across both locked and newly unlocked territory.

---

### PyTorch Observation Wrapper Reference

```python
import torch
import torch.nn as nn
import numpy as np

class KaggricultureObservationEncoder:
    def __init__(self, board_size=10):
        self.board_size = board_size
        # 1 (mask) + 8 (entities) + 5 (crop health) + 4 (animals) + 2 (units) + 8 (global)
        self.num_channels = 28 

    def encode(self, obs: dict) -> torch.Tensor:
        tensor = np.zeros((self.num_channels, self.board_size, self.board_size), dtype=np.float32)
        
        # 1. Unlocked Quadrant Mask Channel
        unlocked = obs['farms'][obs['player']]['unlocked_quadrants']
        quad_coords = {
            'NW': (slice(0, 5), slice(0, 5)),
            'NE': (slice(0, 5), slice(5, 10)),
            'SW': (slice(5, 10), slice(0, 5)),
            'SE': (slice(5, 10), slice(5, 10))
        }
        for q in unlocked:
            r, c = quad_coords[q]
            tensor[0, r, c] = 1.0  # Unlocked tiles set to 1.0
            
        # 2. Populate Tile Entity Channels (Channels 1..8)
        tiles = obs['farms'][obs['player']]['tiles']
        for y in range(10):
            for x in range(10):
                tile = tiles[y][x]
                if tile == "LOCKED" or tile is None:
                    continue
                # Map tile dicts (PLANT, WEED, COOP, PASTURE) into one-hot channels...
                
        # 3. Broadcast Global Features (Day, Cash, Market Prices) across 10x10
        day_norm = obs['day'] / 30.0
        cash_log = np.log10(max(1.0, obs['farms'][obs['player']]['money'])) / 5.0
        tensor[20, :, :] = day_norm
        tensor[21, :, :] = cash_log
        
        return torch.from_numpy(tensor).unsqueeze(0)  # Shape:
```
