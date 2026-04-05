"""
Search Policy Network -- the ONLY learned component.

Like AlphaGo's policy network: it doesn't learn the rules of Go,
it learns which moves are worth trying. Our network doesn't learn
physics -- Vina handles that. It learns which pose adjustments
are likely to improve binding.

Its hidden states are probed by DESCARTES (Module 6) to determine
if it learned genuine binding intuition or is a pharmaceutical zombie.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SearchPolicyNetwork(nn.Module):
    """
    GRU-based policy network for docking search guidance.

    Input: [pocket_features | ligand_features | interaction_features
            | score_history | current_vina_score]

    Output:
        - policy: action probabilities (which pose adjustment)
        - value: predicted final binding energy

    Hidden states are saved for DESCARTES probing.
    """

    def __init__(
        self,
        pocket_dim: int = 40,       # From PocketFeatures.to_feature_vector()
        ligand_dim: int = 16,        # From LigandFeatures.to_feature_vector()
        interaction_dim: int = 20,   # From interaction features
        score_history_len: int = 10, # Last N Vina scores
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_actions: int = 22,         # 12 base + 10 torsion (max)
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pocket_dim = pocket_dim
        self.ligand_dim = ligand_dim
        self.interaction_dim = interaction_dim
        self.score_history_len = score_history_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_actions = n_actions

        # Input projection
        total_input = pocket_dim + ligand_dim + interaction_dim + score_history_len + 1
        self.total_input = total_input

        self.input_proj = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GRU core -- sequential decision making
        # The hidden state accumulates information across pose adjustments
        # THIS IS WHAT DESCARTES PROBES
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )

        # Policy head: which action to take next
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # Value head: how good is the current state
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Storage for hidden states (for DESCARTES probing)
        self._hidden_states_log: list = []
        self._logging_enabled: bool = False

    def forward(self, x: torch.Tensor, h: torch.Tensor = None):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
            h: (n_layers, batch, hidden_dim) previous hidden state

        Returns:
            policy_logits: (batch, n_actions) raw logits
            value: (batch, 1) state value estimate
            h_new: (n_layers, batch, hidden_dim) new hidden state
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq dim

        # Project input
        projected = self.input_proj(x)

        # GRU forward
        if h is None:
            h = torch.zeros(
                self.n_layers, x.size(0), self.hidden_dim, device=x.device
            )

        gru_out, h_new = self.gru(projected, h)

        # Use last timestep output
        last_hidden = gru_out[:, -1, :]

        # Log hidden states for DESCARTES probing
        if self._logging_enabled:
            self._hidden_states_log.append(
                h_new[-1].detach().cpu().numpy()  # Last layer hidden state
            )

        # Heads
        policy_logits = self.policy_head(last_hidden)
        value = self.value_head(last_hidden)

        return policy_logits, value, h_new

    def select_action(self, x: torch.Tensor, h: torch.Tensor = None,
                      temperature: float = 1.0):
        """
        Select an action using the policy (with temperature for exploration).

        Args:
            x: (1, input_dim) current state -- or (input_dim,) which gets unsqueezed
            h: hidden state
            temperature: >1 = more exploration, <1 = more exploitation

        Returns:
            action: int, selected action index
            log_prob: float, log probability of the action
            value: float, state value estimate
            h_new: new hidden state
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        policy_logits, value, h_new = self.forward(x, h)

        # Temperature-scaled softmax
        probs = F.softmax(policy_logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, value.squeeze(), h_new

    # === DESCARTES PROBING INTERFACE ===

    def enable_logging(self):
        """Start logging hidden states for DESCARTES probing."""
        self._logging_enabled = True
        self._hidden_states_log = []

    def disable_logging(self):
        """Stop logging hidden states."""
        self._logging_enabled = False

    def get_hidden_states(self) -> np.ndarray:
        """
        Get logged hidden states for DESCARTES probing.

        Returns: (n_timesteps, hidden_dim) array
        """
        if not self._hidden_states_log:
            return np.array([])
        return np.concatenate(self._hidden_states_log, axis=0)

    def clear_hidden_log(self):
        """Clear the hidden state log."""
        self._hidden_states_log = []
