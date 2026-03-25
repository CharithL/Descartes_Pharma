"""
DESCARTES-PHARMA: Surrogate model training and hidden state extraction.

Supports LSTM (for HH simulator) and GNN (for molecular datasets).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple


class LSTMSurrogate(nn.Module):
    """LSTM surrogate for time-series data (HH simulator)."""

    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_hidden=False):
        h, (hn, cn) = self.lstm(x)
        out = self.fc(h)
        if return_hidden:
            return out, h
        return out


class GNNSurrogate(nn.Module):
    """Graph Neural Network surrogate for molecular property prediction."""

    def __init__(self, node_feat_dim=9, hidden_dim=128, output_dim=1,
                 n_layers=4, readout='mean', dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.readout = readout

        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, node_features, adjacency, batch_idx=None,
                return_embeddings=False):
        x = self.node_embed(node_features)
        x = torch.relu(x)

        layer_embeddings = [x]
        for conv, bn in zip(self.convs, self.bns):
            if adjacency.is_sparse:
                h = torch.sparse.mm(adjacency, x)
            else:
                h = torch.matmul(adjacency, x)
            h = conv(h)
            h = bn(h)
            h = torch.relu(h)
            h = self.dropout(h)
            x = x + h
            layer_embeddings.append(x)

        if batch_idx is not None:
            n_graphs = batch_idx.max().item() + 1
            graph_embed = torch.zeros(n_graphs, self.hidden_dim, device=x.device)
            graph_embed.scatter_reduce_(
                0, batch_idx.unsqueeze(1).expand_as(x), x, reduce='mean')
        else:
            graph_embed = x.mean(dim=0, keepdim=True)

        out = self.fc(graph_embed)

        if return_embeddings:
            return out, graph_embed, layer_embeddings
        return out


def normalize_data(data: Dict) -> Tuple[Dict, Dict]:
    """Z-score normalize inputs and outputs. Returns normalized data and stats."""
    stats = {}
    normalized = {}

    for key in ['inputs', 'outputs']:
        arr = data[key]
        mean = arr.mean()
        std = arr.std() + 1e-8
        normalized[key] = (arr - mean) / std
        stats[key] = {'mean': mean, 'std': std}

    # Copy non-numeric fields
    for key in data:
        if key not in ['inputs', 'outputs']:
            normalized[key] = data[key]

    return normalized, stats


def train_surrogate(model, train_data, val_data=None, epochs=300,
                    lr=1e-3, device='cpu', batch_size=32):
    """Train a surrogate model with mini-batching, LR scheduling, and best-model checkpoint."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    train_inputs = torch.tensor(train_data['inputs'], dtype=torch.float32)
    train_targets = torch.tensor(train_data['outputs'], dtype=torch.float32)
    n_train = train_inputs.shape[0]

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            batch_in = train_inputs[idx].to(device)
            batch_tgt = train_targets[idx].to(device)

            optimizer.zero_grad()
            outputs = model(batch_in)
            loss = criterion(outputs, batch_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_loss = avg_train_loss
        if val_data is not None and epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_in = torch.tensor(val_data['inputs'], dtype=torch.float32).to(device)
                val_tgt = torch.tensor(val_data['outputs'], dtype=torch.float32).to(device)
                val_loss = criterion(model(val_in), val_tgt).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step(val_loss)

        if epoch % 50 == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: train={avg_train_loss:.6f} "
                  f"val={val_loss:.6f} lr={cur_lr:.6f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.6f})")

    return model


def extract_hidden_states(model, data, device='cpu', batch_size=16):
    """Extract hidden states from a trained surrogate model.

    Processes in mini-batches to avoid CUDA OOM on large datasets.
    Hidden states are collected on CPU and concatenated at the end.
    Returns 3D array: (n_trials, timesteps, hidden_dim).
    """
    model.to(device)
    model.eval()

    if isinstance(model, LSTMSurrogate):
        all_inputs = data['inputs']  # (n_trials, timesteps, feat)
        n_trials = all_inputs.shape[0]
        hidden_chunks = []

        with torch.no_grad():
            for start in range(0, n_trials, batch_size):
                end = min(start + batch_size, n_trials)
                batch = torch.tensor(
                    all_inputs[start:end], dtype=torch.float32
                ).to(device)
                _, h = model(batch, return_hidden=True)
                hidden_chunks.append(h.cpu())
                del batch, h
                if device != 'cpu':
                    torch.cuda.empty_cache()

        hidden = torch.cat(hidden_chunks, dim=0).numpy()
        return hidden  # (n_trials, timesteps, hidden_dim)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
