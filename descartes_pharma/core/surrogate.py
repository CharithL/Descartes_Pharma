"""
DESCARTES-PHARMA: Surrogate model training and hidden state extraction.

Supports LSTM (for HH simulator) and GNN (for molecular datasets).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


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


def train_surrogate(model, train_data, val_data=None, epochs=200,
                    lr=1e-3, device='cpu'):
    """Train a surrogate model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(train_data['inputs'], dtype=torch.float32).to(device)
        targets = torch.tensor(train_data['outputs'], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if val_data is not None and epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_in = torch.tensor(val_data['inputs'], dtype=torch.float32).to(device)
                val_tgt = torch.tensor(val_data['outputs'], dtype=torch.float32).to(device)
                val_loss = criterion(model(val_in), val_tgt).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: train_loss={loss.item():.6f}")

    return model


def extract_hidden_states(model, data, device='cpu', batch_size=16):
    """Extract hidden states from a trained surrogate model.

    Processes in mini-batches to avoid CUDA OOM on large datasets.
    Hidden states are collected on CPU and concatenated at the end.
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
                # h: (batch, timesteps, hidden_dim) -> move to CPU immediately
                hidden_chunks.append(h.cpu())
                del batch, h
                if device != 'cpu':
                    torch.cuda.empty_cache()

        hidden = torch.cat(hidden_chunks, dim=0).numpy()
        return hidden.reshape(-1, model.hidden_dim)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
