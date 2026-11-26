"""Phase 5 model definitions: HeteroGNN and no-graph MLP control."""

from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

VOCAB_TYPES = ["code", "category", "chapter", "medication"]

HIERARCHY = {
    ("code", "in_category", "category"),
    ("category", "has_code", "code"),
    ("category", "in_chapter", "chapter"),
    ("chapter", "has_category", "category"),
}

VISITS = {("encounter", "next_visit", "encounter")}


class HeteroGNN(nn.Module):
    """3-hop heterogeneous GraphSAGE model over encounters, ICD-9 hierarchy, and medications."""

    def __init__(
        self,
        data: HeteroData,
        hidden: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        edge_types: Optional[Iterable[tuple[str, str, str]]] = None,
    ):
        super().__init__()
        self.edge_types = list(edge_types if edge_types is not None else data.edge_types)
        self.dropout = dropout
        self.enc_in = nn.Linear(data["encounter"].x.size(1), hidden)
        self.emb = nn.ModuleDict(
            {t: nn.Embedding(data[t].num_nodes, hidden) for t in VOCAB_TYPES}
        )
        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {et: SAGEConv(hidden, hidden) for et in self.edge_types},
                    aggr="sum",
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x = {"encounter": self.enc_in(data["encounter"].x)}
        for t in VOCAB_TYPES:
            x[t] = self.emb[t].weight

        edge_index = {et: data[et].edge_index for et in self.edge_types}

        for conv in self.convs:
            out = conv(x, edge_index)
            out = {
                k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                for k, v in out.items()
            }
            # Preserve node types that received no incoming messages in this layer (e.g., medication)
            x = {**x, **out}

        return self.head(x["encounter"]).squeeze(-1)


class MLP(nn.Module):
    """Control model: same encounter features and prediction head, with no graph message passing."""

    def __init__(self, data: HeteroData, hidden: int = 64, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        d_in = data["encounter"].x.size(1)
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(d_in, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            d_in = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        return self.head(self.body(data["encounter"].x)).squeeze(-1)