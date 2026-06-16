# model/behaviour_transformer.py
# Exact architecture that produced best_combine_model_purchase_prediction.joblib.
# DO NOT change any dimension – the saved state_dict must match exactly.

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (not learned)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BehaviourTransformer(nn.Module):
    """
    Transformer encoder with a learnable [CLS] token.

    Architecture verified from the saved state_dict keys:
      proj   : Sequential( Linear(n_feats→d_model), LayerNorm(d_model) )
      encoder: TransformerEncoder with final LayerNorm
      head   : Sequential( Linear(d_model→64), GELU, Dropout, Linear(64→1) )

    Input : (B, T, n_feats)  – sequence of 5-second behavioural windows
    Output: (B, 1)            – raw purchase logit (apply sigmoid for probability)

    Loading from joblib state_dict (NOT torch.save):
        model = BehaviourTransformer(...)
        model.load_state_dict(joblib.load("best_combine_model_purchase_prediction.joblib"))
        model.eval()
    """

    def __init__(
        self,
        n_feats: int   = 42,
        max_seq: int   = 24,
        d_model: int   = 128,
        n_heads: int   = 4,
        n_layers: int  = 4,
        d_ff: int      = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        # proj.0 = Linear, proj.1 = LayerNorm  (verified from state_dict)
        self.proj = nn.Sequential(
            nn.Linear(n_feats, d_model),
            nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc   = SinusoidalPE(d_model, max_len=max_seq + 1)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # encoder has a final LayerNorm (encoder.norm.* in state_dict)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        # head.0=Linear(128→64), head.1=GELU, head.2=Dropout, head.3=Linear(64→1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.size(0)
        x = self.proj(x)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        if key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.head(x[:, 0])
