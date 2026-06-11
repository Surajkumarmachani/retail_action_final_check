# model/predictor.py
# Loads model weights (from joblib state_dict) + StandardScaler, runs inference.

import logging
from typing import List

import joblib
import numpy as np
import torch

from constants import (
    D_FF, D_MODEL, DROPOUT, FEATURE_COLS,
    MAX_SEQ, N_FEATS, N_HEADS, N_LAYERS, PURCHASE_THRESHOLD,
)
from model.behaviour_transformer import BehaviourTransformer

logger = logging.getLogger(__name__)


class PurchasePredictor:
    """
    Stateless inference wrapper.  Load once at server startup; call predict()
    for every track whenever a new 5-second window completes.

    Critical loading sequence (model is joblib state_dict, NOT a sklearn estimator):
        state = joblib.load("transformer_behaviour_best.joblib")
        model.load_state_dict(state)          # ← this, not model = joblib.load(...)
        model.eval()
    """

    def __init__(self, model_path: str, scaler_path: str, device: str = "cpu"):
        self.device    = torch.device(device)
        self.threshold = PURCHASE_THRESHOLD

        # Reconstruct the architecture, then fill weights from the joblib state_dict
        self.model = BehaviourTransformer(
            n_feats=N_FEATS, max_seq=MAX_SEQ, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
        )
        state_dict = joblib.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("BehaviourTransformer loaded from %s", model_path)

        # StandardScaler fitted on training data – REQUIRED for valid predictions
        self.scaler = joblib.load(scaler_path)
        logger.info("StandardScaler loaded from %s", scaler_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, windows: List[dict]) -> float:
        """
        windows : list of raw (unscaled) feature dicts, one per completed 5-s window.
                  Missing keys are filled with 0 (matches fillna(0) in training).
        Returns : purchase probability in [0, 1].

        Called only when a new window has been flushed – typically every 5 seconds
        per track.  Fast enough (<5 ms on CPU for T≤24) to run on the event loop,
        but can be wrapped in run_in_executor if latency becomes an issue.
        """
        if not windows:
            return 0.0

        # Build (T, 41) float array in the fixed column order
        arr = np.array(
            [[w.get(col, 0.0) for col in FEATURE_COLS] for w in windows],
            dtype=np.float32,
        )

        # Apply the training-time StandardScaler
        arr = self.scaler.transform(arr).astype(np.float32)

        T = arr.shape[0]
        if T > MAX_SEQ:
            arr = arr[-MAX_SEQ:]    # keep most-recent windows (sliding window)
            T   = MAX_SEQ

        pad_len = MAX_SEQ - T
        if pad_len:
            padding = np.zeros((pad_len, N_FEATS), dtype=np.float32)
            arr = np.concatenate([arr, padding], axis=0)

        x    = torch.tensor(arr).unsqueeze(0).to(self.device)   # (1, MAX_SEQ, 41)
        mask = torch.zeros(1, MAX_SEQ, dtype=torch.bool, device=self.device)
        if pad_len:
            mask[0, T:] = True                                   # mark padding

        with torch.no_grad():
            logit = self.model(x, key_padding_mask=mask)
            prob  = torch.sigmoid(logit).item()

        return float(prob)

    def verdict(self, prob: float) -> str:
        return "LIKELY BUY" if prob >= self.threshold else "BROWSING"
