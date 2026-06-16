# model/predictor.py
# Loads the combined Transformer + XGBoost bundle and runs ensemble inference.
#
# Bundle structure (best_combine_model_purchase_prediction.joblib):
#   {
#     'transformer_weights': state_dict,
#     'xgboost_model':       XGBClassifier,
#     'scaler':              StandardScaler (42 features),
#     'config': {
#       'alpha':                float,   # transformer weight in blend
#       'optimal_threshold':    float,
#       'sequence_length':      int,
#       'transformer_features': list[str],  # 42 raw feature names
#       'xgboost_features':     list[str],  # 126 = 42 * (mean, max, std)
#     }
#   }
#
# Inference flow:
#   raw windows → scale(42) → Transformer → prob_t
#                           → agg(mean/max/std) → XGBoost → prob_x
#   final = alpha * prob_t + (1 - alpha) * prob_x

import logging
from typing import List

import joblib
import numpy as np
import torch

from constants import D_FF, D_MODEL, DROPOUT, MAX_SEQ, N_HEADS, N_LAYERS, PURCHASE_THRESHOLD
from model.behaviour_transformer import BehaviourTransformer

logger = logging.getLogger(__name__)


class PurchasePredictor:

    def __init__(self, model_path: str, scaler_path: str = ""):
        # Load combined bundle (scaler_path kept for API compatibility, ignored)
        bundle = joblib.load(model_path)

        cfg = bundle["config"]
        self.alpha            = float(cfg["alpha"])
        self.threshold        = float(cfg.get("optimal_threshold", PURCHASE_THRESHOLD))
        self.seq_len          = int(cfg.get("sequence_length", MAX_SEQ))
        self.transformer_cols = list(cfg["transformer_features"])   # 42 features
        self.xgboost_cols     = list(cfg["xgboost_features"])       # 126 features
        n_feats               = len(self.transformer_cols)

        # Reconstruct transformer and load weights
        self.model = BehaviourTransformer(
            n_feats=n_feats, max_seq=self.seq_len,
            d_model=D_MODEL, n_heads=N_HEADS,
            n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
        )
        self.model.load_state_dict(bundle["transformer_weights"])
        self.model.eval()
        logger.info("BehaviourTransformer loaded  n_feats=%d  seq_len=%d", n_feats, self.seq_len)

        self.scaler = bundle["scaler"]
        # Extract the raw Booster from the XGBClassifier — avoids sklearn API
        # incompatibilities when the model was saved with xgboost <=1.6
        import xgboost as _xgb
        self.xgb_booster: _xgb.Booster = bundle["xgboost_model"].get_booster()
        logger.info(
            "Combined model loaded  alpha=%.2f  threshold=%.4f",
            self.alpha, self.threshold,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, windows: List[dict]) -> float:
        """
        windows : list of raw (unscaled) feature dicts, one per completed 5-s window.
        Returns : blended purchase probability in [0, 1].
        """
        if not windows:
            return 0.0

        # ── Build raw (T, 42) array ────────────────────────────────────────────
        raw = np.array(
            [[w.get(col, 0.0) for col in self.transformer_cols] for w in windows],
            dtype=np.float64,
        )

        # ── Transformer branch ─────────────────────────────────────────────────
        scaled = self.scaler.transform(raw).astype(np.float32)

        T = scaled.shape[0]
        if T > self.seq_len:
            scaled = scaled[-self.seq_len:]
            T = self.seq_len

        pad_len = self.seq_len - T
        if pad_len:
            padding = np.zeros((pad_len, scaled.shape[1]), dtype=np.float32)
            padded  = np.concatenate([scaled, padding], axis=0)
        else:
            padded = scaled

        x    = torch.tensor(padded).unsqueeze(0)                  # (1, seq_len, 42)
        mask = torch.zeros(1, self.seq_len, dtype=torch.bool)
        if pad_len:
            mask[0, T:] = True

        with torch.no_grad():
            logit  = self.model(x, key_padding_mask=mask)
            prob_t = float(torch.sigmoid(logit).item())

        # ── XGBoost branch: mean / max / std per feature ───────────────────────
        agg: dict[str, float] = {}
        for i, feat in enumerate(self.transformer_cols):
            col_vals = raw[:, i]
            agg[f"{feat}__mean"] = float(np.mean(col_vals))
            agg[f"{feat}__max"]  = float(np.max(col_vals))
            agg[f"{feat}__std"]  = float(np.std(col_vals)) if len(col_vals) > 1 else 0.0

        xgb_arr = np.array(
            [[agg.get(f, 0.0) for f in self.xgboost_cols]],
            dtype=np.float32,
        )
        import xgboost as _xgb
        dmat   = _xgb.DMatrix(xgb_arr)
        prob_x = float(self.xgb_booster.predict(dmat)[0])

        # ── Ensemble ───────────────────────────────────────────────────────────
        return self.alpha * prob_t + (1.0 - self.alpha) * prob_x

    def verdict(self, prob: float) -> str:
        return "LIKELY BUY" if prob >= self.threshold else "BROWSING"
