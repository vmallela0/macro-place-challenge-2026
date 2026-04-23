"""Tiny MLP surrogate for probe ranking — your idea #1.

Workflow:
  1. Log every real proxy-cost probe: (features, Δcost).
  2. Train a small MLP to predict Δcost from features.
  3. At probe time, score many candidates cheaply; verify only top-k with
     real eval.

Features per probe (11-dim), zero-copied from IncrementalEvaluator state:
  [macro_cx, macro_cy, proposed_dx, proposed_dy, canvas_frac_x, canvas_frac_y,
   net_count, net_centroid_dx, net_centroid_dy,
   dist_to_centroid_before, dist_to_centroid_after]

MPS is used when available (Apple Silicon). Otherwise CPU. Training takes
~1-2s for 10k samples; inference is vectorised so 1000 candidates scored
in ~1-2ms on MPS (~10x with CPU).
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
    _DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
except Exception:
    _HAS_TORCH = False
    _DEV = None


class ProbeLogger:
    """Collect (features, Δcost) samples during CD.

    Usage:
        logger = ProbeLogger(max_samples=50000)
        # in the probe loop:
        feats = logger.build_features(incr_eval, macro_idx, old_x, old_y, new_x, new_y)
        new_cost = incr_eval.move_macro(...)
        delta = new_cost - old_cost
        logger.add(feats, delta)
        incr_eval.undo_move()
    """

    FEATURE_DIM = 11

    def __init__(self, max_samples=50000):
        self.feats = np.zeros((max_samples, self.FEATURE_DIM), dtype=np.float32)
        self.deltas = np.zeros(max_samples, dtype=np.float32)
        self.n = 0
        self.max_samples = max_samples

    def build_features(self, incr_eval, macro_idx, old_x, old_y, new_x, new_y):
        pos = incr_eval.macro_pos
        # Net centroid of connected macros (bounded: use first 8 nets for speed)
        cx_sum, cy_sum, cnt = 0.0, 0.0, 0
        for nid in incr_eval.macro_nets[macro_idx][:8]:
            start = int(incr_eval.net_starts[nid])
            end = int(incr_eval.net_starts[nid + 1])
            for pidx in range(start, end):
                pm = int(incr_eval.pin_macro[pidx])
                if pm == macro_idx:
                    continue
                if pm < 0:
                    cx_sum += float(incr_eval.pin_xoff[pidx])
                    cy_sum += float(incr_eval.pin_yoff[pidx])
                else:
                    cx_sum += float(pos[pm, 0])
                    cy_sum += float(pos[pm, 1])
                cnt += 1
        if cnt > 0:
            ncx = cx_sum / cnt
            ncy = cy_sum / cnt
        else:
            ncx, ncy = old_x, old_y

        cw = incr_eval.cw
        ch = incr_eval.ch
        dx = new_x - old_x
        dy = new_y - old_y
        return np.array([
            old_x, old_y,
            dx, dy,
            old_x / cw, old_y / ch,
            min(20, len(incr_eval.macro_nets[macro_idx])),
            ncx - old_x, ncy - old_y,
            ((old_x - ncx) ** 2 + (old_y - ncy) ** 2) ** 0.5,
            ((new_x - ncx) ** 2 + (new_y - ncy) ** 2) ** 0.5,
        ], dtype=np.float32)

    def add(self, features, delta):
        if self.n < self.max_samples:
            self.feats[self.n] = features
            self.deltas[self.n] = delta
            self.n += 1

    def is_ready(self, min_samples=2000):
        return self.n >= min_samples


class ProbeSurrogate(nn.Module if _HAS_TORCH else object):
    """2-layer MLP: 11 → 256 → 256 → 1. ~70k params."""

    def __init__(self, input_dim=11, hidden=256):
        if not _HAS_TORCH:
            raise RuntimeError("torch not available")
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.bn_feats = None
        self.feat_mean = None
        self.feat_std = None

    def forward(self, x):
        if self.feat_mean is not None:
            x = (x - self.feat_mean) / (self.feat_std + 1e-6)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h).squeeze(-1)


def train_surrogate(logger: ProbeLogger, epochs=30, batch_size=512):
    """Train a ProbeSurrogate on the logged data.

    Returns the trained model on _DEV, or None if not enough data or no torch.
    """
    if not _HAS_TORCH:
        return None
    if logger.n < 500:
        return None

    feats = torch.tensor(logger.feats[:logger.n], device=_DEV, dtype=torch.float32)
    deltas = torch.tensor(logger.deltas[:logger.n], device=_DEV, dtype=torch.float32)

    # Normalize
    feat_mean = feats.mean(dim=0, keepdim=True)
    feat_std = feats.std(dim=0, keepdim=True)
    delta_mean = deltas.mean()
    delta_std = deltas.std() + 1e-6

    model = ProbeSurrogate(input_dim=logger.FEATURE_DIM, hidden=256).to(_DEV)
    model.feat_mean = feat_mean
    model.feat_std = feat_std

    # Target is (delta - delta_mean) / delta_std — zero-mean, unit-variance.
    # We'll de-normalize at inference.
    targets = (deltas - delta_mean) / delta_std

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n = logger.n
    idx = torch.arange(n, device=_DEV)
    for ep in range(epochs):
        perm = idx[torch.randperm(n)]
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            b = perm[i:j]
            pred = model(feats[b])
            loss = F.mse_loss(pred, targets[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model._delta_mean = delta_mean.item()
    model._delta_std = delta_std.item()
    return model


def surrogate_rank_candidates(model, feature_batch):
    """Score candidate probes. Returns (predicted_delta, indices_sorted_ascending).

    feature_batch: (K, D) numpy array.
    """
    if model is None:
        return None, None
    with torch.no_grad():
        t = torch.tensor(feature_batch, device=_DEV, dtype=torch.float32)
        out = model(t).cpu().numpy()
        # De-normalize
        out = out * model._delta_std + model._delta_mean
    order = np.argsort(out)  # ascending — most-improving first
    return out, order
