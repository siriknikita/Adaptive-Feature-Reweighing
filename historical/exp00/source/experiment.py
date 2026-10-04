import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Reproducibility / device
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def pick_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable. Use --device cpu or verify your PyTorch install.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------
# Synthetic world
# -----------------------------

@dataclass
class Config:
    seed: int = 7
    d: int = 32
    stable_dims: int = 6
    dormant_start: int = 20
    train_n: int = 24000
    val_n: int = 6000
    test_n: int = 10000
    train_transform_prob: float = 0.30
    test_transform_prob: float = 1.00
    noise_std: float = 1.0
    stable_signal: float = 1.25
    dormant_signal: float = 3.0
    corrupt_strength: float = 2.4
    batch_size: int = 512
    epochs: int = 18
    lr: float = 2e-3
    weight_decay: float = 1e-4
    gate_l1: float = 2e-3
    gate_identity: float = 1e-3
    support_shots: int = 8
    adapt_steps: int = 25
    adapt_lr: float = 0.08


# Transform definitions:
# 0 = clean
# 1/2/3 = seen during training
# 4 = HELD OUT, analogous structure but never present in training
# Each transform corrupts two baseline-stable dimensions and makes one dormant
# dimension highly predictive of y. The held-out transform uses a new trio.
TRANSFORMS = {
    0: {"name": "clean", "corrupt": [], "dormant": None},
    1: {"name": "seen_A", "corrupt": [0, 1], "dormant": 20},
    2: {"name": "seen_B", "corrupt": [2, 3], "dormant": 21},
    3: {"name": "seen_C", "corrupt": [4, 5], "dormant": 22},
    4: {"name": "heldout_D", "corrupt": [0, 5], "dormant": 23},
}


def make_split(n: int, cfg: Config, allowed_transforms, transform_prob: float, rng: np.random.Generator):
    d = cfg.d
    y = rng.integers(0, 2, size=n, endpoint=False).astype(np.float32)
    sign = 2.0 * y - 1.0
    x = rng.normal(0.0, cfg.noise_std, size=(n, d)).astype(np.float32)

    # Baseline-stable predictive features.
    for j in range(cfg.stable_dims):
        x[:, j] += sign * cfg.stable_signal

    # Most dormant features are just noise on clean/general data.
    transform_ids = np.zeros(n, dtype=np.int64)
    do_transform = rng.random(n) < transform_prob
    choices = np.array(allowed_transforms, dtype=np.int64)
    sampled = rng.choice(choices, size=int(do_transform.sum())) if do_transform.any() else np.array([], dtype=np.int64)
    transform_ids[do_transform] = sampled

    for tid in allowed_transforms:
        idx = np.where(transform_ids == tid)[0]
        if len(idx) == 0:
            continue
        spec = TRANSFORMS[tid]
        # Corrupt the normally useful features: flip / distort their semantics.
        for j in spec["corrupt"]:
            x[idx, j] = rng.normal(0.0, cfg.noise_std, size=len(idx)).astype(np.float32) - sign[idx] * cfg.corrupt_strength
        # Activate a normally useless dormant feature.
        dj = spec["dormant"]
        if dj is not None:
            x[idx, dj] += sign[idx] * cfg.dormant_signal

    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(transform_ids),
    )


# -----------------------------
# Models
# -----------------------------

class BaseLinear(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.weight, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


class JudgeAdaptive(nn.Module):
    """
    Fixed baseline evidence vector + input-conditioned multiplicative gate.

    effective_w(x) = base_w * gate(x)

    The gate is centered around 1.0, so the default behavior is the baseline.
    """
    def __init__(self, d: int, hidden: int = 96):
        super().__init__()
        self.base_weight = nn.Parameter(torch.zeros(d))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.base_weight, std=0.03)
        self.judge = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        # Start near identity gating.
        nn.init.zeros_(self.judge[-1].weight)
        nn.init.zeros_(self.judge[-1].bias)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        # exp(tanh(.)) keeps gates positive and bounded roughly [e^-1, e^1]
        return torch.exp(torch.tanh(self.judge(x)))

    def forward(self, x: torch.Tensor, delta_log_gate: torch.Tensor = None):
        g = self.gate(x)
        if delta_log_gate is not None:
            # Few-shot adaptation learns only a tiny global per-regime offset.
            g = g * torch.exp(torch.tanh(delta_log_gate)).unsqueeze(0)
        logits = (x * g) @ self.base_weight + self.bias
        return logits, g


class PlainMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# -----------------------------
# Training / evaluation
# -----------------------------

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (logits >= 0).float()
    return (pred == y).float().mean().item()


def evaluate_model(model, x, y, tids, device, adaptive=False, delta=None) -> Dict[str, float]:
    model.eval()
    out = {}
    with torch.no_grad():
        xb, yb = x.to(device), y.to(device)
        if adaptive:
            logits, _ = model(xb, delta)
        else:
            logits = model(xb)
        out["all"] = accuracy(logits, yb)
        for tid in sorted(torch.unique(tids).tolist()):
            mask = tids == tid
            if mask.any():
                if adaptive:
                    l, _ = model(x[mask].to(device), delta)
                else:
                    l = model(x[mask].to(device))
                out[TRANSFORMS[int(tid)]["name"]] = accuracy(l, y[mask].to(device))
    return out


def train_base(model, train_loader, val, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = None
    best_val = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
        metrics = evaluate_model(model, *val, device, adaptive=False)
        if metrics["all"] > best_val:
            best_val = metrics["all"]
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)


def train_mlp(model, train_loader, val, cfg, device):
    train_base(model, train_loader, val, cfg, device)


def train_adaptive(model, train_loader, val, cfg, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = None
    best_val = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, g = model(xb)
            cls = F.binary_cross_entropy_with_logits(logits, yb)
            # Keep most gates near baseline and avoid gratuitous rewrites.
            reg_identity = (torch.log(g).pow(2)).mean()
            reg_sparse = torch.abs(torch.log(g)).mean()
            loss = cls + cfg.gate_identity * reg_identity + cfg.gate_l1 * reg_sparse
            loss.backward()
            opt.step()
        metrics = evaluate_model(model, *val, device, adaptive=True)
        if metrics["all"] > best_val:
            best_val = metrics["all"]
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)


def adapt_delta(model: JudgeAdaptive, x_support, y_support, cfg, device):
    """Few-shot test-time recalibration. Model stays frozen; only d numbers adapt."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    delta = torch.zeros(cfg.d, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.adapt_lr)
    xs, ys = x_support.to(device), y_support.to(device)
    for _ in range(cfg.adapt_steps):
        opt.zero_grad(set_to_none=True)
        logits, _ = model(xs, delta)
        loss = F.binary_cross_entropy_with_logits(logits, ys) + 0.01 * delta.pow(2).mean()
        loss.backward()
        opt.step()
    return delta.detach()


def gate_summary(model: JudgeAdaptive, x, tids, device, delta=None) -> Dict[str, Dict[str, float]]:
    model.eval()
    result = {}
    with torch.no_grad():
        for tid in sorted(torch.unique(tids).tolist()):
            mask = tids == tid
            xb = x[mask].to(device)
            _, g = model(xb, delta)
            mean_g = g.mean(dim=0).cpu().numpy()
            # Report the most changed dimensions.
            changed = np.argsort(np.abs(np.log(mean_g + 1e-9)))[::-1][:8]
            result[TRANSFORMS[int(tid)]["name"]] = {str(int(j)): float(mean_g[j]) for j in changed}
    return result


def subset_for_shots(x, y, tids, tid: int, shots: int, seed: int):
    idx = torch.where(tids == tid)[0]
    g = torch.Generator().manual_seed(seed)
    idx = idx[torch.randperm(len(idx), generator=g)]

    # Aim for balanced few-shot support if possible.
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    half = shots // 2
    support = torch.cat([pos[:half], neg[: shots - half]])
    if len(support) < shots:
        support = idx[:shots]
    support_set = set(support.tolist())
    query = torch.tensor([i.item() for i in idx if i.item() not in support_set], dtype=torch.long)
    return support, query


def main():
    p = argparse.ArgumentParser(description="Adaptive per-input feature reweighting toy experiment")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--shots", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="Smaller smoke-test run")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = Config(seed=args.seed, epochs=args.epochs, support_shots=args.shots)
    if args.quick:
        cfg.train_n = 6000
        cfg.val_n = 1500
        cfg.test_n = 3000
        cfg.epochs = min(cfg.epochs, 6)

    seed_everything(cfg.seed)
    device = pick_device(args.device)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    train = make_split(cfg.train_n, cfg, allowed_transforms=[1, 2, 3], transform_prob=cfg.train_transform_prob, rng=rng)
    val = make_split(cfg.val_n, cfg, allowed_transforms=[1, 2, 3], transform_prob=cfg.train_transform_prob, rng=rng)
    # Held-out transform only: model never sees transform 4 during training.
    test = make_split(cfg.test_n, cfg, allowed_transforms=[4], transform_prob=cfg.test_transform_prob, rng=rng)

    train_loader = DataLoader(TensorDataset(*train), batch_size=cfg.batch_size, shuffle=True)

    base = BaseLinear(cfg.d).to(device)
    mlp = PlainMLP(cfg.d).to(device)
    adaptive = JudgeAdaptive(cfg.d).to(device)

    print(f"Device: {device}")
    print("Training base linear classifier...")
    train_base(base, train_loader, val, cfg, device)
    print("Training plain MLP control...")
    train_mlp(mlp, train_loader, val, cfg, device)
    print("Training adaptive judge model...")
    train_adaptive(adaptive, train_loader, val, cfg, device)

    base_test = evaluate_model(base, *test, device, adaptive=False)
    mlp_test = evaluate_model(mlp, *test, device, adaptive=False)
    adaptive_test = evaluate_model(adaptive, *test, device, adaptive=True)

    # Few-shot adaptation on the unseen held-out transform.
    support_idx, query_idx = subset_for_shots(*test, tid=4, shots=cfg.support_shots, seed=cfg.seed + 99)
    delta = adapt_delta(adaptive, test[0][support_idx], test[1][support_idx], cfg, device)
    query = (test[0][query_idx], test[1][query_idx], test[2][query_idx])
    adaptive_fewshot = evaluate_model(adaptive, *query, device, adaptive=True, delta=delta)

    gates_before = gate_summary(adaptive, test[0], test[2], device, delta=None)
    gates_after = gate_summary(adaptive, query[0], query[2], device, delta=delta)

    # Key diagnostic: how much did held-out dormant feature 23 get resurrected?
    with torch.no_grad():
        _, g0 = adaptive(test[0].to(device), None)
        _, g1 = adaptive(query[0].to(device), delta)
    dormant_23_gate_before = float(g0[:, 23].mean().cpu())
    dormant_23_gate_after = float(g1[:, 23].mean().cpu())

    results = {
        "config": asdict(cfg),
        "device": str(device),
        "heldout_transform": TRANSFORMS[4],
        "base_linear": base_test,
        "plain_mlp": mlp_test,
        "adaptive_zero_shot": adaptive_test,
        "adaptive_fewshot_query": adaptive_fewshot,
        "support_shots": int(len(support_idx)),
        "heldout_dormant_feature_23_mean_gate_before": dormant_23_gate_before,
        "heldout_dormant_feature_23_mean_gate_after": dormant_23_gate_after,
        "gate_summary_before": gates_before,
        "gate_summary_after": gates_after,
    }

    with open(outdir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    rows = [
        ["base_linear", base_test["all"]],
        ["plain_mlp", mlp_test["all"]],
        ["adaptive_zero_shot", adaptive_test["all"]],
        ["adaptive_fewshot", adaptive_fewshot["all"]],
    ]
    with open(outdir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "heldout_accuracy"])
        w.writerows(rows)

    # Save learned vectors for inspection.
    np.save(outdir / "base_linear_weights.npy", base.weight.detach().cpu().numpy())
    np.save(outdir / "adaptive_base_weights.npy", adaptive.base_weight.detach().cpu().numpy())
    np.save(outdir / "fewshot_delta_log_gate.npy", delta.detach().cpu().numpy())

    print("\n=== HELD-OUT TRANSFORM RESULTS ===")
    for name, value in rows:
        print(f"{name:22s}: {value:.4f}")
    print(f"\nFeature 23 mean gate before few-shot: {dormant_23_gate_before:.4f}")
    print(f"Feature 23 mean gate after few-shot : {dormant_23_gate_after:.4f}")
    print(f"\nSaved results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
