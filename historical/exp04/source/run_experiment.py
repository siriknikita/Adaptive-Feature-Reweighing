
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF


# ============================================================================
# Reproducibility / device
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


# ============================================================================
# Synthetic evidence world
# ============================================================================

class SyntheticEvidenceDataset(Dataset):
    def __init__(
        self,
        n: int,
        image_size: int,
        seed: int,
        corruption: str = "clean",
        cue_b_strength: float = 0.72,
        cue_c_strength: float = 0.66,
    ):
        self.n = n
        self.s = image_size
        self.seed = seed
        self.corruption = corruption
        self.cue_b_strength = cue_b_strength
        self.cue_c_strength = cue_c_strength

    def __len__(self) -> int:
        return self.n

    @staticmethod
    def maybe_flip(rng: random.Random, p_correct: float, y: int) -> int:
        return y if rng.random() < p_correct else 1 - y

    def make_base(self, idx: int) -> Tuple[Image.Image, int]:
        s = self.s
        y = idx % 2
        rng = random.Random(self.seed + idx * 7919)
        rng_np = np.random.default_rng(self.seed * 100003 + idx)

        arr = np.full((s, s, 3), 118.0, dtype=np.float32)
        arr += rng_np.normal(0, 10.0, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        draw.ellipse(
            [int(.18*s), int(.08*s), int(.82*s), int(.92*s)],
            outline=(150, 150, 150),
            width=max(1, s // 64),
        )

        # A: dominant fragile eye cue.
        eye_y = int(.34*s)
        r = max(2, s // 20)
        ax = int((.34 if y == 0 else .66) * s)
        dx = int((.66 if y == 0 else .34) * s)
        draw.ellipse([ax-r, eye_y-r, ax+r, eye_y+r], fill=(246, 246, 246))
        draw.ellipse([dx-r, eye_y-r, dx+r, eye_y+r], fill=(72, 72, 72))

        # B: weaker stable geometry cue.
        b_label = self.maybe_flip(rng, self.cue_b_strength, y)
        cx, cy = int(.50*s), int(.70*s)
        span = int(.18*s)
        if b_label == 0:
            draw.line(
                [cx-span, cy+span//2, cx, cy-span//2, cx+span, cy+span//2],
                fill=(222, 222, 222),
                width=max(2, s // 28),
            )
        else:
            draw.line(
                [cx-span, cy-span//2, cx, cy+span//2, cx+span, cy-span//2],
                fill=(222, 222, 222),
                width=max(2, s // 28),
            )

        # C: weaker stable chroma cue.
        c_label = self.maybe_flip(rng, self.cue_c_strength, y)
        arr = np.asarray(img).astype(np.float32)
        delta = 8.0
        if c_label == 0:
            arr[..., 0] += delta
            arr[..., 2] -= delta
        else:
            arr[..., 2] += delta
            arr[..., 0] -= delta

        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), y

    def corrupt(self, img: Image.Image, idx: int) -> Image.Image:
        kind = self.corruption
        if kind == "clean":
            return img

        rng = random.Random(self.seed * 17 + idx)
        s = self.s
        out = img.copy()
        draw = ImageDraw.Draw(out)

        if kind == "eye_patch":
            draw.rectangle(
                [int(.20*s), int(.25*s), int(.80*s), int(.45*s)],
                fill=(rng.randint(80, 150),) * 3,
            )
            return out

        if kind == "lower_patch":
            draw.rectangle(
                [int(.25*s), int(.56*s), int(.75*s), int(.86*s)],
                fill=(rng.randint(80, 150),) * 3,
            )
            return out

        if kind == "desaturate":
            g = TF.rgb_to_grayscale(TF.to_tensor(out), num_output_channels=3)
            return TF.to_pil_image(g)

        if kind == "blur":
            return out.filter(ImageFilter.GaussianBlur(radius=1.7))

        if kind == "jpeg":
            from io import BytesIO
            buf = BytesIO()
            out.save(buf, format="JPEG", quality=20)
            buf.seek(0)
            return Image.open(buf).convert("RGB")

        if kind == "glasses":
            # Never used in slow-router meta-training.
            lens_y0, lens_y1 = int(.27*s), int(.42*s)
            lens_w = int(.23*s)
            gap = int(.055*s)
            lx0 = s // 2 - gap // 2 - lens_w
            rx0 = s // 2 + gap // 2
            dark = (20, 20, 20)
            frame = (3, 3, 3)

            draw.rounded_rectangle(
                [lx0, lens_y0, lx0+lens_w, lens_y1],
                radius=max(2, s // 40),
                fill=dark,
                outline=frame,
                width=max(1, s // 64),
            )
            draw.rounded_rectangle(
                [rx0, lens_y0, rx0+lens_w, lens_y1],
                radius=max(2, s // 40),
                fill=dark,
                outline=frame,
                width=max(1, s // 64),
            )
            mid = (lens_y0 + lens_y1) // 2
            draw.line([lx0+lens_w, mid, rx0, mid], fill=frame, width=max(1, s // 48))
            draw.line([int(.10*s), mid, lx0, mid], fill=frame, width=max(1, s // 64))
            draw.line([rx0+lens_w, mid, int(.90*s), mid], fill=frame, width=max(1, s // 64))
            return out

        raise ValueError(f"Unknown corruption: {kind}")

    def __getitem__(self, idx: int):
        img, y = self.make_base(idx)
        img = self.corrupt(img, idx)
        x = TF.to_tensor(img.resize((self.s, self.s), Image.Resampling.BILINEAR))
        return x, y


# ============================================================================
# Isolated feature views
# ============================================================================

def view_a(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    h = x.shape[-2]
    out[..., int(.20*h):int(.48*h), :] = x[..., int(.20*h):int(.48*h), :]
    return out


def view_b(x: torch.Tensor) -> torch.Tensor:
    g = 0.299*x[:, 0:1] + 0.587*x[:, 1:2] + 0.114*x[:, 2:3]
    out = torch.zeros_like(g)
    h = x.shape[-2]
    out[..., int(.50*h):, :] = g[..., int(.50*h):, :]
    return out


def view_c(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([r-g, b-g, r-b], dim=1)


FEATURES = {
    "A_eye": (view_a, 3),
    "B_lower_geometry": (view_b, 1),
    "C_chroma": (view_c, 3),
}


# ============================================================================
# Models
# ============================================================================

class TinyExpert(nn.Module):
    def __init__(self, in_ch: int, width: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width*2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width*2, width*3, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width*3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).flatten(1))


class ExpertBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.names = list(FEATURES)
        self.experts = nn.ModuleDict({
            name: TinyExpert(FEATURES[name][1])
            for name in self.names
        })

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for name in self.names:
            fn, _ = FEATURES[name]
            outs.append(self.experts[name](fn(x)))
        return torch.stack(outs, dim=1)


class SlowGate(nn.Module):
    def __init__(self, n_experts: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(12, 20, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(20 + 3*n_experts, 48),
            nn.GELU(),
            nn.Linear(48, n_experts),
        )

    def forward(self, x: torch.Tensor, expert_logits: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x).flatten(1)
        p = expert_logits.softmax(-1)
        conf = p.max(-1).values
        margin = (p[..., 1] - p[..., 0]).abs()
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1)
        return self.mlp(torch.cat([z, conf, margin, entropy], dim=1))


# ============================================================================
# Routing math
# ============================================================================

def fuse(expert_logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (expert_logits * weights.unsqueeze(-1)).sum(1)


def weight_entropy(w: torch.Tensor) -> torch.Tensor:
    return -(w * w.clamp_min(1e-8).log()).sum(1)


def softmax_jacobian_frobenius(p: torch.Tensor) -> torch.Tensor:
    """
    p: [B,E]
    Returns one ||diag(p)-p p^T||_F value per sample.
    """
    diag = torch.diag_embed(p)
    outer = p.unsqueeze(2) * p.unsqueeze(1)
    j = diag - outer
    return torch.sqrt((j*j).sum(dim=(1, 2)) + 1e-20)


def logit_of_probability(p: float) -> float:
    return math.log(p / (1.0 - p))


# ============================================================================
# Data / training utilities
# ============================================================================

def make_loader(ds, batch: int, shuffle: bool = False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def standalone_accuracies(bank, dl, device):
    bank.eval()
    correct = torch.zeros(len(bank.names), dtype=torch.long)
    n = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        ex = bank.forward_all(x)
        pred = ex.argmax(-1)
        correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum(0)
        n += y.numel()
    return {name: correct[i].item() / max(1, n) for i, name in enumerate(bank.names)}


def train_experts(bank, train_dl, val_dl, device, epochs: int, lr: float):
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=1e-4)
    best = None
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        bank.train()
        total_loss = total = 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            ex = bank.forward_all(x)
            loss = sum(F.cross_entropy(ex[:, i], y) for i in range(ex.shape[1])) / ex.shape[1]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.numel()
            total += y.numel()

        accs = standalone_accuracies(bank, val_dl, device)
        score = sum(accs.values()) / len(accs)
        print(f"[experts] epoch={epoch:02d} loss={total_loss/max(1,total):.4f} val={accs}")

        if score > best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in bank.state_dict().items()}

    if best is not None:
        bank.load_state_dict(best)


def train_global(bank, train_dl, device, epochs: int, lr: float):
    for p in bank.parameters():
        p.requires_grad_(False)

    logits = nn.Parameter(torch.zeros(len(bank.names), device=device))
    opt = torch.optim.Adam([logits], lr=lr)

    for epoch in range(1, epochs + 1):
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                ex = bank.forward_all(x)

            w = logits.softmax(0).unsqueeze(0).expand(x.shape[0], -1)
            out = fuse(ex, w)
            loss = F.cross_entropy(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        print(
            f"[global] epoch={epoch:02d} "
            f"weights={logits.softmax(0).detach().cpu().numpy().round(3).tolist()}"
        )

    return logits.detach().clone()


# ============================================================================
# One shared slow-router meta-training run per seed
# ============================================================================

def meta_train_slow_gate(bank, gate, global_logits, args, device, seed):
    """
    Only one slow gate is trained per seed.
    All Experiment #4 adaptation methods share this exact gate.
    """
    meta_corruptions = ["clean", "eye_patch", "lower_patch", "desaturate", "blur", "jpeg"]
    rng = random.Random(seed + 5000)
    outer = torch.optim.AdamW(gate.parameters(), lr=args.meta_lr, weight_decay=1e-4)

    for p in bank.parameters():
        p.requires_grad_(False)

    for ep in range(1, args.meta_episodes + 1):
        corr = meta_corruptions[rng.randrange(len(meta_corruptions))]
        ds = SyntheticEvidenceDataset(
            args.train_n,
            args.image_size,
            seed + 100,
            corr,
            args.cue_b_strength,
            args.cue_c_strength,
        )

        ids = list(range(len(ds)))
        rng.shuffle(ids)
        sids = ids[:args.support_size]
        qids = ids[args.support_size:args.support_size + args.query_size]

        fast_gate = copy.deepcopy(gate).to(device)
        inner = torch.optim.SGD(fast_gate.parameters(), lr=args.inner_lr)

        x, y = next(iter(make_loader(Subset(ds, sids), args.support_size, True)))
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            ex = bank.forward_all(x)

        z = fast_gate(x, ex)

        # Same anti-collapse pressure as the previous toy experiment.
        if rng.random() < args.expert_dropout:
            j = rng.randrange(ex.shape[1])
            z = z.clone()
            z[:, j] -= 6.0

        w = z.softmax(1)
        out = fuse(ex, w)
        ce = F.cross_entropy(out, y)

        ent = weight_entropy(w).mean()
        target_ent = args.entropy_floor * math.log(ex.shape[1])
        ent_pen = F.relu(torch.tensor(target_ent, device=device) - ent)
        loss = ce + args.entropy_lambda * ent_pen

        inner.zero_grad(set_to_none=True)
        loss.backward()
        inner.step()

        xq, yq = next(iter(make_loader(Subset(ds, qids), args.query_size, False)))
        xq, yq = xq.to(device), yq.to(device)

        with torch.no_grad():
            exq = bank.forward_all(xq)

        qz = fast_gate(xq, exq)
        qw = qz.softmax(1)
        qout = fuse(exq, qw)
        qloss = F.cross_entropy(qout, yq)

        fast_gate.zero_grad(set_to_none=True)
        qloss.backward()

        outer.zero_grad(set_to_none=True)
        for p, fp in zip(gate.parameters(), fast_gate.parameters()):
            if fp.grad is not None:
                p.grad = fp.grad.detach().clone()
        outer.step()

        if ep == 1 or ep % args.log_every == 0:
            print(
                f"[meta slow] {ep:04d}/{args.meta_episodes} "
                f"corr={corr:12s} qloss={qloss.item():.4f}"
            )


# ============================================================================
# Evaluation helpers
# ============================================================================

@torch.no_grad()
def evaluate_weights(bank, dl, device, weight_fn):
    bank.eval()
    n = correct = 0
    loss_sum = 0.0
    wsum = torch.zeros(len(bank.names), device=device)
    entropy_sum = 0.0
    saturated = 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        ex = bank.forward_all(x)
        w = weight_fn(x, ex)

        out = fuse(ex, w)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()

        wsum += w.sum(0)
        entropy_sum += weight_entropy(w).sum().item()
        saturated += (w.max(1).values > 0.99).sum().item()

    return {
        "accuracy": correct / max(1, n),
        "loss": loss_sum / max(1, n),
        "mean_weights": (wsum / max(1, n)).detach().cpu().tolist(),
        "mean_entropy": entropy_sum / max(1, n),
        "saturation_rate": saturated / max(1, n),
        "n": n,
    }


def balanced_indices(ds, k: int, seed: int):
    by = {0: [], 1: []}
    for i in range(len(ds)):
        _, y = ds[i]
        by[int(y)].append(i)

    rng = random.Random(seed)
    out = []
    each = max(1, k // 2)

    for c in (0, 1):
        rng.shuffle(by[c])
        out.extend(by[c][:each])

    return out[:k]


# ============================================================================
# First-step diagnostics
# ============================================================================

def diagnostic_slow_sgd(bank, gate, support_batch, device):
    x, y = support_batch
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        ex = bank.forward_all(x)

    z = gate(x, ex)
    z.retain_grad()
    w = z.softmax(1)
    out = fuse(ex, w)
    loss = F.cross_entropy(out, y)

    gate.zero_grad(set_to_none=True)
    loss.backward()

    grad = z.grad.detach()
    backup_grad = torch.sqrt((grad[:, 1:]**2).sum(1)).mean().item()
    conductance = softmax_jacobian_frobenius(w.detach()).mean().item()

    return backup_grad, conductance


def diagnostic_additive_fast(bank, gate, support_batch, device):
    x, y = support_batch
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        ex = bank.forward_all(x)
        zslow = gate(x, ex)

    a = torch.zeros(len(bank.names), device=device, requires_grad=True)
    w = (zslow + a.unsqueeze(0)).softmax(1)
    out = fuse(ex, w)
    loss = F.cross_entropy(out, y)
    loss.backward()

    backup_grad = torch.sqrt((a.grad[1:]**2).sum()).item()
    conductance = softmax_jacobian_frobenius(w.detach()).mean().item()

    return backup_grad, conductance


def diagnostic_mixture(bank, gate, support_batch, device, alpha: float):
    x, y = support_batch
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        ex = bank.forward_all(x)
        pslow = gate(x, ex).softmax(1)

    a = torch.zeros(len(bank.names), device=device, requires_grad=True)
    pfast = a.softmax(0)
    w = (1.0 - alpha) * pslow + alpha * pfast.unsqueeze(0)

    out = fuse(ex, w)
    loss = F.cross_entropy(out, y)
    loss.backward()

    backup_grad = torch.sqrt((a.grad[1:]**2).sum()).item()

    pfast_batch = pfast.detach().unsqueeze(0).expand(x.shape[0], -1)
    conductance = (
        alpha * softmax_jacobian_frobenius(pfast_batch)
    ).mean().item()

    return backup_grad, conductance


# ============================================================================
# Adaptation mechanisms
# ============================================================================

class AdaptedWeights:
    def __init__(self, method, gate=None, fast_logits=None, alpha=None):
        self.method = method
        self.gate = gate
        self.fast_logits = fast_logits
        self.alpha = alpha


def adapt_slow_sgd(bank, slow_gate, support_dl, device, args):
    gate = copy.deepcopy(slow_gate).to(device)
    opt = torch.optim.SGD(gate.parameters(), lr=args.test_adapt_lr)
    cached = [(x.to(device), y.to(device)) for x, y in support_dl]

    for _ in range(args.test_adapt_steps):
        for x, y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)
            w = gate(x, ex).softmax(1)
            out = fuse(ex, w)
            loss = F.cross_entropy(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return AdaptedWeights("slow_sgd", gate=gate)


def adapt_additive_fast(bank, slow_gate, support_dl, device, args):
    a = nn.Parameter(torch.zeros(len(bank.names), device=device))
    opt = torch.optim.Adam([a], lr=args.fast_lr)
    cached = [(x.to(device), y.to(device)) for x, y in support_dl]

    for _ in range(args.fast_steps):
        for x, y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)
                zslow = slow_gate(x, ex)

            w = (zslow + a.unsqueeze(0)).softmax(1)
            out = fuse(ex, w)
            loss = F.cross_entropy(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return AdaptedWeights(
        "additive_fast",
        gate=slow_gate,
        fast_logits=a.detach(),
    )


def adapt_mixture_fixed(bank, slow_gate, support_dl, device, args):
    a = nn.Parameter(torch.zeros(len(bank.names), device=device))
    alpha = args.fixed_alpha
    opt = torch.optim.Adam([a], lr=args.fast_lr)
    cached = [(x.to(device), y.to(device)) for x, y in support_dl]

    for _ in range(args.fast_steps):
        for x, y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)
                pslow = slow_gate(x, ex).softmax(1)

            pfast = a.softmax(0)
            w = (1.0-alpha)*pslow + alpha*pfast.unsqueeze(0)
            out = fuse(ex, w)
            loss = F.cross_entropy(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return AdaptedWeights(
        "mixture_fixed",
        gate=slow_gate,
        fast_logits=a.detach(),
        alpha=float(alpha),
    )


def adapt_mixture_learned(bank, slow_gate, support_dl, device, args):
    a = nn.Parameter(torch.zeros(len(bank.names), device=device))
    beta = nn.Parameter(
        torch.tensor(logit_of_probability(args.learned_alpha_init), device=device)
    )
    opt = torch.optim.Adam([a, beta], lr=args.fast_lr)
    cached = [(x.to(device), y.to(device)) for x, y in support_dl]

    for _ in range(args.fast_steps):
        for x, y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)
                pslow = slow_gate(x, ex).softmax(1)

            pfast = a.softmax(0)
            alpha = beta.sigmoid()
            w = (1.0-alpha)*pslow + alpha*pfast.unsqueeze(0)
            out = fuse(ex, w)
            loss = F.cross_entropy(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return AdaptedWeights(
        "mixture_learned",
        gate=slow_gate,
        fast_logits=a.detach(),
        alpha=float(beta.sigmoid().detach().item()),
    )


def adapt_fast_only(bank, support_dl, device, args):
    a = nn.Parameter(torch.zeros(len(bank.names), device=device))
    opt = torch.optim.Adam([a], lr=args.fast_lr)
    cached = [(x.to(device), y.to(device)) for x, y in support_dl]

    for _ in range(args.fast_steps):
        for x, y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)

            w = a.softmax(0).unsqueeze(0).expand(x.shape[0], -1)
            out = fuse(ex, w)
            loss = F.cross_entropy(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return AdaptedWeights(
        "fast_only",
        fast_logits=a.detach(),
        alpha=1.0,
    )


def adapt_oracle(bank, support_dl, device, args):
    a = nn.Parameter(torch.zeros(len(bank.names), device=device))
    opt = torch.optim.Adam([a], lr=args.oracle_lr)
    cached = [(x.to(device), y.to(device)) for x, y in support_dl]

    for _ in range(args.oracle_steps):
        for x, y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)

            w = a.softmax(0).unsqueeze(0).expand(x.shape[0], -1)
            out = fuse(ex, w)
            loss = F.cross_entropy(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return AdaptedWeights(
        "oracle",
        fast_logits=a.detach(),
        alpha=1.0,
    )


def make_weight_fn(adapted: AdaptedWeights):
    method = adapted.method

    if method == "slow_sgd":
        def fn(x, ex):
            return adapted.gate(x, ex).softmax(1)
        return fn

    if method == "additive_fast":
        def fn(x, ex):
            zslow = adapted.gate(x, ex)
            return (zslow + adapted.fast_logits.unsqueeze(0)).softmax(1)
        return fn

    if method in ("mixture_fixed", "mixture_learned"):
        def fn(x, ex):
            pslow = adapted.gate(x, ex).softmax(1)
            pfast = adapted.fast_logits.softmax(0)
            alpha = adapted.alpha
            return (1.0-alpha)*pslow + alpha*pfast.unsqueeze(0)
        return fn

    if method in ("fast_only", "oracle"):
        def fn(x, ex):
            return adapted.fast_logits.softmax(0).unsqueeze(0).expand(x.shape[0], -1)
        return fn

    raise ValueError(method)


# ============================================================================
# Result rows
# ============================================================================

@dataclass
class Row:
    seed: int
    method: str
    corruption: str
    k_shot: int
    accuracy: float
    loss: float
    mean_entropy: float
    saturation_rate: float
    w_A_eye: float
    w_B_lower_geometry: float
    w_C_chroma: float
    A_eye_acc: float
    B_lower_geometry_acc: float
    C_chroma_acc: float
    release_A: float
    recovery_BC: float
    initial_backup_grad_norm: float
    initial_route_conductance: float
    alpha_after: float
    clean_accuracy_after_adaptation: float


def row_from_metrics(
    seed,
    method,
    corruption,
    k,
    m,
    eacc,
    zero_weights,
    grad_norm,
    conductance,
    alpha_after,
    clean_after,
):
    w = m["mean_weights"]

    if zero_weights is None:
        release_a = 0.0
        recovery_bc = 0.0
    else:
        release_a = zero_weights[0] - w[0]
        recovery_bc = (w[1] + w[2]) - (zero_weights[1] + zero_weights[2])

    return Row(
        seed=seed,
        method=method,
        corruption=corruption,
        k_shot=k,
        accuracy=m["accuracy"],
        loss=m["loss"],
        mean_entropy=m["mean_entropy"],
        saturation_rate=m["saturation_rate"],
        w_A_eye=w[0],
        w_B_lower_geometry=w[1],
        w_C_chroma=w[2],
        A_eye_acc=eacc["A_eye"],
        B_lower_geometry_acc=eacc["B_lower_geometry"],
        C_chroma_acc=eacc["C_chroma"],
        release_A=release_a,
        recovery_BC=recovery_bc,
        initial_backup_grad_norm=grad_norm,
        initial_route_conductance=conductance,
        alpha_after=float(alpha_after),
        clean_accuracy_after_adaptation=clean_after,
    )


# ============================================================================
# One seed
# ============================================================================

def run_seed(args, seed: int, out_dir: Path):
    print("\n" + "="*92)
    print(f"SEED {seed}")
    print("="*92)

    seed_everything(seed)
    device = pick_device(args.device)

    print(
        f"device={device}; torch={torch.__version__}; "
        f"mps_available={torch.backends.mps.is_available()}"
    )

    train = SyntheticEvidenceDataset(
        args.train_n, args.image_size, seed+10, "clean",
        args.cue_b_strength, args.cue_c_strength,
    )
    val = SyntheticEvidenceDataset(
        args.val_n, args.image_size, seed+20, "clean",
        args.cue_b_strength, args.cue_c_strength,
    )

    bank = ExpertBank().to(device)

    print("\nStage 1/4 — train isolated experts")
    train_experts(
        bank,
        make_loader(train, args.batch_size, True),
        make_loader(val, args.batch_size, False),
        device,
        args.expert_epochs,
        args.expert_lr,
    )

    print("\nStage 2/4 — learn global fusion")
    global_logits = train_global(
        bank,
        make_loader(train, args.batch_size, True),
        device,
        args.global_epochs,
        args.global_lr,
    )

    print("\nStage 3/4 — train ONE shared slow router")
    slow_gate = SlowGate(len(bank.names)).to(device)
    with torch.no_grad():
        slow_gate.mlp[-1].bias.copy_(global_logits)

    meta_train_slow_gate(
        bank,
        slow_gate,
        global_logits,
        args,
        device,
        seed,
    )

    # Frozen common evaluation worlds.
    clean = SyntheticEvidenceDataset(
        args.test_n, args.image_size, seed+30, "clean",
        args.cue_b_strength, args.cue_c_strength,
    )
    glasses = SyntheticEvidenceDataset(
        args.test_n, args.image_size, seed+30, "glasses",
        args.cue_b_strength, args.cue_c_strength,
    )

    clean_dl = make_loader(clean, args.batch_size, False)
    glasses_dl = make_loader(glasses, args.batch_size, False)

    clean_eacc = standalone_accuracies(bank, clean_dl, device)
    glasses_eacc = standalone_accuracies(bank, glasses_dl, device)

    print(f"[clean experts]   {clean_eacc}")
    print(f"[glasses experts] {glasses_eacc}")

    # Exact shared slow baseline.
    slow_weight_fn = lambda x, ex: slow_gate(x, ex).softmax(1)
    clean_slow = evaluate_weights(bank, clean_dl, device, slow_weight_fn)
    glasses_slow = evaluate_weights(bank, glasses_dl, device, slow_weight_fn)
    zero_weights = glasses_slow["mean_weights"]

    global_weight_fn = lambda x, ex: global_logits.softmax(0).unsqueeze(0).expand(x.shape[0], -1)
    global_glasses = evaluate_weights(bank, glasses_dl, device, global_weight_fn)

    print(
        f"[slow zero-shot glasses] acc={glasses_slow['accuracy']:.4f} "
        f"sat={glasses_slow['saturation_rate']:.3f} "
        f"weights={np.round(zero_weights, 4).tolist()}"
    )

    rows = []

    rows.append(row_from_metrics(
        seed, "slow_zero_shot", "clean", 0, clean_slow, clean_eacc,
        None, 0.0, 0.0, 0.0, clean_slow["accuracy"]
    ))
    rows.append(row_from_metrics(
        seed, "slow_zero_shot", "glasses", 0, glasses_slow, glasses_eacc,
        None, 0.0, 0.0, 0.0, clean_slow["accuracy"]
    ))
    rows.append(row_from_metrics(
        seed, "global", "glasses", 0, global_glasses, glasses_eacc,
        None, 0.0, 0.0, 0.0, 0.0
    ))

    print("\nStage 4/4 — paired fast-adaptation ablation")

    methods = [
        "slow_sgd",
        "additive_fast",
        "mixture_fixed",
        "mixture_learned",
        "fast_only",
        "oracle",
    ]

    for k in args.k_shots:
        # Identical support/query split for every method at this seed and K.
        sids = balanced_indices(glasses, k, seed + k)
        sidset = set(sids)
        qids = [i for i in range(len(glasses)) if i not in sidset]

        support_dl = make_loader(
            Subset(glasses, sids),
            min(k, args.batch_size),
            True,
        )
        query_dl = make_loader(
            Subset(glasses, qids),
            args.batch_size,
            False,
        )

        query_eacc = standalone_accuracies(bank, query_dl, device)

        # Deterministic first batch for diagnostics.
        diagnostic_dl = make_loader(
            Subset(glasses, sids),
            min(k, args.batch_size),
            False,
        )
        support_batch = next(iter(diagnostic_dl))

        slow_grad, slow_cond = diagnostic_slow_sgd(
            bank, slow_gate, support_batch, device
        )
        add_grad, add_cond = diagnostic_additive_fast(
            bank, slow_gate, support_batch, device
        )
        mix_fixed_grad, mix_fixed_cond = diagnostic_mixture(
            bank, slow_gate, support_batch, device, args.fixed_alpha
        )
        mix_learned_grad, mix_learned_cond = diagnostic_mixture(
            bank, slow_gate, support_batch, device, args.learned_alpha_init
        )

        diagnostics = {
            "slow_sgd": (slow_grad, slow_cond),
            "additive_fast": (add_grad, add_cond),
            "mixture_fixed": (mix_fixed_grad, mix_fixed_cond),
            "mixture_learned": (mix_learned_grad, mix_learned_cond),
            # Fast-only starts uniform.
            "fast_only": (
                mix_fixed_grad / max(args.fixed_alpha, 1e-12),
                mix_fixed_cond / max(args.fixed_alpha, 1e-12),
            ),
            "oracle": (
                mix_fixed_grad / max(args.fixed_alpha, 1e-12),
                mix_fixed_cond / max(args.fixed_alpha, 1e-12),
            ),
        }

        adapted_objects = {
            "slow_sgd": adapt_slow_sgd(
                bank, slow_gate, support_dl, device, args
            ),
            "additive_fast": adapt_additive_fast(
                bank, slow_gate, support_dl, device, args
            ),
            "mixture_fixed": adapt_mixture_fixed(
                bank, slow_gate, support_dl, device, args
            ),
            "mixture_learned": adapt_mixture_learned(
                bank, slow_gate, support_dl, device, args
            ),
            "fast_only": adapt_fast_only(
                bank, support_dl, device, args
            ),
            "oracle": adapt_oracle(
                bank, support_dl, device, args
            ),
        }

        for method in methods:
            adapted = adapted_objects[method]
            fn = make_weight_fn(adapted)

            glasses_m = evaluate_weights(
                bank, query_dl, device, fn
            )
            clean_after = evaluate_weights(
                bank, clean_dl, device, fn
            )["accuracy"]

            grad_norm, conductance = diagnostics[method]
            alpha_after = (
                adapted.alpha
                if adapted.alpha is not None
                else 0.0
            )

            rows.append(row_from_metrics(
                seed,
                method,
                "glasses",
                k,
                glasses_m,
                query_eacc,
                zero_weights,
                grad_norm,
                conductance,
                alpha_after,
                clean_after,
            ))

            print(
                f"[k={k:02d} {method:15s}] "
                f"acc={glasses_m['accuracy']:.4f} "
                f"clean_after={clean_after:.4f} "
                f"gradBC={grad_norm:.3e} "
                f"cond={conductance:.3e} "
                f"alpha={alpha_after:.3f} "
                f"w={np.round(glasses_m['mean_weights'], 3).tolist()}"
            )

    torch.save(
        {
            "seed": seed,
            "bank": bank.state_dict(),
            "slow_gate": slow_gate.state_dict(),
            "global_logits": global_logits.detach().cpu(),
            "config": vars(args),
            "expert_names": bank.names,
        },
        out_dir / f"checkpoint_seed_{seed}.pt",
    )

    return rows


# ============================================================================
# Aggregate summary
# ============================================================================

def summarize(rows: List[Row]):
    lines = []
    summary_rows = []

    methods = [
        "slow_sgd",
        "additive_fast",
        "mixture_fixed",
        "mixture_learned",
        "fast_only",
        "oracle",
    ]

    slow_glasses = [
        r for r in rows
        if r.method == "slow_zero_shot"
        and r.corruption == "glasses"
        and r.k_shot == 0
    ]
    global_glasses = [
        r for r in rows
        if r.method == "global"
        and r.corruption == "glasses"
    ]

    slow_mean = float(np.mean([r.accuracy for r in slow_glasses]))
    global_mean = float(np.mean([r.accuracy for r in global_glasses]))

    lines.append(f"slow zero-shot glasses mean accuracy = {slow_mean:.4f}")
    lines.append(f"global glasses mean accuracy        = {global_mean:.4f}")
    lines.append("")

    for method in methods:
        lines.append(method.upper())

        ks = sorted(set(
            r.k_shot for r in rows
            if r.method == method and r.k_shot > 0
        ))

        for k in ks:
            ss = [
                r for r in rows
                if r.method == method
                and r.corruption == "glasses"
                and r.k_shot == k
            ]
            if not ss:
                continue

            acc = np.array([r.accuracy for r in ss], dtype=float)
            clean = np.array([r.clean_accuracy_after_adaptation for r in ss], dtype=float)
            release = np.array([r.release_A for r in ss], dtype=float)
            recover = np.array([r.recovery_BC for r in ss], dtype=float)
            grad = np.array([r.initial_backup_grad_norm for r in ss], dtype=float)
            cond = np.array([r.initial_route_conductance for r in ss], dtype=float)
            alpha = np.array([r.alpha_after for r in ss], dtype=float)
            sat = np.array([r.saturation_rate for r in ss], dtype=float)
            w = np.array([
                [r.w_A_eye, r.w_B_lower_geometry, r.w_C_chroma]
                for r in ss
            ], dtype=float)

            item = {
                "method": method,
                "k_shot": k,
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std()),
                "gain_over_slow_zero_shot": float(acc.mean() - slow_mean),
                "clean_accuracy_after_mean": float(clean.mean()),
                "release_A_mean": float(release.mean()),
                "recovery_BC_mean": float(recover.mean()),
                "initial_backup_grad_norm_mean": float(grad.mean()),
                "initial_route_conductance_mean": float(cond.mean()),
                "alpha_after_mean": float(alpha.mean()),
                "saturation_rate_mean": float(sat.mean()),
                "w_A_mean": float(w[:, 0].mean()),
                "w_B_mean": float(w[:, 1].mean()),
                "w_C_mean": float(w[:, 2].mean()),
            }
            summary_rows.append(item)

            lines.append(
                f"k={k:2d} "
                f"acc={acc.mean():.4f}±{acc.std():.4f} "
                f"gain={acc.mean()-slow_mean:+.4f} "
                f"clean={clean.mean():.4f} "
                f"gradBC={grad.mean():.3e} "
                f"cond={cond.mean():.3e} "
                f"releaseA={release.mean():+.3f} "
                f"alpha={alpha.mean():.3f}"
            )

        lines.append("")

    # Best non-oracle methods.
    candidates = [
        x for x in summary_rows
        if x["method"] not in ("oracle",)
    ]
    best_by_method = []
    for method in [m for m in methods if m != "oracle"]:
        mm = [x for x in candidates if x["method"] == method]
        if mm:
            best_by_method.append(max(mm, key=lambda x: x["accuracy_mean"]))

    best_by_method.sort(
        key=lambda x: x["accuracy_mean"],
        reverse=True,
    )

    lines.append("BEST K PER NON-ORACLE METHOD")
    for x in best_by_method:
        lines.append(
            f"{x['method']:15s} "
            f"k={x['k_shot']:2d} "
            f"acc={x['accuracy_mean']:.4f} "
            f"gain={x['gain_over_slow_zero_shot']:+.4f} "
            f"gradBC={x['initial_backup_grad_norm_mean']:.3e} "
            f"cond={x['initial_route_conductance_mean']:.3e} "
            f"clean={x['clean_accuracy_after_mean']:.4f}"
        )

    return "\n".join(lines), summary_rows


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--output", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])

    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-n", type=int, default=2600)
    p.add_argument("--val-n", type=int, default=700)
    p.add_argument("--test-n", type=int, default=1000)

    p.add_argument("--cue-b-strength", type=float, default=0.72)
    p.add_argument("--cue-c-strength", type=float, default=0.66)

    p.add_argument("--expert-epochs", type=int, default=6)
    p.add_argument("--expert-lr", type=float, default=0.002)
    p.add_argument("--global-epochs", type=int, default=4)
    p.add_argument("--global-lr", type=float, default=0.08)

    p.add_argument("--meta-episodes", type=int, default=300)
    p.add_argument("--support-size", type=int, default=20)
    p.add_argument("--query-size", type=int, default=40)
    p.add_argument("--inner-lr", type=float, default=0.06)
    p.add_argument("--meta-lr", type=float, default=0.001)
    p.add_argument("--expert-dropout", type=float, default=0.35)
    p.add_argument("--entropy-floor", type=float, default=0.42)
    p.add_argument("--entropy-lambda", type=float, default=0.06)
    p.add_argument("--log-every", type=int, default=25)

    p.add_argument("--k-shots", type=int, nargs="+", default=[2, 4, 8, 16, 32])

    # Experiment #2-compatible slow adaptation baseline.
    p.add_argument("--test-adapt-steps", type=int, default=12)
    p.add_argument("--test-adapt-lr", type=float, default=0.03)

    # Independent fast path.
    p.add_argument("--fast-steps", type=int, default=40)
    p.add_argument("--fast-lr", type=float, default=0.08)
    p.add_argument("--fixed-alpha", type=float, default=0.50)
    p.add_argument("--learned-alpha-init", type=float, default=0.10)

    # Diagnostic oracle.
    p.add_argument("--oracle-steps", type=int, default=120)
    p.add_argument("--oracle-lr", type=float, default=0.08)

    args = p.parse_args()

    if not (0.0 < args.fixed_alpha <= 1.0):
        p.error("--fixed-alpha must be in (0, 1].")
    if not (0.0 < args.learned_alpha_init < 1.0):
        p.error("--learned-alpha-init must be in (0, 1).")

    return args


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        all_rows.extend(run_seed(args, seed, out))

    if not all_rows:
        raise RuntimeError("No result rows produced.")

    with open(out / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(all_rows[0]).keys()),
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow(asdict(r))

    summary_text, summary_rows = summarize(all_rows)

    with open(out / "per_method_summary.csv", "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(
                f,
                fieldnames=list(summary_rows[0].keys()),
            )
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)

    payload = {
        "config": vars(args),
        "rows": [asdict(r) for r in all_rows],
        "summary_rows": summary_rows,
        "summary": summary_text,
    }

    (out / "results.json").write_text(json.dumps(payload, indent=2))
    (out / "summary.txt").write_text(summary_text + "\n")

    print("\n" + "="*92)
    print("FINAL SUMMARY")
    print("="*92)
    print(summary_text)
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
