
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF


# =============================================================================
# Reproducibility / device
# =============================================================================

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


# =============================================================================
# Synthetic evidence world
# =============================================================================

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

    def __len__(self):
        return self.n

    @staticmethod
    def maybe_flip(rng: random.Random, p_correct: float, y: int) -> int:
        return y if rng.random() < p_correct else 1 - y

    def base_image(self, idx: int) -> Tuple[Image.Image, int]:
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

        # A: dominant, fragile eye cue.
        eye_y = int(.34*s)
        r = max(2, s // 20)
        ax = int((.34 if y == 0 else .66) * s)
        dx = int((.66 if y == 0 else .34) * s)
        draw.ellipse([ax-r, eye_y-r, ax+r, eye_y+r], fill=(246, 246, 246))
        draw.ellipse([dx-r, eye_y-r, dx+r, eye_y+r], fill=(72, 72, 72))

        # B: weaker stable lower geometry.
        by = self.maybe_flip(rng, self.cue_b_strength, y)
        cx, cy = int(.50*s), int(.70*s)
        span = int(.18*s)
        if by == 0:
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

        # C: weaker stable chroma.
        cy_label = self.maybe_flip(rng, self.cue_c_strength, y)
        arr = np.asarray(img).astype(np.float32)
        delta = 8.0
        if cy_label == 0:
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
            # Held out from slow-router training.
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

        raise ValueError(kind)

    def __getitem__(self, idx):
        img, y = self.base_image(idx)
        img = self.corrupt(img, idx)
        x = TF.to_tensor(img.resize((self.s, self.s), Image.Resampling.BILINEAR))
        return x, y


# =============================================================================
# Feature-isolated experts
# =============================================================================

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

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


class ExpertBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.names = list(FEATURES)
        self.experts = nn.ModuleDict({
            name: TinyExpert(FEATURES[name][1]) for name in self.names
        })

    def forward_all(self, x):
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

    def context_features(self, x, expert_logits):
        image_z = self.encoder(x).flatten(1)
        p = expert_logits.softmax(-1)
        conf = p.max(-1).values
        margin = (p[..., 1] - p[..., 0]).abs()
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1)
        return torch.cat([image_z, conf, margin, entropy], dim=1)

    def forward(self, x, expert_logits):
        return self.mlp(self.context_features(x, expert_logits))


def fuse(expert_logits, weights):
    return (expert_logits * weights.unsqueeze(-1)).sum(1)


def entropy(w):
    return -(w * w.clamp_min(1e-8).log()).sum(1)


# =============================================================================
# Training helpers
# =============================================================================

def make_loader(ds, batch, shuffle=False):
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


def train_experts(bank, train_dl, val_dl, device, epochs, lr):
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=1e-4)
    best, best_score = None, -1.0

    for ep in range(1, epochs + 1):
        bank.train()
        total = loss_sum = 0.0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            ex = bank.forward_all(x)
            loss = sum(F.cross_entropy(ex[:, i], y) for i in range(ex.shape[1])) / ex.shape[1]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += y.numel()
            loss_sum += loss.item() * y.numel()

        accs = standalone_accuracies(bank, val_dl, device)
        score = sum(accs.values()) / len(accs)
        print(f"[experts] epoch={ep:02d} loss={loss_sum/max(1,total):.4f} val={accs}")

        if score > best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in bank.state_dict().items()}

    if best is not None:
        bank.load_state_dict(best)


def train_global(bank, train_dl, device, epochs, lr):
    for p in bank.parameters():
        p.requires_grad_(False)

    logits = nn.Parameter(torch.zeros(len(bank.names), device=device))
    opt = torch.optim.Adam([logits], lr=lr)

    for ep in range(1, epochs + 1):
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

        print(f"[global] epoch={ep:02d} weights={logits.softmax(0).detach().cpu().numpy().round(3).tolist()}")

    return logits.detach().clone()


def meta_train_slow(bank, gate, args, device, seed):
    corrs = ["clean", "eye_patch", "lower_patch", "desaturate", "blur", "jpeg"]
    rng = random.Random(seed + 5000)
    outer = torch.optim.AdamW(gate.parameters(), lr=args.meta_lr, weight_decay=1e-4)

    for p in bank.parameters():
        p.requires_grad_(False)

    for ep in range(1, args.meta_episodes + 1):
        corr = corrs[rng.randrange(len(corrs))]
        ds = SyntheticEvidenceDataset(
            args.train_n, args.image_size, seed + 100, corr,
            args.cue_b_strength, args.cue_c_strength,
        )
        ids = list(range(len(ds)))
        rng.shuffle(ids)
        sids = ids[:args.support_size]
        qids = ids[args.support_size:args.support_size + args.query_size]

        fast = copy.deepcopy(gate).to(device)
        inner = torch.optim.SGD(fast.parameters(), lr=args.inner_lr)

        x, y = next(iter(make_loader(Subset(ds, sids), args.support_size, True)))
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            ex = bank.forward_all(x)

        z = fast(x, ex)
        if rng.random() < args.expert_dropout:
            z = z.clone()
            z[:, rng.randrange(ex.shape[1])] -= 6.0

        w = z.softmax(1)
        out = fuse(ex, w)
        ce = F.cross_entropy(out, y)
        ent = entropy(w).mean()
        target = args.entropy_floor * math.log(ex.shape[1])
        loss = ce + args.entropy_lambda * F.relu(torch.tensor(target, device=device) - ent)

        inner.zero_grad(set_to_none=True)
        loss.backward()
        inner.step()

        xq, yq = next(iter(make_loader(Subset(ds, qids), args.query_size, False)))
        xq, yq = xq.to(device), yq.to(device)
        with torch.no_grad():
            exq = bank.forward_all(xq)

        qout = fuse(exq, fast(xq, exq).softmax(1))
        qloss = F.cross_entropy(qout, yq)

        fast.zero_grad(set_to_none=True)
        qloss.backward()

        outer.zero_grad(set_to_none=True)
        for p, fp in zip(gate.parameters(), fast.parameters()):
            if fp.grad is not None:
                p.grad = fp.grad.detach().clone()
        outer.step()

        if ep == 1 or ep % args.log_every == 0:
            print(f"[meta] {ep:04d}/{args.meta_episodes} corr={corr:12s} qloss={qloss.item():.4f}")


def balanced_indices(ds, n, seed):
    by = {0: [], 1: []}
    for i in range(len(ds)):
        _, y = ds[i]
        by[int(y)].append(i)

    rng = random.Random(seed)
    each = n // 2
    out = []

    for c in (0, 1):
        rng.shuffle(by[c])
        out += by[c][:each]

    # For odd n, take one extra deterministically.
    if len(out) < n:
        extra_class = n % 2
        out.append(by[extra_class][each])

    return out[:n]


# =============================================================================
# Frozen context features and fast policy
# =============================================================================

@torch.no_grad()
def collect_context(bank, gate, dl, device):
    hs, ys = [], []
    bank.eval()
    gate.eval()

    for x, y in dl:
        x = x.to(device)
        ex = bank.forward_all(x)
        h = gate.context_features(x, ex)
        hs.append(h.detach())
        ys.append(y.to(device))

    return torch.cat(hs, 0), torch.cat(ys, 0)


def adapt_fast_logits(bank, support_dl, device, args):
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

    return a.detach()


def adapt_slow_gate(bank, slow_gate, support_dl, device, args):
    gate = copy.deepcopy(slow_gate).to(device)
    opt = torch.optim.SGD(gate.parameters(), lr=args.slow_adapt_lr)
    cached = [(x.to(device), y.to(device)) for x, y in support_dl]

    for _ in range(args.slow_adapt_steps):
        for x, y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)
            w = gate(x, ex).softmax(1)
            loss = F.cross_entropy(fuse(ex, w), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return gate


def learn_global_alpha(bank, slow_gate, fast_logits, support_dl, device, args):
    beta = nn.Parameter(torch.tensor(-2.1972246, device=device))  # sigmoid=0.1
    opt = torch.optim.Adam([beta], lr=args.alpha_lr)
    cached = [(x.to(device), y.to(device)) for x, y in support_dl]

    for _ in range(args.alpha_steps):
        for x, y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)
                ps = slow_gate(x, ex).softmax(1)
                pf = fast_logits.softmax(0).unsqueeze(0).expand(x.shape[0], -1)

            a = beta.sigmoid()
            w = (1-a)*ps + a*pf
            loss = F.cross_entropy(fuse(ex, w), y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return float(beta.sigmoid().detach().item())


# =============================================================================
# Prototype switch
# =============================================================================

def l2_normalize(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def prototype_scores(h, mu_new, mu_clean):
    hn = l2_normalize(h)
    mn = mu_new / mu_new.norm().clamp_min(1e-8)
    mc = mu_clean / mu_clean.norm().clamp_min(1e-8)
    return hn @ mn - hn @ mc


def fit_prototype_switch(h_support, h_clean, args):
    hs = l2_normalize(h_support)
    hc = l2_normalize(h_clean)

    mu_s = hs.mean(0)
    mu_c = hc.mean(0)

    ss = prototype_scores(h_support, mu_s, mu_c)
    sc = prototype_scores(h_clean, mu_s, mu_c)

    # Choose threshold using only adaptation data:
    # support=new regime, clean memory=old regime.
    vals = torch.sort(torch.cat([ss, sc])).values
    if vals.numel() == 1:
        thresholds = vals
    else:
        thresholds = (vals[:-1] + vals[1:]) / 2

    best_thr = thresholds[0]
    best_bal = -1.0

    for t in thresholds:
        tpr = (ss >= t).float().mean()
        tnr = (sc < t).float().mean()
        bal = 0.5 * (tpr + tnr)
        if float(bal) > best_bal:
            best_bal = float(bal)
            best_thr = t

    spread = torch.std(torch.cat([ss, sc]), unbiased=False).clamp_min(1e-4)
    temperature = max(args.prototype_min_temperature, float(spread.item()) * args.prototype_temp_scale)

    return {
        "mu_support": mu_s.detach(),
        "mu_clean": mu_c.detach(),
        "threshold": float(best_thr.detach().item()),
        "temperature": float(temperature),
        "calibration_balanced_accuracy": best_bal,
    }


def prototype_alpha(h, state):
    s = prototype_scores(h, state["mu_support"], state["mu_clean"])
    return torch.sigmoid((s - state["threshold"]) / state["temperature"])


# =============================================================================
# Linear clean-memory context switch
# =============================================================================

class LinearContextSwitch(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, h):
        return self.linear(h).squeeze(1)


def fit_memory_switch(h_support, h_clean, args, device):
    # Standardize with old-regime statistics only.
    mean = h_clean.mean(0, keepdim=True)
    std = h_clean.std(0, unbiased=False, keepdim=True).clamp_min(1e-3)

    hs = (h_support - mean) / std
    hc = (h_clean - mean) / std

    switch = LinearContextSwitch(hs.shape[1]).to(device)
    nn.init.zeros_(switch.linear.weight)
    nn.init.zeros_(switch.linear.bias)

    opt = torch.optim.AdamW(
        switch.parameters(),
        lr=args.context_lr,
        weight_decay=args.context_weight_decay,
    )

    # Balanced environment objective despite clean memory being larger.
    pos_weight = torch.tensor(
        [max(1.0, hc.shape[0] / max(1, hs.shape[0]))],
        device=device,
    )

    x = torch.cat([hs, hc], 0)
    y = torch.cat([
        torch.ones(hs.shape[0], device=device),
        torch.zeros(hc.shape[0], device=device),
    ], 0)

    for _ in range(args.context_steps):
        logits = switch(x)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=pos_weight,
        )
        opt.zero_grad(set_to_none=True)
        bce.backward()
        opt.step()

    return {
        "switch": switch,
        "mean": mean.detach(),
        "std": std.detach(),
    }


def memory_alpha(h, state):
    hz = (h - state["mean"]) / state["std"]
    return torch.sigmoid(state["switch"](hz))


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class EvalMetrics:
    accuracy: float
    loss: float
    mean_entropy: float
    saturation_rate: float
    mean_weights: List[float]
    alpha_mean: float
    activation_rate: float


@torch.no_grad()
def evaluate_method(bank, slow_gate, dl, device, method, state):
    bank.eval()
    slow_gate.eval()

    n = correct = 0
    loss_sum = entropy_sum = alpha_sum = 0.0
    activated = saturated = 0
    wsum = torch.zeros(len(bank.names), device=device)

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        ex = bank.forward_all(x)
        ps = slow_gate(x, ex).softmax(1)

        if method == "slow_zero":
            w = ps
            alpha = torch.zeros(x.shape[0], device=device)

        elif method == "slow_sgd":
            adapted = state["adapted_gate"]
            w = adapted(x, ex).softmax(1)
            alpha = torch.zeros(x.shape[0], device=device)

        elif method == "fast_only":
            pf = state["fast_logits"].softmax(0).unsqueeze(0).expand(x.shape[0], -1)
            w = pf
            alpha = torch.ones(x.shape[0], device=device)

        elif method == "global_mixture":
            pf = state["fast_logits"].softmax(0).unsqueeze(0).expand(x.shape[0], -1)
            alpha = torch.full((x.shape[0],), state["alpha"], device=device)
            w = (1-alpha.unsqueeze(1))*ps + alpha.unsqueeze(1)*pf

        elif method == "prototype_switch":
            pf = state["fast_logits"].softmax(0).unsqueeze(0).expand(x.shape[0], -1)
            h = slow_gate.context_features(x, ex)
            alpha = prototype_alpha(h, state["prototype"])
            w = (1-alpha.unsqueeze(1))*ps + alpha.unsqueeze(1)*pf

        elif method == "memory_switch":
            pf = state["fast_logits"].softmax(0).unsqueeze(0).expand(x.shape[0], -1)
            h = slow_gate.context_features(x, ex)
            alpha = memory_alpha(h, state["memory"])
            w = (1-alpha.unsqueeze(1))*ps + alpha.unsqueeze(1)*pf

        elif method == "environment_oracle":
            # Caller passes state["use_fast"] based on known evaluation environment.
            if state["use_fast"]:
                w = state["fast_logits"].softmax(0).unsqueeze(0).expand(x.shape[0], -1)
                alpha = torch.ones(x.shape[0], device=device)
            else:
                w = ps
                alpha = torch.zeros(x.shape[0], device=device)

        else:
            raise ValueError(method)

        out = fuse(ex, w)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()

        wsum += w.sum(0)
        entropy_sum += entropy(w).sum().item()
        alpha_sum += alpha.sum().item()
        activated += (alpha >= 0.5).sum().item()
        saturated += (w.max(1).values > 0.99).sum().item()

    return EvalMetrics(
        accuracy=correct / max(1, n),
        loss=loss_sum / max(1, n),
        mean_entropy=entropy_sum / max(1, n),
        saturation_rate=saturated / max(1, n),
        mean_weights=(wsum / max(1, n)).detach().cpu().tolist(),
        alpha_mean=alpha_sum / max(1, n),
        activation_rate=activated / max(1, n),
    )


@dataclass
class Row:
    seed: int
    method: str
    k_shot: int
    glasses_accuracy: float
    clean_accuracy: float
    harmonic_score: float
    min_accuracy: float
    glasses_loss: float
    clean_loss: float
    alpha_glasses: float
    alpha_clean: float
    activation_gap: float
    glasses_activation_rate: float
    clean_false_activation_rate: float
    glasses_w_A: float
    glasses_w_B: float
    glasses_w_C: float
    clean_w_A: float
    clean_w_B: float
    clean_w_C: float
    release_A_on_glasses: float
    recovery_BC_on_glasses: float
    A_eye_acc_glasses: float
    B_lower_geometry_acc_glasses: float
    C_chroma_acc_glasses: float


def harmonic(a, b):
    return 0.0 if a + b <= 0 else 2*a*b/(a+b)


def make_row(seed, method, k, gm, cm, zero_w, eacc):
    gw = gm.mean_weights
    cw = cm.mean_weights
    return Row(
        seed=seed,
        method=method,
        k_shot=k,
        glasses_accuracy=gm.accuracy,
        clean_accuracy=cm.accuracy,
        harmonic_score=harmonic(gm.accuracy, cm.accuracy),
        min_accuracy=min(gm.accuracy, cm.accuracy),
        glasses_loss=gm.loss,
        clean_loss=cm.loss,
        alpha_glasses=gm.alpha_mean,
        alpha_clean=cm.alpha_mean,
        activation_gap=gm.alpha_mean - cm.alpha_mean,
        glasses_activation_rate=gm.activation_rate,
        clean_false_activation_rate=cm.activation_rate,
        glasses_w_A=gw[0],
        glasses_w_B=gw[1],
        glasses_w_C=gw[2],
        clean_w_A=cw[0],
        clean_w_B=cw[1],
        clean_w_C=cw[2],
        release_A_on_glasses=zero_w[0] - gw[0],
        recovery_BC_on_glasses=(gw[1]+gw[2]) - (zero_w[1]+zero_w[2]),
        A_eye_acc_glasses=eacc["A_eye"],
        B_lower_geometry_acc_glasses=eacc["B_lower_geometry"],
        C_chroma_acc_glasses=eacc["C_chroma"],
    )


# =============================================================================
# Seed run
# =============================================================================

def run_seed(args, seed, out_dir):
    print("\n" + "="*92)
    print(f"SEED {seed}")
    print("="*92)

    seed_everything(seed)
    device = pick_device(args.device)
    print(f"device={device}; torch={torch.__version__}; mps_available={torch.backends.mps.is_available()}")

    train = SyntheticEvidenceDataset(
        args.train_n, args.image_size, seed+10, "clean",
        args.cue_b_strength, args.cue_c_strength,
    )
    val = SyntheticEvidenceDataset(
        args.val_n, args.image_size, seed+20, "clean",
        args.cue_b_strength, args.cue_c_strength,
    )

    bank = ExpertBank().to(device)

    print("\nStage 1/4 — experts")
    train_experts(
        bank,
        make_loader(train, args.batch_size, True),
        make_loader(val, args.batch_size, False),
        device,
        args.expert_epochs,
        args.expert_lr,
    )

    print("\nStage 2/4 — global fusion")
    global_logits = train_global(
        bank,
        make_loader(train, args.batch_size, True),
        device,
        args.global_epochs,
        args.global_lr,
    )

    print("\nStage 3/4 — one slow router")
    slow_gate = SlowGate(len(bank.names)).to(device)
    with torch.no_grad():
        slow_gate.mlp[-1].bias.copy_(global_logits)

    meta_train_slow(bank, slow_gate, args, device, seed)

    clean_test = SyntheticEvidenceDataset(
        args.test_n, args.image_size, seed+30, "clean",
        args.cue_b_strength, args.cue_c_strength,
    )
    glasses_test = SyntheticEvidenceDataset(
        args.test_n, args.image_size, seed+30, "glasses",
        args.cue_b_strength, args.cue_c_strength,
    )

    clean_dl = make_loader(clean_test, args.batch_size, False)
    glasses_dl = make_loader(glasses_test, args.batch_size, False)

    eacc = standalone_accuracies(bank, glasses_dl, device)

    slow_g = evaluate_method(bank, slow_gate, glasses_dl, device, "slow_zero", {})
    slow_c = evaluate_method(bank, slow_gate, clean_dl, device, "slow_zero", {})
    zero_w = slow_g.mean_weights

    print(f"[slow glasses] acc={slow_g.accuracy:.4f} weights={np.round(zero_w, 4).tolist()}")
    print(f"[slow clean]   acc={slow_c.accuracy:.4f}")

    rows = []

    print("\nStage 4/4 — selective fast-path ablation")

    for k in args.k_shots:
        # Exact same support/query split for every method.
        sids = balanced_indices(glasses_test, k, seed+k)
        sidset = set(sids)
        qids = [i for i in range(len(glasses_test)) if i not in sidset]

        support_ds = Subset(glasses_test, sids)
        query_ds = Subset(glasses_test, qids)
        support_dl = make_loader(support_ds, min(k, args.batch_size), True)
        support_context_dl = make_loader(support_ds, min(k, args.batch_size), False)
        query_dl = make_loader(query_ds, args.batch_size, False)

        # Clean reference memory comes from validation, never clean test.
        memory_ids = balanced_indices(
            val,
            min(args.clean_memory_size, len(val)),
            seed + 10000 + k,
        )
        clean_memory_ds = Subset(val, memory_ids)
        clean_memory_dl = make_loader(clean_memory_ds, args.batch_size, False)

        # One identical fast policy shared by all context-switch methods.
        fast_logits = adapt_fast_logits(bank, support_dl, device, args)

        adapted_slow = adapt_slow_gate(bank, slow_gate, support_dl, device, args)
        global_alpha = learn_global_alpha(
            bank, slow_gate, fast_logits, support_dl, device, args
        )

        h_support, _ = collect_context(bank, slow_gate, support_context_dl, device)
        h_clean, _ = collect_context(bank, slow_gate, clean_memory_dl, device)

        prototype_state = fit_prototype_switch(h_support, h_clean, args)
        memory_state = fit_memory_switch(h_support, h_clean, args, device)

        method_states = {
            "slow_sgd": {"adapted_gate": adapted_slow},
            "fast_only": {"fast_logits": fast_logits},
            "global_mixture": {
                "fast_logits": fast_logits,
                "alpha": global_alpha,
            },
            "prototype_switch": {
                "fast_logits": fast_logits,
                "prototype": prototype_state,
            },
            "memory_switch": {
                "fast_logits": fast_logits,
                "memory": memory_state,
            },
        }

        for method, state in method_states.items():
            gm = evaluate_method(
                bank, slow_gate, query_dl, device, method, state
            )
            cm = evaluate_method(
                bank, slow_gate, clean_dl, device, method, state
            )
            row = make_row(seed, method, k, gm, cm, zero_w, eacc)
            rows.append(row)

            print(
                f"[k={k:02d} {method:16s}] "
                f"G={gm.accuracy:.4f} C={cm.accuracy:.4f} "
                f"H={row.harmonic_score:.4f} "
                f"aG={gm.alpha_mean:.3f} aC={cm.alpha_mean:.3f} "
                f"FPR={cm.activation_rate:.3f}"
            )

        # Diagnostic environment oracle.
        gm = evaluate_method(
            bank, slow_gate, query_dl, device,
            "environment_oracle",
            {"fast_logits": fast_logits, "use_fast": True},
        )
        cm = evaluate_method(
            bank, slow_gate, clean_dl, device,
            "environment_oracle",
            {"fast_logits": fast_logits, "use_fast": False},
        )
        rows.append(make_row(
            seed, "environment_oracle", k, gm, cm, zero_w, eacc
        ))

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


# =============================================================================
# Aggregate summary
# =============================================================================

def summarize(rows: List[Row]):
    methods = [
        "slow_sgd",
        "fast_only",
        "global_mixture",
        "prototype_switch",
        "memory_switch",
        "environment_oracle",
    ]

    lines = []
    summary_rows = []

    for method in methods:
        lines.append(method.upper())
        ks = sorted(set(r.k_shot for r in rows if r.method == method))

        for k in ks:
            ss = [r for r in rows if r.method == method and r.k_shot == k]
            if not ss:
                continue

            def arr(field):
                return np.array([getattr(r, field) for r in ss], dtype=float)

            g = arr("glasses_accuracy")
            c = arr("clean_accuracy")
            h = arr("harmonic_score")
            mn = arr("min_accuracy")
            ag = arr("alpha_glasses")
            ac = arr("alpha_clean")
            gap = arr("activation_gap")
            tpr = arr("glasses_activation_rate")
            fpr = arr("clean_false_activation_rate")
            rel = arr("release_A_on_glasses")

            item = {
                "method": method,
                "k_shot": k,
                "glasses_accuracy_mean": float(g.mean()),
                "glasses_accuracy_std": float(g.std()),
                "clean_accuracy_mean": float(c.mean()),
                "clean_accuracy_std": float(c.std()),
                "harmonic_score_mean": float(h.mean()),
                "harmonic_score_std": float(h.std()),
                "min_accuracy_mean": float(mn.mean()),
                "alpha_glasses_mean": float(ag.mean()),
                "alpha_clean_mean": float(ac.mean()),
                "activation_gap_mean": float(gap.mean()),
                "glasses_activation_rate_mean": float(tpr.mean()),
                "clean_false_activation_rate_mean": float(fpr.mean()),
                "release_A_mean": float(rel.mean()),
            }
            summary_rows.append(item)

            lines.append(
                f"k={k:2d} "
                f"G={g.mean():.4f}±{g.std():.4f} "
                f"C={c.mean():.4f}±{c.std():.4f} "
                f"H={h.mean():.4f} "
                f"min={mn.mean():.4f} "
                f"aG={ag.mean():.3f} "
                f"aC={ac.mean():.3f} "
                f"TPR={tpr.mean():.3f} "
                f"FPR={fpr.mean():.3f}"
            )
        lines.append("")

    # Rank deployable methods by best harmonic score.
    deployable = [
        "slow_sgd",
        "fast_only",
        "global_mixture",
        "prototype_switch",
        "memory_switch",
    ]
    best = []

    for method in deployable:
        xs = [x for x in summary_rows if x["method"] == method]
        if xs:
            best.append(max(xs, key=lambda x: x["harmonic_score_mean"]))

    best.sort(key=lambda x: x["harmonic_score_mean"], reverse=True)

    lines.append("BEST K BY STABILITY-PLASTICITY HARMONIC SCORE")
    for x in best:
        lines.append(
            f"{x['method']:16s} "
            f"k={x['k_shot']:2d} "
            f"H={x['harmonic_score_mean']:.4f} "
            f"G={x['glasses_accuracy_mean']:.4f} "
            f"C={x['clean_accuracy_mean']:.4f} "
            f"aGap={x['activation_gap_mean']:+.3f}"
        )

    return "\n".join(lines), summary_rows


# =============================================================================
# CLI
# =============================================================================

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

    p.add_argument("--slow-adapt-steps", type=int, default=12)
    p.add_argument("--slow-adapt-lr", type=float, default=0.03)

    p.add_argument("--fast-steps", type=int, default=40)
    p.add_argument("--fast-lr", type=float, default=0.08)

    p.add_argument("--alpha-steps", type=int, default=40)
    p.add_argument("--alpha-lr", type=float, default=0.08)

    p.add_argument("--clean-memory-size", type=int, default=128)

    p.add_argument("--prototype-temp-scale", type=float, default=0.35)
    p.add_argument("--prototype-min-temperature", type=float, default=0.03)

    p.add_argument("--context-steps", type=int, default=200)
    p.add_argument("--context-lr", type=float, default=0.03)
    p.add_argument("--context-weight-decay", type=float, default=0.02)

    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        all_rows.extend(run_seed(args, seed, out))

    if not all_rows:
        raise RuntimeError("No rows produced.")

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))

    summary_text, summary_rows = summarize(all_rows)

    with open(out / "per_method_summary.csv", "w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

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
