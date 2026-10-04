
#!/usr/bin/env python3
"""
Adaptive Feature Reweighting experiment.

Core question:
Can a model learn to rapidly change the importance of feature families for a
new, held-out corruption using only a few labeled support examples, without
retraining the main classifier and without a larger teacher?

Default mode uses a fully synthetic binary image dataset so the experiment is
self-contained. Folder mode accepts a real/fake ImageFolder dataset.

The test corruption "glasses" is NEVER used during meta-training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# ----------------------------- Reproducibility ----------------------------- #

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


# ------------------------------ Image helpers ------------------------------ #

def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    return tfm(img.convert("RGB"))


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(x)


def jpeg_roundtrip(img: Image.Image, quality: int = 25) -> Image.Image:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def corrupt_pil(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size

    if kind == "clean":
        return img

    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.4))

    if kind == "jpeg":
        return jpeg_roundtrip(img, quality=24)

    if kind == "brightness":
        arr = np.asarray(img).astype(np.float32) / 255.0
        factor = rng.uniform(0.55, 0.75)
        arr = np.clip(arr * factor, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    if kind == "bottom_occlusion":
        out = img.copy()
        draw = ImageDraw.Draw(out)
        y0 = int(h * 0.68)
        draw.rectangle([0, y0, w, h], fill=(rng.randint(0, 35),) * 3)
        return out

    if kind == "glasses":
        # Deliberately held out from meta-training.
        # A simple "glasses-like" occluder: two dark lenses + bridge + temples.
        out = img.copy()
        draw = ImageDraw.Draw(out)
        cy = int(h * 0.38)
        lens_w = int(w * 0.24)
        lens_h = int(h * 0.15)
        gap = int(w * 0.05)
        left_x0 = w // 2 - gap // 2 - lens_w
        right_x0 = w // 2 + gap // 2
        y0 = cy - lens_h // 2
        y1 = y0 + lens_h
        alpha_fill = (18, 18, 18)
        outline = (3, 3, 3)
        draw.rounded_rectangle([left_x0, y0, left_x0 + lens_w, y1],
                               radius=max(2, lens_h // 5),
                               fill=alpha_fill, outline=outline, width=max(1, w // 64))
        draw.rounded_rectangle([right_x0, y0, right_x0 + lens_w, y1],
                               radius=max(2, lens_h // 5),
                               fill=alpha_fill, outline=outline, width=max(1, w // 64))
        draw.line([left_x0 + lens_w, cy, right_x0, cy],
                  fill=outline, width=max(1, w // 48))
        draw.line([0, cy, left_x0, cy], fill=outline, width=max(1, w // 64))
        draw.line([right_x0 + lens_w, cy, w, cy], fill=outline, width=max(1, w // 64))
        return out

    raise ValueError(f"Unknown corruption: {kind}")


# ------------------------------- Datasets ---------------------------------- #

class SyntheticBinaryDataset(Dataset):
    """
    Self-contained toy dataset.

    Class 0 and class 1 differ in several cues:
      - broad color/layout cue (easy / globally useful)
      - edge orientation cue
      - small local cue around the upper-middle region

    The held-out 'glasses' corruption damages the local region and some edges,
    forcing a good gate to rely more on other branches after few-shot adaptation.
    """

    def __init__(self, n: int, image_size: int, seed: int, corruption: str = "clean"):
        self.n = n
        self.image_size = image_size
        self.seed = seed
        self.corruption = corruption

    def __len__(self) -> int:
        return self.n

    def _make_one(self, idx: int) -> Tuple[Image.Image, int]:
        rng = random.Random(self.seed + idx * 1009)
        s = self.image_size
        y = idx % 2

        base = np.zeros((s, s, 3), dtype=np.uint8)
        # Weak noisy background.
        noise = np.clip(rng.gauss(0, 1) * 0 + np.random.default_rng(self.seed + idx).normal(0, 13, (s, s, 3)), -28, 28)
        base[:] = 112
        base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(base)
        draw = ImageDraw.Draw(img)

        # Broad cue: class-dependent left/right illumination.
        if y == 0:
            draw.rectangle([0, 0, s // 2, s], fill=(105, 124, 142))
            draw.rectangle([s // 2, 0, s, s], fill=(122, 111, 101))
        else:
            draw.rectangle([0, 0, s // 2, s], fill=(122, 111, 101))
            draw.rectangle([s // 2, 0, s, s], fill=(105, 124, 142))

        # Edge cue.
        pad = s // 7
        if y == 0:
            draw.line([pad, s - pad, s - pad, pad], fill=(235, 235, 235), width=max(2, s // 24))
        else:
            draw.line([pad, pad, s - pad, s - pad], fill=(235, 235, 235), width=max(2, s // 24))

        # Local cue around "eye" region.
        cy = int(s * 0.38)
        r = max(2, s // 18)
        cx = int(s * (0.37 if y == 0 else 0.63))
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(250, 230, 40))

        # Re-add light sensor noise.
        arr = np.asarray(img).astype(np.float32)
        rng_np = np.random.default_rng(self.seed * 17 + idx)
        arr += rng_np.normal(0, 5.0, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        rng2 = random.Random(self.seed * 31 + idx)
        img = corrupt_pil(img, self.corruption, rng2)
        return img, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, y = self._make_one(idx)
        return pil_to_tensor(img, self.image_size), y


class CorruptedImageFolder(Dataset):
    def __init__(self, root: str, split: str, image_size: int, corruption: str, seed: int):
        split_root = Path(root) / split
        if not split_root.exists():
            raise FileNotFoundError(
                f"Missing split directory: {split_root}\n"
                f"Expected e.g. {split_root}/real and {split_root}/fake"
            )
        self.ds = datasets.ImageFolder(split_root)
        if len(self.ds.classes) != 2:
            raise ValueError(f"Expected exactly 2 classes, got {self.ds.classes}")
        self.image_size = image_size
        self.corruption = corruption
        self.seed = seed

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, y = self.ds.samples[idx]
        img = Image.open(path).convert("RGB")
        img = corrupt_pil(img, self.corruption, random.Random(self.seed + idx))
        return pil_to_tensor(img, self.image_size), y


def make_dataset(args, split: str, corruption: str):
    if args.dataset == "synthetic":
        n = {"train": args.synthetic_train, "val": args.synthetic_val, "test": args.synthetic_test}[split]
        split_seed = {"train": args.seed + 10, "val": args.seed + 20, "test": args.seed + 30}[split]
        return SyntheticBinaryDataset(n, args.image_size, split_seed, corruption)
    return CorruptedImageFolder(args.data, split, args.image_size, corruption, args.seed)


# ---------------------------- Feature families ----------------------------- #

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    g = rgb_to_gray(x)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device).view(1,1,3,3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return mag.clamp(0, 1)


def highpass(x: torch.Tensor) -> torch.Tensor:
    blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
    hp = (x - blur).abs()
    return hp.clamp(0, 1)


def chroma(x: torch.Tensor) -> torch.Tensor:
    # Crude opponent-color channels: R-G, B-G, R-B.
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    c = torch.cat([(r-g).abs(), (b-g).abs(), (r-b).abs()], dim=1)
    return c.clamp(0, 1)


FEATURE_FNS = {
    "rgb": lambda x: x,
    "edge": sobel_edges,
    "highpass": highpass,
    "chroma": chroma,
}


class TinyExpert(nn.Module):
    def __init__(self, in_ch: int, width: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 3, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 3, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x).flatten(1)
        return self.head(z), z


class MultiExpertModel(nn.Module):
    def __init__(self, expert_names: Sequence[str]):
        super().__init__()
        self.expert_names = list(expert_names)
        channels = {"rgb": 3, "edge": 1, "highpass": 3, "chroma": 3}
        self.experts = nn.ModuleDict({
            name: TinyExpert(channels[name]) for name in self.expert_names
        })

    def forward_all(self, x: torch.Tensor):
        logits, feats = [], []
        for name in self.expert_names:
            xi = FEATURE_FNS[name](x)
            li, zi = self.experts[name](xi)
            logits.append(li)
            feats.append(zi)
        return torch.stack(logits, dim=1), feats  # [B,E,C]


class Gate(nn.Module):
    """
    Per-sample router.
    It sees a cheap summary of the raw input plus expert confidence statistics.
    It outputs one weight per expert.
    """
    def __init__(self, n_experts: int):
        super().__init__()
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 12, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(12, 20, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(20 + n_experts * 3, 48),
            nn.GELU(),
            nn.Linear(48, n_experts),
        )

    def forward(self, x: torch.Tensor, expert_logits: torch.Tensor) -> torch.Tensor:
        img_z = self.img_encoder(x).flatten(1)
        probs = expert_logits.softmax(dim=-1)
        conf = probs.max(dim=-1).values
        margin = (probs[..., 1] - probs[..., 0]).abs()
        entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)
        stats = torch.cat([conf, margin, entropy], dim=1)
        gate_logits = self.mlp(torch.cat([img_z, stats], dim=1))
        return gate_logits


def fused_logits(expert_logits: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    w = gate_logits.softmax(dim=1)
    return (expert_logits * w.unsqueeze(-1)).sum(dim=1)


# ---------------------------- Training routines ---------------------------- #

@torch.no_grad()
def evaluate(model, gate, loader, device, mode: str = "adaptive") -> Dict:
    model.eval()
    gate.eval()
    total = correct = 0
    loss_sum = 0.0
    weight_sum = torch.zeros(len(model.expert_names), device=device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ex_logits, _ = model.forward_all(x)

        if mode == "uniform":
            gate_logits = torch.zeros(x.shape[0], len(model.expert_names), device=device)
        elif mode == "global":
            # learned global weights are stored as gate.global_logits if present
            gl = getattr(gate, "global_logits")
            gate_logits = gl.unsqueeze(0).expand(x.shape[0], -1)
        else:
            gate_logits = gate(x, ex_logits)

        out = fused_logits(ex_logits, gate_logits)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        weight_sum += gate_logits.softmax(dim=1).sum(dim=0)

    return {
        "accuracy": correct / max(1, total),
        "loss": loss_sum / max(1, total),
        "mean_weights": (weight_sum / max(1, total)).detach().cpu().tolist(),
        "n": total,
    }


def train_experts(model, loader, val_loader, device, epochs: int, lr: float):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = None
    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            ex_logits, _ = model.forward_all(x)
            # Train every expert to be independently useful.
            loss = sum(F.cross_entropy(ex_logits[:, i], y) for i in range(ex_logits.shape[1])) / ex_logits.shape[1]
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.numel()
            total += y.numel()

        # Simple uniform validation.
        dummy = Gate(len(model.expert_names)).to(device)
        m = evaluate(model, dummy, val_loader, device, mode="uniform")
        print(f"[experts] epoch={epoch:02d} train_loss={total_loss/max(1,total):.4f} val_acc={m['accuracy']:.4f}")
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)


def train_global_weights(model, gate, loader, device, epochs: int, lr: float):
    for p in model.parameters():
        p.requires_grad_(False)

    gate.global_logits = nn.Parameter(torch.zeros(len(model.expert_names), device=device))
    opt = torch.optim.Adam([gate.global_logits], lr=lr)

    for epoch in range(1, epochs + 1):
        total = 0
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                ex_logits, _ = model.forward_all(x)
            gl = gate.global_logits.unsqueeze(0).expand(x.shape[0], -1)
            out = fused_logits(ex_logits, gl)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += y.numel()
            loss_sum += loss.item() * y.numel()
        print(f"[global] epoch={epoch:02d} loss={loss_sum/max(1,total):.4f} "
              f"weights={gate.global_logits.softmax(0).detach().cpu().numpy().round(3).tolist()}")


def make_loader(ds, batch_size: int, shuffle: bool, workers: int):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=False)


def meta_train_gate(
    model, gate, args, device,
    train_corruptions: Sequence[str],
):
    """
    First-order episodic meta-training.

    For each episode:
      1. Clone the gate.
      2. Do one/few gradient steps on a small support batch from one corruption.
      3. Evaluate query loss on a fresh batch from same corruption.
      4. First-order update the original gate toward the adapted gate's query gradient.

    This teaches the gate an initialization that is easy to adapt.
    It is intentionally lightweight and MPS-friendly.
    """
    import copy

    for p in model.parameters():
        p.requires_grad_(False)

    outer_opt = torch.optim.AdamW(gate.parameters(), lr=args.meta_lr, weight_decay=1e-4)
    rng = random.Random(args.seed + 999)

    for ep in range(1, args.meta_episodes + 1):
        corruption = rng.choice(list(train_corruptions))
        ds = make_dataset(args, "train", corruption)

        # Sample disjoint support/query indices.
        n = len(ds)
        idx = list(range(n))
        rng.shuffle(idx)
        s_idx = idx[:args.support_size]
        q_idx = idx[args.support_size:args.support_size + args.query_size]
        s_loader = make_loader(Subset(ds, s_idx), args.support_size, False, 0)
        q_loader = make_loader(Subset(ds, q_idx), args.query_size, False, 0)

        fast_gate = copy.deepcopy(gate).to(device)
        inner_opt = torch.optim.SGD(fast_gate.parameters(), lr=args.inner_lr)

        fast_gate.train()
        for _ in range(args.inner_steps):
            x, y = next(iter(s_loader))
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                ex_logits, _ = model.forward_all(x)
            out = fused_logits(ex_logits, fast_gate(x, ex_logits))
            loss = F.cross_entropy(out, y)
            inner_opt.zero_grad(set_to_none=True)
            loss.backward()
            inner_opt.step()

        # Query gradient on fast gate.
        xq, yq = next(iter(q_loader))
        xq, yq = xq.to(device), yq.to(device)
        with torch.no_grad():
            exq, _ = model.forward_all(xq)
        qout = fused_logits(exq, fast_gate(xq, exq))
        qloss = F.cross_entropy(qout, yq)

        fast_gate.zero_grad(set_to_none=True)
        qloss.backward()

        # First-order MAML-style transfer of gradients.
        outer_opt.zero_grad(set_to_none=True)
        for p, fp in zip(gate.parameters(), fast_gate.parameters()):
            if fp.grad is not None:
                p.grad = fp.grad.detach().clone()
        outer_opt.step()

        if ep == 1 or ep % args.log_every == 0:
            print(f"[meta] episode={ep:04d}/{args.meta_episodes} corruption={corruption} query_loss={qloss.item():.4f}")


def adapt_gate_on_support(model, gate, support_loader, device, steps: int, lr: float):
    import copy
    adapted = copy.deepcopy(gate).to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.SGD(adapted.parameters(), lr=lr)
    adapted.train()

    # Cache support to make exact K-shot adaptation deterministic.
    support_batches = [(x.to(device), y.to(device)) for x, y in support_loader]
    for _ in range(steps):
        for x, y in support_batches:
            with torch.no_grad():
                ex_logits, _ = model.forward_all(x)
            out = fused_logits(ex_logits, adapted(x, ex_logits))
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return adapted


# ------------------------------ Experiment --------------------------------- #

@dataclass
class ResultRow:
    corruption: str
    method: str
    k_shot: int
    accuracy: float
    loss: float
    mean_weights: List[float]


def balanced_support_indices(ds: Dataset, k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for i in range(len(ds)):
        _, y = ds[i]
        by_class[int(y)].append(i)
    each = max(1, k // 2)
    out = []
    for c in (0, 1):
        rng.shuffle(by_class[c])
        out.extend(by_class[c][:each])
    return out[:k]


def run(args):
    seed_everything(args.seed)
    device = pick_device(args.device)
    print(f"device={device}")
    print(f"torch={torch.__version__}")
    if device.type == "mps":
        print("MPS available: yes")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    expert_names = ["rgb", "edge", "highpass", "chroma"]
    model = MultiExpertModel(expert_names).to(device)
    gate = Gate(len(expert_names)).to(device)

    clean_train = make_dataset(args, "train", "clean")
    clean_val = make_dataset(args, "val", "clean")
    train_loader = make_loader(clean_train, args.batch_size, True, args.workers)
    val_loader = make_loader(clean_val, args.batch_size, False, args.workers)

    print("\n=== Stage 1: train feature experts on CLEAN data only ===")
    train_experts(model, train_loader, val_loader, device, args.expert_epochs, args.expert_lr)

    print("\n=== Stage 2: learn one global fusion vector ===")
    train_global_weights(model, gate, train_loader, device, args.global_epochs, args.global_lr)

    global_weights = gate.global_logits.detach().clone()

    # Initialize adaptive gate near the global solution by biasing final layer.
    with torch.no_grad():
        gate.mlp[-1].bias.copy_(global_weights)

    print("\n=== Stage 3: meta-train the gate on perturbations EXCLUDING 'glasses' ===")
    meta_corruptions = ["clean", "blur", "jpeg", "brightness", "bottom_occlusion"]
    meta_train_gate(model, gate, args, device, meta_corruptions)

    results: List[ResultRow] = []

    print("\n=== Stage 4: evaluate held-out perturbation: GLASSES ===")
    for corruption in ["clean", "glasses"]:
        test_ds = make_dataset(args, "test", corruption)
        test_loader = make_loader(test_ds, args.batch_size, False, args.workers)

        # Uniform baseline.
        m = evaluate(model, gate, test_loader, device, mode="uniform")
        results.append(ResultRow(corruption, "uniform", 0, m["accuracy"], m["loss"], m["mean_weights"]))

        # Global fixed weighting baseline.
        m = evaluate(model, gate, test_loader, device, mode="global")
        results.append(ResultRow(corruption, "global", 0, m["accuracy"], m["loss"], m["mean_weights"]))

        # Meta-trained adaptive gate, no test-time adaptation.
        m = evaluate(model, gate, test_loader, device, mode="adaptive")
        results.append(ResultRow(corruption, "adaptive_zero_shot", 0, m["accuracy"], m["loss"], m["mean_weights"]))

        if corruption == "glasses":
            for k in args.k_shots:
                support_idx = balanced_support_indices(test_ds, k, args.seed + k)
                query_idx = [i for i in range(len(test_ds)) if i not in set(support_idx)]
                support_loader = make_loader(Subset(test_ds, support_idx), min(k, args.batch_size), True, 0)
                query_loader = make_loader(Subset(test_ds, query_idx), args.batch_size, False, args.workers)

                adapted = adapt_gate_on_support(
                    model, gate, support_loader, device,
                    steps=args.test_adapt_steps, lr=args.test_adapt_lr
                )
                m = evaluate(model, adapted, query_loader, device, mode="adaptive")
                results.append(ResultRow(corruption, "adaptive_few_shot", k, m["accuracy"], m["loss"], m["mean_weights"]))
                print(f"[held-out glasses] k={k:02d} acc={m['accuracy']:.4f} "
                      f"weights={np.round(m['mean_weights'], 3).tolist()}")

    # Save artifacts.
    rows = [asdict(r) for r in results]
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "config": vars(args),
            "experts": expert_names,
            "results": rows,
        }, f, indent=2)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "method", "k_shot", "accuracy", "loss", *[f"w_{n}" for n in expert_names]])
        for r in results:
            w.writerow([r.corruption, r.method, r.k_shot, r.accuracy, r.loss, *r.mean_weights])

    torch.save({
        "model": model.state_dict(),
        "gate": gate.state_dict(),
        "global_logits": gate.global_logits.detach().cpu(),
        "experts": expert_names,
        "config": vars(args),
    }, out_dir / "checkpoint.pt")

    print("\n=== Summary ===")
    print("experts:", expert_names)
    for r in results:
        print(f"{r.corruption:8s} | {r.method:18s} | k={r.k_shot:2d} | "
              f"acc={r.accuracy:.4f} | weights={np.round(r.mean_weights, 3).tolist()}")

    print(f"\nSaved to: {out_dir.resolve()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "folder"], default="synthetic")
    p.add_argument("--data", type=str, default="")
    p.add_argument("--output", type=str, default="results")
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--synthetic-train", type=int, default=2400)
    p.add_argument("--synthetic-val", type=int, default=600)
    p.add_argument("--synthetic-test", type=int, default=800)

    p.add_argument("--expert-epochs", type=int, default=6)
    p.add_argument("--expert-lr", type=float, default=2e-3)
    p.add_argument("--global-epochs", type=int, default=3)
    p.add_argument("--global-lr", type=float, default=0.08)

    p.add_argument("--meta-episodes", type=int, default=220)
    p.add_argument("--support-size", type=int, default=16)
    p.add_argument("--query-size", type=int, default=32)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=0.08)
    p.add_argument("--meta-lr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=20)

    p.add_argument("--k-shots", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--test-adapt-steps", type=int, default=8)
    p.add_argument("--test-adapt-lr", type=float, default=0.04)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
