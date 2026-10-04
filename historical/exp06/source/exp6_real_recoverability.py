
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)


EXPERT_NAMES = ["rgb", "highpass", "chroma", "structure"]
SEEN_ROUTER_CONDITIONS = [
    "clean",
    "blur",
    "jpeg",
    "dark",
    "eye_patch",
    "desaturate",
]


# =============================================================================
# General utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(name)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. In Colab select a T4 GPU runtime.")
    return d


def safe_json(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Dataset and intervention
# =============================================================================

def balanced_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = sorted(np.unique(labels).tolist())
    if len(classes) != 2:
        raise ValueError(f"Expected binary labels, got {classes}")

    per = n // 2
    selected = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        if len(idx) < per:
            raise ValueError(f"Not enough class-{c} samples for n={n}")
        selected.append(rng.choice(idx, size=per, replace=False))

    if n % 2:
        c = classes[seed % 2]
        idx = np.setdiff1d(np.flatnonzero(labels == c), selected[classes.index(c)])
        selected.append(rng.choice(idx, size=1, replace=False))

    out = np.concatenate(selected)
    rng.shuffle(out)
    return out.astype(np.int64)


def subset_hf(ds, split: str, n: int, seed: int):
    part = ds[split]
    labels = np.asarray(part["label"], dtype=np.int64)
    ids = balanced_indices(labels, min(n, len(labels)), seed)
    return part.select(ids.tolist()), ids


def _det_rng(key: int, salt: int = 0) -> random.Random:
    return random.Random(99173 + key * 104729 + salt * 7919)


def add_glasses(img: Image.Image, key: int) -> Image.Image:
    """
    Procedural eyewear on a face-centered crop.

    This is intentionally a controlled intervention, not a claim of perfect
    photorealistic eyewear synthesis. Several style parameters are randomized
    deterministically by sample key.
    """
    rng = _det_rng(key, 41)
    out = img.convert("RGBA")
    w, h = out.size

    # RetinaFace-centered crops are usually stable enough for a relative anchor.
    center_x = 0.50 * w + rng.uniform(-0.018, 0.018) * w
    center_y = 0.395 * h + rng.uniform(-0.018, 0.018) * h

    total_w = rng.uniform(0.55, 0.66) * w
    lens_h = rng.uniform(0.14, 0.19) * h
    gap = rng.uniform(0.035, 0.060) * w
    lens_w = (total_w - gap) / 2.0
    angle = rng.uniform(-3.0, 3.0)

    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    x1 = center_x - gap/2 - lens_w
    x2 = center_x - gap/2
    x3 = center_x + gap/2
    x4 = center_x + gap/2 + lens_w
    y1 = center_y - lens_h/2
    y2 = center_y + lens_h/2

    frame_alpha = rng.randint(210, 255)
    frame_gray = rng.randint(5, 35)
    frame = (frame_gray, frame_gray, frame_gray, frame_alpha)
    lens_alpha = rng.randint(35, 95)
    tint = rng.choice([
        (10, 10, 10, lens_alpha),
        (28, 38, 48, lens_alpha),
        (50, 36, 24, lens_alpha),
    ])
    thickness = max(2, int(round(rng.uniform(0.010, 0.018) * w)))
    radius = max(3, int(0.035 * w))

    for box in ([x1, y1, x2, y2], [x3, y1, x4, y2]):
        box = [int(v) for v in box]
        d.rounded_rectangle(box, radius=radius, fill=tint, outline=frame, width=thickness)

    bridge_y = int(center_y)
    d.line(
        [(int(x2), bridge_y), (int(x3), bridge_y)],
        fill=frame,
        width=thickness,
    )

    temple_y = int(center_y - rng.uniform(-0.01, 0.01) * h)
    d.line(
        [(int(x1), temple_y), (int(0.10*w), int(temple_y-0.025*h))],
        fill=frame,
        width=max(1, thickness-1),
    )
    d.line(
        [(int(x4), temple_y), (int(0.90*w), int(temple_y-0.025*h))],
        fill=frame,
        width=max(1, thickness-1),
    )

    # Small highlight makes the lenses less like opaque rectangles.
    if rng.random() < 0.75:
        hi = (255, 255, 255, rng.randint(35, 85))
        for sx in (x1, x3):
            d.line(
                [(int(sx+0.18*lens_w), int(y1+0.18*lens_h)),
                 (int(sx+0.50*lens_w), int(y1+0.38*lens_h))],
                fill=hi,
                width=max(1, thickness//2),
            )

    if abs(angle) > 0.2:
        overlay = overlay.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            center=(center_x, center_y),
        )

    out = Image.alpha_composite(out, overlay).convert("RGB")
    return out


def apply_condition(img: Image.Image, condition: str, key: int) -> Image.Image:
    img = img.convert("RGB")
    rng = _det_rng(key, 7)
    w, h = img.size

    if condition == "clean":
        return img

    if condition == "glasses":
        return add_glasses(img, key)

    if condition == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 2.2)))

    if condition == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(22, 42))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if condition == "dark":
        return ImageEnhance.Brightness(img).enhance(rng.uniform(0.45, 0.68))

    if condition == "desaturate":
        g = ImageOps.grayscale(img)
        return Image.merge("RGB", (g, g, g))

    if condition == "eye_patch":
        out = img.copy()
        d = ImageDraw.Draw(out)
        y0, y1 = int(0.27*h), int(0.50*h)
        x0, x1 = int(0.16*w), int(0.84*w)
        gray = rng.randint(70, 160)
        d.rectangle([x0, y0, x1, y1], fill=(gray, gray, gray))
        return out

    if condition == "lower_patch":
        out = img.copy()
        d = ImageDraw.Draw(out)
        gray = rng.randint(70, 160)
        d.rectangle(
            [int(0.20*w), int(0.56*h), int(0.80*w), int(0.90*h)],
            fill=(gray, gray, gray),
        )
        return out

    raise ValueError(f"Unknown condition: {condition}")


class HFImageDataset(TorchDataset):
    def __init__(
        self,
        hf_subset,
        condition: str = "clean",
        image_size: int = 224,
        augment: bool = False,
        base_key: int = 0,
    ):
        self.ds = hf_subset
        self.condition = condition
        self.image_size = image_size
        self.augment = augment
        self.base_key = base_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]
        img = row["image"].convert("RGB")
        key = self.base_key + int(idx)
        img = apply_condition(img, self.condition, key)

        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)

        img = TF.resize(
            img,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        x = TF.to_tensor(img)
        y = int(row["label"])
        return x, y, key


# =============================================================================
# Experts
# =============================================================================

class FeatureExpert(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        if kind not in EXPERT_NAMES:
            raise ValueError(kind)
        self.kind = kind
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(dim, 2)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def feature_view(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "rgb":
            v = x

        elif self.kind == "highpass":
            blur = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
            residual = x - blur
            v = 0.5 + 0.5 * torch.tanh(7.0 * residual)

        elif self.kind == "chroma":
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            y = 0.299*r + 0.587*g + 0.114*b
            cb = (0.5 + 0.7*(b-y)).clamp(0, 1)
            cr = (0.5 + 0.7*(r-y)).clamp(0, 1)
            v = torch.cat([y, cb, cr], dim=1)

        elif self.kind == "structure":
            gray = (
                0.299*x[:, 0:1]
                + 0.587*x[:, 1:2]
                + 0.114*x[:, 2:3]
            )
            kx = torch.tensor(
                [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                device=x.device,
                dtype=x.dtype,
            ).view(1, 1, 3, 3)
            ky = kx.transpose(-1, -2)
            gx = F.conv2d(gray, kx, padding=1)
            gy = F.conv2d(gray, ky, padding=1)
            mag = torch.sqrt(gx*gx + gy*gy + 1e-8)
            mag = (mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-6)).clamp(0, 1)
            lap = gray - F.avg_pool2d(gray, 3, 1, 1)
            lap = 0.5 + 0.5*torch.tanh(8.0*lap)
            v = torch.cat([gray, mag, lap], dim=1)

        else:
            raise RuntimeError(self.kind)

        return (v - self.mean) / self.std

    def forward(self, x):
        v = self.feature_view(x)
        emb = self.backbone(v)
        logits = self.head(emb)
        return logits, emb


@torch.no_grad()
def predict_expert(model, dl, device, amp=True):
    model.eval()
    all_logits, all_labels = [], []
    for x, y, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(amp and device.type == "cuda"),
        ):
            logits, _ = model(x)
        all_logits.append(logits.float().cpu())
        all_labels.append(torch.as_tensor(y))
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def binary_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    p = t.softmax(-1).numpy()
    pred = p.argmax(1)
    # Dataset convention: 0=fake, 1=real. Treat fake as positive for AUC/AP.
    y_fake = (np.asarray(labels) == 0).astype(np.int64)
    fake_score = p[:, 0]
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "auc_fake": float(roc_auc_score(y_fake, fake_score)),
        "ap_fake": float(average_precision_score(y_fake, fake_score)),
        "nll": float(F.cross_entropy(t, torch.from_numpy(labels).long()).item()),
    }


def train_one_expert(
    kind: str,
    train_subset,
    val_subset,
    ckpt_path: Path,
    device: torch.device,
    args,
):
    if ckpt_path.exists() and not args.force_retrain:
        print(f"[expert:{kind}] resume checkpoint: {ckpt_path}")
        return

    seed_everything(args.model_seed + EXPERT_NAMES.index(kind)*1000)

    train_ds = HFImageDataset(
        train_subset,
        "clean",
        args.image_size,
        augment=True,
        base_key=100000,
    )
    val_ds = HFImageDataset(
        val_subset,
        "clean",
        args.image_size,
        augment=False,
        base_key=200000,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    model = FeatureExpert(kind).to(device)
    model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=args.expert_lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_auc = -1.0
    best_state = None

    for epoch in range(1, args.expert_epochs + 1):
        model.train()
        total = loss_sum = 0.0
        tic = time.time()

        for x, y, _ in train_dl:
            x = x.to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += loss.item() * y.numel()
            total += y.numel()

        logits, labels = predict_expert(model, val_dl, device)
        met = binary_metrics_from_logits(logits, labels)

        print(
            f"[expert:{kind}] epoch={epoch:02d} "
            f"train_loss={loss_sum/max(1,total):.4f} "
            f"val_auc={met['auc_fake']:.4f} "
            f"val_acc={met['accuracy']:.4f} "
            f"time={time.time()-tic:.1f}s"
        )

        if met["auc_fake"] > best_auc:
            best_auc = met["auc_fake"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    torch.save(
        {
            "kind": kind,
            "state_dict": best_state,
            "best_val_auc": best_auc,
            "model_seed": args.model_seed,
        },
        ckpt_path,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_expert(kind: str, ckpt_path: Path, device: torch.device):
    model = FeatureExpert(kind)
    payload = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model = model.to(device).eval()
    model = model.to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# =============================================================================
# Caching
# =============================================================================

def cache_path(cache_dir: Path, split_name: str, condition: str) -> Path:
    return cache_dir / f"{split_name}__{condition}.npz"


@torch.no_grad()
def build_cache(
    models: Dict[str, FeatureExpert],
    hf_subset,
    split_name: str,
    condition: str,
    out_path: Path,
    device: torch.device,
    args,
    base_key: int,
):
    if out_path.exists() and not args.force_recache:
        print(f"[cache] resume {out_path.name}")
        return

    ds = HFImageDataset(
        hf_subset,
        condition,
        args.image_size,
        augment=False,
        base_key=base_key,
    )
    dl = DataLoader(
        ds,
        batch_size=args.cache_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    logits_all = []
    rgb_emb_all = []
    labels_all = []
    keys_all = []

    for step, (x, y, key) in enumerate(dl):
        x = x.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )

        per = []
        rgb_emb = None

        for kind in EXPERT_NAMES:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits, emb = models[kind](x)

            per.append(logits.float().cpu())
            if kind == "rgb":
                rgb_emb = emb.float().cpu()

        logits_all.append(torch.stack(per, dim=1))
        rgb_emb_all.append(rgb_emb)
        labels_all.append(torch.as_tensor(y))
        keys_all.append(torch.as_tensor(key))

        if step % 50 == 0:
            print(
                f"[cache:{split_name}:{condition}] "
                f"{min((step+1)*args.cache_batch_size, len(ds))}/{len(ds)}"
            )

    arr_logits = torch.cat(logits_all).numpy().astype(np.float32)
    arr_emb = torch.cat(rgb_emb_all).numpy().astype(np.float16)
    arr_labels = torch.cat(labels_all).numpy().astype(np.int64)
    arr_keys = torch.cat(keys_all).numpy().astype(np.int64)

    np.savez_compressed(
        out_path,
        logits=arr_logits,
        rgb_emb=arr_emb,
        labels=arr_labels,
        keys=arr_keys,
        split=split_name,
        condition=condition,
    )
    print(f"[cache] saved {out_path}")


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {
        "logits": z["logits"].astype(np.float32),
        "rgb_emb": z["rgb_emb"].astype(np.float32),
        "labels": z["labels"].astype(np.int64),
        "keys": z["keys"].astype(np.int64),
    }


def context_from_cache(c: Dict[str, np.ndarray]) -> np.ndarray:
    logits = torch.from_numpy(c["logits"])
    p = logits.softmax(-1)
    conf = p.max(-1).values
    margin = (p[..., 1] - p[..., 0]).abs()
    ent = -(p * p.clamp_min(1e-8).log()).sum(-1)
    stats = torch.cat([conf, margin, ent], dim=1).numpy()
    return np.concatenate([c["rgb_emb"], stats], axis=1).astype(np.float32)


# =============================================================================
# Fusion and router
# =============================================================================

def fuse_logits(expert_logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (expert_logits * weights.unsqueeze(-1)).sum(1)


def train_global_fusion(train_cache: Dict[str, np.ndarray], device, steps=500):
    ex = torch.from_numpy(train_cache["logits"]).to(device)
    y = torch.from_numpy(train_cache["labels"]).long().to(device)

    z = nn.Parameter(torch.zeros(ex.shape[1], device=device))
    opt = torch.optim.Adam([z], lr=0.05)

    for _ in range(steps):
        w = z.softmax(0).unsqueeze(0).expand(ex.shape[0], -1)
        loss = F.cross_entropy(fuse_logits(ex, w), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return z.detach()


class SlowRouter(nn.Module):
    def __init__(self, input_dim: int, n_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_experts),
        )

    def forward(self, x):
        return self.net(x)


def fit_context_scaler(train_cache):
    x = context_from_cache(train_cache)
    mean = x.mean(0, keepdims=True)
    std = x.std(0, keepdims=True)
    std = np.maximum(std, 1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_context(c, mean, std):
    return ((context_from_cache(c) - mean) / std).astype(np.float32)


def train_slow_router(
    meta_caches: List[Dict[str, np.ndarray]],
    context_mean,
    context_std,
    global_logits,
    device,
    args,
):
    xs, exs, ys = [], [], []
    for c in meta_caches:
        xs.append(normalize_context(c, context_mean, context_std))
        exs.append(c["logits"])
        ys.append(c["labels"])

    x = torch.from_numpy(np.concatenate(xs)).float()
    ex = torch.from_numpy(np.concatenate(exs)).float()
    y = torch.from_numpy(np.concatenate(ys)).long()

    ds = TensorDataset(x, ex, y)
    dl = DataLoader(ds, batch_size=512, shuffle=True)

    gate = SlowRouter(x.shape[1], ex.shape[1]).to(device)
    with torch.no_grad():
        gate.net[-1].bias.copy_(global_logits.to(device))

    opt = torch.optim.AdamW(
        gate.parameters(),
        lr=args.router_lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, args.router_epochs + 1):
        gate.train()
        loss_sum = n = 0

        for xb, eb, yb in dl:
            xb, eb, yb = xb.to(device), eb.to(device), yb.to(device)
            z = gate(xb)

            # Preserve alternative paths during router training.
            if random.random() < args.router_expert_dropout:
                j = random.randrange(eb.shape[1])
                z = z.clone()
                z[:, j] -= 5.0

            w = z.softmax(1)
            out = fuse_logits(eb, w)
            ce = F.cross_entropy(out, yb)

            h = -(w * w.clamp_min(1e-8).log()).sum(1).mean()
            h_floor = args.router_entropy_floor * math.log(eb.shape[1])
            reg = F.relu(torch.tensor(h_floor, device=device) - h)

            loss = ce + args.router_entropy_lambda * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * yb.numel()
            n += yb.numel()

        if epoch == 1 or epoch == args.router_epochs or epoch % 5 == 0:
            print(f"[router] epoch={epoch:02d} loss={loss_sum/max(1,n):.4f}")

    return gate


# =============================================================================
# Evaluation and adaptation
# =============================================================================

def metric_dict(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logits = logits.detach().cpu().float()
    labels = labels.detach().cpu().long()
    p = logits.softmax(-1).numpy()
    y = labels.numpy()
    pred = p.argmax(1)
    y_fake = (y == 0).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "auc_fake": float(roc_auc_score(y_fake, p[:, 0])),
        "ap_fake": float(average_precision_score(y_fake, p[:, 0])),
        "nll": float(F.cross_entropy(logits, labels).item()),
    }


def cache_tensors(cache, mean, std, device):
    ctx = torch.from_numpy(
        normalize_context(cache, mean, std)
    ).float().to(device)
    ex = torch.from_numpy(cache["logits"]).float().to(device)
    y = torch.from_numpy(cache["labels"]).long().to(device)
    return ctx, ex, y


@torch.no_grad()
def eval_weight_fn(ctx, ex, y, weight_fn):
    ws = []
    outs = []
    bs = 2048
    for s in range(0, len(y), bs):
        xb = ctx[s:s+bs]
        eb = ex[s:s+bs]
        w = weight_fn(xb, eb)
        outs.append(fuse_logits(eb, w))
        ws.append(w)
    out = torch.cat(outs)
    w = torch.cat(ws)
    met = metric_dict(out, y)
    met.update({
        "mean_weights": w.mean(0).detach().cpu().tolist(),
        "mean_entropy": float(
            (-(w*w.clamp_min(1e-8).log()).sum(1)).mean().item()
        ),
        "saturation_rate": float(
            (w.max(1).values > 0.99).float().mean().item()
        ),
    })
    return met, w, out


def support_indices(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    return balanced_indices(labels, k, seed)


def complement_indices(n: int, chosen: np.ndarray) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(chosen)] = False
    return np.flatnonzero(mask)


def adapt_slow(
    gate,
    ctx,
    ex,
    y,
    args,
):
    g = copy.deepcopy(gate)
    g.train()
    opt = torch.optim.SGD(g.parameters(), lr=args.slow_adapt_lr)

    for _ in range(args.slow_adapt_steps):
        w = g(ctx).softmax(1)
        loss = F.cross_entropy(fuse_logits(ex, w), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    g.eval()
    return g


def adapt_fast_logits(ex, y, steps, lr):
    a = nn.Parameter(torch.zeros(ex.shape[1], device=ex.device))
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(steps):
        w = a.softmax(0).unsqueeze(0).expand(ex.shape[0], -1)
        loss = F.cross_entropy(fuse_logits(ex, w), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return a.detach()


def adapt_additive_fast(gate, ctx, ex, y, args):
    with torch.no_grad():
        slow_z = gate(ctx)
    a = nn.Parameter(torch.zeros(ex.shape[1], device=ex.device))
    opt = torch.optim.Adam([a], lr=args.fast_lr)
    for _ in range(args.fast_steps):
        w = (slow_z + a.unsqueeze(0)).softmax(1)
        loss = F.cross_entropy(fuse_logits(ex, w), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return a.detach()


def learn_global_alpha(gate, fast_logits, ctx, ex, y, args):
    with torch.no_grad():
        ps = gate(ctx).softmax(1)
        pf = fast_logits.softmax(0).unsqueeze(0).expand(ex.shape[0], -1)

    beta = nn.Parameter(torch.tensor(-2.1972246, device=ex.device))
    opt = torch.optim.Adam([beta], lr=args.alpha_lr)

    for _ in range(args.alpha_steps):
        alpha = beta.sigmoid()
        w = (1-alpha)*ps + alpha*pf
        loss = F.cross_entropy(fuse_logits(ex, w), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return float(beta.sigmoid().detach().item())


class EnvSwitch(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def train_memory_switch(pos_ctx, neg_ctx, args):
    d = pos_ctx.shape[1]
    switch = EnvSwitch(d).to(pos_ctx.device)
    nn.init.zeros_(switch.linear.weight)
    nn.init.zeros_(switch.linear.bias)

    x = torch.cat([pos_ctx, neg_ctx], 0)
    y = torch.cat([
        torch.ones(pos_ctx.shape[0], device=x.device),
        torch.zeros(neg_ctx.shape[0], device=x.device),
    ])

    pos_weight = torch.tensor(
        [max(1.0, neg_ctx.shape[0] / max(1, pos_ctx.shape[0]))],
        device=x.device,
    )

    opt = torch.optim.AdamW(
        switch.parameters(),
        lr=args.switch_lr,
        weight_decay=args.switch_weight_decay,
    )

    for _ in range(args.switch_steps):
        logit = switch(x)
        loss = F.binary_cross_entropy_with_logits(
            logit,
            y,
            pos_weight=pos_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    switch.eval()
    return switch


def fit_prototype(pos_ctx, neg_ctx):
    pos = F.normalize(pos_ctx, dim=1)
    neg = F.normalize(neg_ctx, dim=1)
    mu_p = F.normalize(pos.mean(0), dim=0)
    mu_n = F.normalize(neg.mean(0), dim=0)

    s_pos = pos @ mu_p - pos @ mu_n
    s_neg = neg @ mu_p - neg @ mu_n

    vals = torch.sort(torch.cat([s_pos, s_neg])).values
    if len(vals) > 1:
        thresholds = (vals[:-1] + vals[1:]) / 2
    else:
        thresholds = vals

    best_thr = thresholds[0]
    best_bal = -1.0
    for t in thresholds:
        tpr = (s_pos >= t).float().mean()
        tnr = (s_neg < t).float().mean()
        bal = float((0.5*(tpr+tnr)).item())
        if bal > best_bal:
            best_bal = bal
            best_thr = t

    spread = torch.std(torch.cat([s_pos, s_neg]), unbiased=False)
    temp = max(0.03, float(spread.item()) * 0.35)

    return {
        "mu_pos": mu_p.detach(),
        "mu_neg": mu_n.detach(),
        "threshold": float(best_thr.item()),
        "temperature": temp,
        "calibration_bal_acc": best_bal,
    }


def prototype_alpha(ctx, state):
    x = F.normalize(ctx, dim=1)
    score = x @ state["mu_pos"] - x @ state["mu_neg"]
    return torch.sigmoid(
        (score - state["threshold"]) / state["temperature"]
    )


def method_eval(
    name,
    gate,
    fast_logits,
    additive_logits,
    global_alpha,
    memory_switch,
    prototype_state,
    ctx,
    ex,
    y,
):
    if name == "slow_zero":
        fn = lambda x, e: gate(x).softmax(1)

    elif name == "fast_only":
        pf = fast_logits.softmax(0)
        fn = lambda x, e: pf.unsqueeze(0).expand(x.shape[0], -1)

    elif name == "additive_fast":
        def fn(x, e):
            return (gate(x) + additive_logits.unsqueeze(0)).softmax(1)

    elif name == "global_mixture":
        pf = fast_logits.softmax(0)
        def fn(x, e):
            ps = gate(x).softmax(1)
            return (1-global_alpha)*ps + global_alpha*pf.unsqueeze(0)

    elif name == "memory_switch":
        pf = fast_logits.softmax(0)
        def fn(x, e):
            ps = gate(x).softmax(1)
            a = torch.sigmoid(memory_switch(x))
            return (1-a.unsqueeze(1))*ps + a.unsqueeze(1)*pf.unsqueeze(0)

    elif name == "prototype_switch":
        pf = fast_logits.softmax(0)
        def fn(x, e):
            ps = gate(x).softmax(1)
            a = prototype_alpha(x, prototype_state)
            return (1-a.unsqueeze(1))*ps + a.unsqueeze(1)*pf.unsqueeze(0)

    else:
        raise ValueError(name)

    return eval_weight_fn(ctx, ex, y, fn)


def alpha_stats(name, memory_switch, prototype_state, ctx):
    if name == "memory_switch":
        with torch.no_grad():
            a = torch.sigmoid(memory_switch(ctx))
    elif name == "prototype_switch":
        with torch.no_grad():
            a = prototype_alpha(ctx, prototype_state)
    elif name == "fast_only":
        a = torch.ones(ctx.shape[0], device=ctx.device)
    else:
        return {"alpha_mean": float("nan"), "activation_rate": float("nan")}

    return {
        "alpha_mean": float(a.mean().item()),
        "activation_rate": float((a >= 0.5).float().mean().item()),
    }


# =============================================================================
# Preview
# =============================================================================

def save_glasses_preview(test_subset, out_path: Path, n=12):
    n = min(n, len(test_subset))
    cols = 4
    rows = math.ceil(n / cols)
    thumb = 180

    canvas = Image.new("RGB", (cols*thumb*2, rows*thumb), "white")

    for i in range(n):
        img = test_subset[i]["image"].convert("RGB")
        gla = add_glasses(img, 300000+i)

        for j, im in enumerate((img, gla)):
            im = ImageOps.fit(im, (thumb, thumb))
            x = (i % cols)*thumb*2 + j*thumb
            y = (i // cols)*thumb
            canvas.paste(im, (x, y))

    canvas.save(out_path, quality=95)


# =============================================================================
# Main
# =============================================================================

@dataclass
class ResultRow:
    routing_seed: int
    k_shot: int
    method: str
    condition: str
    accuracy: float
    auc_fake: float
    ap_fake: float
    nll: float
    mean_entropy: float
    saturation_rate: float
    w_rgb: float
    w_highpass: float
    w_chroma: float
    w_structure: float
    alpha_mean: float
    activation_rate: float


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset-id",
        default="ThinothW/Deepfake-Identity-Isolated-Dataset-PreP",
    )
    p.add_argument("--work-dir", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--train-n", type=int, default=8000)
    p.add_argument("--val-n", type=int, default=2000)
    p.add_argument("--test-n", type=int, default=3000)

    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument(
        "--routing-seeds",
        type=int,
        nargs="+",
        default=[11, 22, 33, 44, 55],
    )

    p.add_argument("--expert-epochs", type=int, default=2)
    p.add_argument("--expert-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cache-batch-size", type=int, default=96)
    p.add_argument("--workers", type=int, default=2)

    p.add_argument("--router-epochs", type=int, default=20)
    p.add_argument("--router-lr", type=float, default=8e-4)
    p.add_argument("--router-expert-dropout", type=float, default=0.25)
    p.add_argument("--router-entropy-floor", type=float, default=0.25)
    p.add_argument("--router-entropy-lambda", type=float, default=0.03)

    p.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
    )

    p.add_argument("--slow-adapt-steps", type=int, default=15)
    p.add_argument("--slow-adapt-lr", type=float, default=0.02)

    p.add_argument("--fast-steps", type=int, default=80)
    p.add_argument("--fast-lr", type=float, default=0.06)

    p.add_argument("--alpha-steps", type=int, default=60)
    p.add_argument("--alpha-lr", type=float, default=0.08)

    p.add_argument("--clean-memory-size", type=int, default=128)
    p.add_argument("--switch-steps", type=int, default=250)
    p.add_argument("--switch-lr", type=float, default=0.02)
    p.add_argument("--switch-weight-decay", type=float, default=0.02)

    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--force-recache", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.model_seed)
    device = choose_device(args.device)

    work = ensure_dir(Path(args.work_dir))
    checkpoints = ensure_dir(work / "checkpoints")
    caches = ensure_dir(work / "caches")
    results_dir = ensure_dir(work / "results")

    print("="*100)
    print("Experiment #6 — Real-Data Recoverability")
    print("="*100)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"vram={props.total_memory / 1024**3:.1f} GiB")
    print(f"dataset={args.dataset_id}")
    print(f"work_dir={work}")
    print()

    print("Stage 0 — load real dataset")
    ds = load_dataset(args.dataset_id)

    train_subset, train_source_ids = subset_hf(
        ds, "train", args.train_n, args.model_seed + 1
    )
    val_subset, val_source_ids = subset_hf(
        ds, "validation", args.val_n, args.model_seed + 2
    )
    test_subset, test_source_ids = subset_hf(
        ds, "test", args.test_n, args.model_seed + 3
    )

    manifest = {
        "dataset_id": args.dataset_id,
        "label_convention": {"0": "fake", "1": "real"},
        "model_seed": args.model_seed,
        "train_source_ids": train_source_ids.tolist(),
        "val_source_ids": val_source_ids.tolist(),
        "test_source_ids": test_source_ids.tolist(),
        "args": vars(args),
    }
    (work / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=safe_json)
    )

    save_glasses_preview(
        test_subset,
        work / "glasses_intervention_preview.jpg",
        n=12,
    )
    print(f"saved intervention preview: {work / 'glasses_intervention_preview.jpg'}")

    print("\nStage 1 — train/resume four real-image experts")
    for kind in EXPERT_NAMES:
        train_one_expert(
            kind,
            train_subset,
            val_subset,
            checkpoints / f"expert_{kind}.pt",
            device,
            args,
        )

    print("\nStage 2 — load frozen experts")
    models = {
        kind: load_expert(
            kind,
            checkpoints / f"expert_{kind}.pt",
            device,
        )
        for kind in EXPERT_NAMES
    }

    print("\nStage 3 — build/resume representation caches")
    specs = [
        ("train", train_subset, "clean", 100000),
        ("val", val_subset, "clean", 200000),
        ("test", test_subset, "clean", 300000),
        ("test", test_subset, "glasses", 300000),
    ]
    for cond_i, cond in enumerate(SEEN_ROUTER_CONDITIONS):
        if cond == "clean":
            continue
        specs.append(("val", val_subset, cond, 200000))

    for split_name, subset, condition, base_key in specs:
        build_cache(
            models,
            subset,
            split_name,
            condition,
            cache_path(caches, split_name, condition),
            device,
            args,
            base_key,
        )

    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_cache = load_cache(cache_path(caches, "train", "clean"))
    val_clean_cache = load_cache(cache_path(caches, "val", "clean"))
    test_clean_cache = load_cache(cache_path(caches, "test", "clean"))
    test_glasses_cache = load_cache(cache_path(caches, "test", "glasses"))

    print("\nStage 4 — standalone expert reliability")
    expert_rows = []
    for condition, c in [
        ("clean", test_clean_cache),
        ("glasses", test_glasses_cache),
    ]:
        for i, kind in enumerate(EXPERT_NAMES):
            met = binary_metrics_from_logits(
                c["logits"][:, i, :],
                c["labels"],
            )
            expert_rows.append({
                "condition": condition,
                "expert": kind,
                **met,
            })
            print(
                f"[expert {kind:9s} / {condition:7s}] "
                f"AUC={met['auc_fake']:.4f} "
                f"ACC={met['accuracy']:.4f}"
            )

    pd.DataFrame(expert_rows).to_csv(
        results_dir / "expert_reliability.csv",
        index=False,
    )

    print("\nStage 5 — learn clean global fusion + slow router")
    global_logits = train_global_fusion(
        train_cache,
        device,
        steps=500,
    )
    print(
        "global weights=",
        global_logits.softmax(0).detach().cpu().numpy().round(4).tolist(),
    )

    context_mean, context_std = fit_context_scaler(train_cache)

    meta_caches = [
        load_cache(cache_path(caches, "val", cond))
        for cond in SEEN_ROUTER_CONDITIONS
    ]

    seed_everything(args.model_seed + 999)
    slow_gate = train_slow_router(
        meta_caches,
        context_mean,
        context_std,
        global_logits,
        device,
        args,
    )
    slow_gate.eval()

    torch.save(
        {
            "state_dict": slow_gate.state_dict(),
            "input_dim": int(context_mean.shape[1]),
            "context_mean": context_mean,
            "context_std": context_std,
            "global_logits": global_logits.detach().cpu(),
        },
        checkpoints / "slow_router.pt",
    )

    clean_ctx, clean_ex, clean_y = cache_tensors(
        test_clean_cache,
        context_mean,
        context_std,
        device,
    )
    glass_ctx, glass_ex, glass_y = cache_tensors(
        test_glasses_cache,
        context_mean,
        context_std,
        device,
    )

    def global_fn(x, e):
        w = global_logits.softmax(0)
        return w.unsqueeze(0).expand(x.shape[0], -1)

    global_clean, _, _ = eval_weight_fn(
        clean_ctx, clean_ex, clean_y, global_fn
    )
    global_glasses, _, _ = eval_weight_fn(
        glass_ctx, glass_ex, glass_y, global_fn
    )

    slow_clean, _, _ = eval_weight_fn(
        clean_ctx, clean_ex, clean_y,
        lambda x, e: slow_gate(x).softmax(1),
    )
    slow_glasses, _, _ = eval_weight_fn(
        glass_ctx, glass_ex, glass_y,
        lambda x, e: slow_gate(x).softmax(1),
    )

    print(
        f"global clean/glasses AUC = "
        f"{global_clean['auc_fake']:.4f} / {global_glasses['auc_fake']:.4f}"
    )
    print(
        f"slow   clean/glasses AUC = "
        f"{slow_clean['auc_fake']:.4f} / {slow_glasses['auc_fake']:.4f}"
    )

    print("\nStage 6 — few-shot recoverability experiments")

    # Clean memory from validation, balanced by fake/real to prevent the environment
    # switch from simply learning the class.
    val_ctx, val_ex, val_y = cache_tensors(
        val_clean_cache,
        context_mean,
        context_std,
        device,
    )
    memory_ids_np = support_indices(
        val_clean_cache["labels"],
        min(args.clean_memory_size, len(val_clean_cache["labels"])),
        args.model_seed + 4242,
    )
    memory_ids = torch.from_numpy(memory_ids_np).long().to(device)
    clean_memory_ctx = val_ctx[memory_ids]

    rows: List[ResultRow] = []
    diagnostics = []

    for routing_seed in args.routing_seeds:
        seed_everything(routing_seed)

        for k in args.k_shots:
            sid_np = support_indices(
                test_glasses_cache["labels"],
                k,
                routing_seed + k,
            )
            qid_np = complement_indices(
                len(test_glasses_cache["labels"]),
                sid_np,
            )

            sid = torch.from_numpy(sid_np).long().to(device)
            qid = torch.from_numpy(qid_np).long().to(device)

            s_ctx, s_ex, s_y = glass_ctx[sid], glass_ex[sid], glass_y[sid]
            qg_ctx, qg_ex, qg_y = glass_ctx[qid], glass_ex[qid], glass_y[qid]
            qc_ctx, qc_ex, qc_y = clean_ctx[qid], clean_ex[qid], clean_y[qid]

            # All methods share identical support/query splits.
            adapted_gate = adapt_slow(
                slow_gate,
                s_ctx,
                s_ex,
                s_y,
                args,
            )
            fast_logits = adapt_fast_logits(
                s_ex,
                s_y,
                args.fast_steps,
                args.fast_lr,
            )
            additive_logits = adapt_additive_fast(
                slow_gate,
                s_ctx,
                s_ex,
                s_y,
                args,
            )
            global_alpha = learn_global_alpha(
                slow_gate,
                fast_logits,
                s_ctx,
                s_ex,
                s_y,
                args,
            )

            memory_switch = train_memory_switch(
                s_ctx,
                clean_memory_ctx,
                args,
            )
            prototype_state = fit_prototype(
                s_ctx,
                clean_memory_ctx,
            )

            method_objects = {
                "slow_sgd": {
                    "gate": adapted_gate,
                },
                "fast_only": {},
                "additive_fast": {},
                "global_mixture": {},
                "memory_switch": {},
                "prototype_switch": {},
            }

            for method in method_objects:
                gate_for_method = (
                    adapted_gate if method == "slow_sgd" else slow_gate
                )

                if method == "slow_sgd":
                    g_met, _, _ = eval_weight_fn(
                        qg_ctx, qg_ex, qg_y,
                        lambda x, e: adapted_gate(x).softmax(1),
                    )
                    c_met, _, _ = eval_weight_fn(
                        qc_ctx, qc_ex, qc_y,
                        lambda x, e: adapted_gate(x).softmax(1),
                    )
                    g_alpha = c_alpha = {
                        "alpha_mean": float("nan"),
                        "activation_rate": float("nan"),
                    }
                else:
                    g_met, _, _ = method_eval(
                        method,
                        gate_for_method,
                        fast_logits,
                        additive_logits,
                        global_alpha,
                        memory_switch,
                        prototype_state,
                        qg_ctx, qg_ex, qg_y,
                    )
                    c_met, _, _ = method_eval(
                        method,
                        gate_for_method,
                        fast_logits,
                        additive_logits,
                        global_alpha,
                        memory_switch,
                        prototype_state,
                        qc_ctx, qc_ex, qc_y,
                    )

                    if method == "global_mixture":
                        g_alpha = c_alpha = {
                            "alpha_mean": global_alpha,
                            "activation_rate": float(global_alpha >= 0.5),
                        }
                    else:
                        g_alpha = alpha_stats(
                            method,
                            memory_switch,
                            prototype_state,
                            qg_ctx,
                        )
                        c_alpha = alpha_stats(
                            method,
                            memory_switch,
                            prototype_state,
                            qc_ctx,
                        )

                for condition, met, ast in [
                    ("glasses", g_met, g_alpha),
                    ("clean", c_met, c_alpha),
                ]:
                    mw = met["mean_weights"]
                    rows.append(ResultRow(
                        routing_seed=routing_seed,
                        k_shot=k,
                        method=method,
                        condition=condition,
                        accuracy=met["accuracy"],
                        auc_fake=met["auc_fake"],
                        ap_fake=met["ap_fake"],
                        nll=met["nll"],
                        mean_entropy=met["mean_entropy"],
                        saturation_rate=met["saturation_rate"],
                        w_rgb=mw[0],
                        w_highpass=mw[1],
                        w_chroma=mw[2],
                        w_structure=mw[3],
                        alpha_mean=ast["alpha_mean"],
                        activation_rate=ast["activation_rate"],
                    ))

            diagnostics.append({
                "routing_seed": routing_seed,
                "k_shot": k,
                "support_ids": sid_np.tolist(),
                "global_alpha": global_alpha,
                "fast_weights": fast_logits.softmax(0).detach().cpu().tolist(),
                "prototype_calibration_bal_acc": prototype_state["calibration_bal_acc"],
            })

            print(
                f"[seed={routing_seed} k={k:02d}] "
                f"fast={fast_logits.softmax(0).detach().cpu().numpy().round(3).tolist()} "
                f"global_alpha={global_alpha:.3f}"
            )

    rows_df = pd.DataFrame([asdict(r) for r in rows])
    rows_df.to_csv(results_dir / "routing_results.csv", index=False)

    # Aggregate clean + glasses into one stability/plasticity table.
    merged = rows_df.pivot_table(
        index=["routing_seed", "k_shot", "method"],
        columns="condition",
        values=[
            "accuracy",
            "auc_fake",
            "alpha_mean",
            "activation_rate",
        ],
    )
    merged.columns = [f"{a}_{b}" for a, b in merged.columns]
    merged = merged.reset_index()

    merged["harmonic_accuracy"] = (
        2 * merged["accuracy_clean"] * merged["accuracy_glasses"]
        / (merged["accuracy_clean"] + merged["accuracy_glasses"]).clip(lower=1e-12)
    )
    merged["harmonic_auc"] = (
        2 * merged["auc_fake_clean"] * merged["auc_fake_glasses"]
        / (merged["auc_fake_clean"] + merged["auc_fake_glasses"]).clip(lower=1e-12)
    )
    merged["activation_gap"] = (
        merged["alpha_mean_glasses"] - merged["alpha_mean_clean"]
    )
    merged.to_csv(
        results_dir / "stability_plasticity_per_seed.csv",
        index=False,
    )

    agg = (
        merged.groupby(["method", "k_shot"])
        .agg(
            glasses_acc_mean=("accuracy_glasses", "mean"),
            glasses_acc_std=("accuracy_glasses", "std"),
            clean_acc_mean=("accuracy_clean", "mean"),
            clean_acc_std=("accuracy_clean", "std"),
            glasses_auc_mean=("auc_fake_glasses", "mean"),
            glasses_auc_std=("auc_fake_glasses", "std"),
            clean_auc_mean=("auc_fake_clean", "mean"),
            clean_auc_std=("auc_fake_clean", "std"),
            harmonic_acc_mean=("harmonic_accuracy", "mean"),
            harmonic_auc_mean=("harmonic_auc", "mean"),
            alpha_glasses_mean=("alpha_mean_glasses", "mean"),
            alpha_clean_mean=("alpha_mean_clean", "mean"),
            activation_gap_mean=("activation_gap", "mean"),
            glasses_activation_rate_mean=("activation_rate_glasses", "mean"),
            clean_activation_rate_mean=("activation_rate_clean", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(
        results_dir / "stability_plasticity_summary.csv",
        index=False,
    )

    # Baseline summary.
    baseline_payload = {
        "global_clean": global_clean,
        "global_glasses": global_glasses,
        "slow_clean": slow_clean,
        "slow_glasses": slow_glasses,
        "global_weights": global_logits.softmax(0).detach().cpu().tolist(),
        "expert_reliability": expert_rows,
    }

    (results_dir / "baseline_summary.json").write_text(
        json.dumps(baseline_payload, indent=2, default=safe_json)
    )
    (results_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, default=safe_json)
    )

    # Human-readable summary.
    lines = []
    lines.append("Experiment #6 — Real-Data Recoverability")
    lines.append("="*72)
    lines.append(f"Dataset: {args.dataset_id}")
    lines.append(
        f"Samples: train={len(train_subset)}, val={len(val_subset)}, "
        f"test={len(test_subset)}"
    )
    lines.append("")
    lines.append(
        f"GLOBAL: clean AUC={global_clean['auc_fake']:.4f}, "
        f"glasses AUC={global_glasses['auc_fake']:.4f}, "
        f"clean ACC={global_clean['accuracy']:.4f}, "
        f"glasses ACC={global_glasses['accuracy']:.4f}"
    )
    lines.append(
        f"SLOW-0: clean AUC={slow_clean['auc_fake']:.4f}, "
        f"glasses AUC={slow_glasses['auc_fake']:.4f}, "
        f"clean ACC={slow_clean['accuracy']:.4f}, "
        f"glasses ACC={slow_glasses['accuracy']:.4f}"
    )
    lines.append("")
    lines.append("Standalone expert reliability:")
    for r in expert_rows:
        lines.append(
            f"  {r['condition']:7s} {r['expert']:9s} "
            f"AUC={r['auc_fake']:.4f} ACC={r['accuracy']:.4f}"
        )
    lines.append("")
    lines.append("Best K per method by harmonic AUC:")
    for method in sorted(agg["method"].unique()):
        sub = agg[agg["method"] == method]
        best = sub.loc[sub["harmonic_auc_mean"].idxmax()]
        lines.append(
            f"  {method:16s} k={int(best['k_shot']):2d} "
            f"G_AUC={best['glasses_auc_mean']:.4f} "
            f"C_AUC={best['clean_auc_mean']:.4f} "
            f"H_AUC={best['harmonic_auc_mean']:.4f} "
            f"G_ACC={best['glasses_acc_mean']:.4f} "
            f"C_ACC={best['clean_acc_mean']:.4f}"
        )

    summary_text = "\n".join(lines)
    (results_dir / "summary.txt").write_text(summary_text + "\n")
    print("\n" + summary_text)

    # Bundle only results + manifest + preview, not large checkpoints/caches.
    zip_base = work / "results_exp6"
    zip_file = shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=work,
        base_dir="results",
    )

    # Add manifest + preview to final zip.
    temp_zip = work / "results_exp6_with_manifest.zip"
    import zipfile
    with zipfile.ZipFile(zip_file, "r") as src, zipfile.ZipFile(
        temp_zip, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        dst.write(work / "manifest.json", arcname="manifest.json")
        dst.write(
            work / "glasses_intervention_preview.jpg",
            arcname="glasses_intervention_preview.jpg",
        )

    Path(zip_file).unlink(missing_ok=True)
    temp_zip.rename(work / "results_exp6.zip")

    print(f"\nFINAL ARCHIVE: {work / 'results_exp6.zip'}")
    print("Send results_exp6.zip back to ChatGPT.")


if __name__ == "__main__":
    main()
