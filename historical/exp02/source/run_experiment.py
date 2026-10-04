
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF


# --------------------------- reproducibility/device ---------------------------

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


# ------------------------------ synthetic data -------------------------------

class SyntheticEvidenceDataset(Dataset):
    """
    Binary classification with 3 deliberately separated cues.

    Cue A (dominant/fragile): eye-band marker position.
    Cue B (weaker/stable): lower-face shape orientation.
    Cue C (weaker/stable): subtle global chroma balance.

    Corruptions are applied after image construction.
    """

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

    def _bernoulli_correct(self, rng: random.Random, p: float, y: int) -> int:
        return y if rng.random() < p else 1 - y

    def _base_image(self, idx: int) -> Tuple[Image.Image, int]:
        s = self.s
        y = idx % 2
        rng = random.Random(self.seed + idx * 7919)
        rng_np = np.random.default_rng(self.seed * 100003 + idx)

        arr = np.full((s, s, 3), 118.0, dtype=np.float32)
        arr += rng_np.normal(0, 10.0, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # Face-like oval for visual structure only.
        face_box = [int(.18*s), int(.08*s), int(.82*s), int(.92*s)]
        draw.ellipse(face_box, outline=(150, 150, 150), width=max(1, s//64))

        # Cue A: almost perfectly predictive eye-band marker.
        # Class 0 -> left bright marker; class 1 -> right bright marker.
        eye_y = int(0.34 * s)
        r = max(2, s // 20)
        ax = int((0.34 if y == 0 else 0.66) * s)
        draw.ellipse([ax-r, eye_y-r, ax+r, eye_y+r], fill=(246, 246, 246))

        # A distractor marker makes the task less trivial visually.
        dx = int((0.66 if y == 0 else 0.34) * s)
        draw.ellipse([dx-r, eye_y-r, dx+r, eye_y+r], fill=(72, 72, 72))

        # Cue B: lower-face geometry, only probabilistically aligned with class.
        by = self._bernoulli_correct(rng, self.cue_b_strength, y)
        cx, cy = int(.50*s), int(.70*s)
        span = int(.18*s)
        if by == 0:
            draw.line([cx-span, cy+span//2, cx, cy-span//2, cx+span, cy+span//2],
                      fill=(222, 222, 222), width=max(2, s//28))
        else:
            draw.line([cx-span, cy-span//2, cx, cy+span//2, cx+span, cy-span//2],
                      fill=(222, 222, 222), width=max(2, s//28))

        # Cue C: weak global chroma bias, again probabilistically aligned.
        cy_label = self._bernoulli_correct(rng, self.cue_c_strength, y)
        arr = np.asarray(img).astype(np.float32)
        delta = 8.0
        if cy_label == 0:
            arr[..., 0] += delta
            arr[..., 2] -= delta
        else:
            arr[..., 2] += delta
            arr[..., 0] -= delta
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        return img, y

    def _corrupt(self, img: Image.Image, idx: int) -> Image.Image:
        kind = self.corruption
        if kind == "clean":
            return img

        rng = random.Random(self.seed * 17 + idx)
        s = self.s
        out = img.copy()
        draw = ImageDraw.Draw(out)

        if kind == "eye_patch":
            # Seen during meta-training; rectangular eye-band corruption.
            draw.rectangle([int(.20*s), int(.25*s), int(.80*s), int(.45*s)],
                           fill=(rng.randint(80, 150),) * 3)
            return out

        if kind == "lower_patch":
            # Seen during meta-training; hurts B instead.
            draw.rectangle([int(.25*s), int(.56*s), int(.75*s), int(.86*s)],
                           fill=(rng.randint(80, 150),) * 3)
            return out

        if kind == "desaturate":
            # Seen during meta-training; hurts chroma C.
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
            # HELD OUT during meta-training.
            # Novel visual form, but it specifically attacks the same fragile
            # eye-band evidence A.
            lens_y0, lens_y1 = int(.27*s), int(.42*s)
            lens_w = int(.23*s)
            gap = int(.055*s)
            lx0 = s//2 - gap//2 - lens_w
            rx0 = s//2 + gap//2
            dark = (20, 20, 20)
            frame = (3, 3, 3)

            draw.rounded_rectangle([lx0, lens_y0, lx0+lens_w, lens_y1],
                                   radius=max(2, s//40), fill=dark, outline=frame, width=max(1, s//64))
            draw.rounded_rectangle([rx0, lens_y0, rx0+lens_w, lens_y1],
                                   radius=max(2, s//40), fill=dark, outline=frame, width=max(1, s//64))
            mid_y = (lens_y0 + lens_y1)//2
            draw.line([lx0+lens_w, mid_y, rx0, mid_y], fill=frame, width=max(1, s//48))
            draw.line([int(.10*s), mid_y, lx0, mid_y], fill=frame, width=max(1, s//64))
            draw.line([rx0+lens_w, mid_y, int(.90*s), mid_y], fill=frame, width=max(1, s//64))
            return out

        raise ValueError(kind)

    def __getitem__(self, idx: int):
        img, y = self._base_image(idx)
        img = self._corrupt(img, idx)
        x = TF.to_tensor(img.resize((self.s, self.s), Image.Resampling.BILINEAR))
        return x, y


# ---------------------------- isolated feature views --------------------------

def view_a_eye_band(x: torch.Tensor) -> torch.Tensor:
    # Keep only the eye region. Everything else is zeroed.
    out = torch.zeros_like(x)
    h = x.shape[-2]
    out[..., int(.20*h):int(.48*h), :] = x[..., int(.20*h):int(.48*h), :]
    return out


def view_b_lower_gray(x: torch.Tensor) -> torch.Tensor:
    # Lower face only, grayscale. Chroma is removed by design.
    g = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
    out = torch.zeros_like(g)
    h = x.shape[-2]
    out[..., int(.50*h):, :] = g[..., int(.50*h):, :]
    return out


def view_c_chroma(x: torch.Tensor) -> torch.Tensor:
    # Global opponent color; largely ignores luminance geometry.
    r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
    return torch.cat([r-g, b-g, r-b], dim=1)


FEATURES = {
    "A_eye": (view_a_eye_band, 3),
    "B_lower_geometry": (view_b_lower_gray, 1),
    "C_chroma": (view_c_chroma, 3),
}


# --------------------------------- models ------------------------------------

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
        z = self.net(x).flatten(1)
        return self.head(z)


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
        return torch.stack(outs, dim=1)  # [B,E,C]


class Gate(nn.Module):
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

    def forward(self, x, expert_logits):
        z = self.encoder(x).flatten(1)
        p = expert_logits.softmax(-1)
        conf = p.max(-1).values
        margin = (p[...,1] - p[...,0]).abs()
        entropy = -(p * p.clamp_min(1e-8).log()).sum(-1)
        return self.mlp(torch.cat([z, conf, margin, entropy], dim=1))


def fuse(expert_logits, gate_logits):
    w = gate_logits.softmax(1)
    return (expert_logits * w.unsqueeze(-1)).sum(1), w


# ------------------------------ training utils -------------------------------

def loader(ds, batch, shuffle=False):
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
    return {name: float(correct[i].item()/max(1,n)) for i,name in enumerate(bank.names)}


def train_experts(bank, train_dl, val_dl, device, epochs, lr):
    opt = torch.optim.AdamW(bank.parameters(), lr=lr, weight_decay=1e-4)
    best, best_score = None, -1
    for epoch in range(1, epochs+1):
        bank.train()
        total_loss = total = 0
        for x, y in train_dl:
            x,y = x.to(device), y.to(device)
            ex = bank.forward_all(x)
            loss = sum(F.cross_entropy(ex[:,i], y) for i in range(ex.shape[1])) / ex.shape[1]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()*y.numel()
            total += y.numel()

        accs = standalone_accuracies(bank, val_dl, device)
        mean_acc = sum(accs.values())/len(accs)
        print(f"[experts] epoch={epoch:02d} loss={total_loss/max(1,total):.4f} val={accs}")
        if mean_acc > best_score:
            best_score = mean_acc
            best = {k:v.detach().cpu().clone() for k,v in bank.state_dict().items()}
    if best:
        bank.load_state_dict(best)


def train_global(bank, train_dl, device, epochs, lr):
    for p in bank.parameters():
        p.requires_grad_(False)
    logits = nn.Parameter(torch.zeros(len(bank.names), device=device))
    opt = torch.optim.Adam([logits], lr=lr)
    for epoch in range(1, epochs+1):
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                ex = bank.forward_all(x)
            out,_ = fuse(ex, logits.unsqueeze(0).expand(x.shape[0],-1))
            loss = F.cross_entropy(out,y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"[global] epoch={epoch:02d} weights={logits.softmax(0).detach().cpu().numpy().round(3).tolist()}")
    return logits.detach().clone()


def balanced_indices(ds, k, seed):
    by = {0:[], 1:[]}
    for i in range(len(ds)):
        _, y = ds[i]
        by[int(y)].append(i)
    rng = random.Random(seed)
    out = []
    each = max(1, k//2)
    for c in (0,1):
        rng.shuffle(by[c])
        out += by[c][:each]
    return out[:k]


def meta_train_gate(bank, gate, args, device, seed):
    """
    First-order MAML-like training of the router.

    Meta-training perturbations exclude 'glasses'.
    Expert dropout and a weak entropy floor discourage collapse.
    """
    meta_corruptions = ["clean", "eye_patch", "lower_patch", "desaturate", "blur", "jpeg"]
    rng = random.Random(seed + 5000)
    outer = torch.optim.AdamW(gate.parameters(), lr=args.meta_lr, weight_decay=1e-4)

    for p in bank.parameters():
        p.requires_grad_(False)

    for ep in range(1, args.meta_episodes+1):
        corr = rng.choice(meta_corruptions)
        ds = SyntheticEvidenceDataset(
            args.train_n, args.image_size, seed+100, corr,
            args.cue_b_strength, args.cue_c_strength
        )
        ids = list(range(len(ds)))
        rng.shuffle(ids)
        sids = ids[:args.support_size]
        qids = ids[args.support_size:args.support_size+args.query_size]

        fast = copy.deepcopy(gate).to(device)
        inner = torch.optim.SGD(fast.parameters(), lr=args.inner_lr)

        for _ in range(args.inner_steps):
            for x,y in loader(Subset(ds,sids), args.support_size, True):
                x,y = x.to(device), y.to(device)
                with torch.no_grad():
                    ex = bank.forward_all(x)
                gl = fast(x,ex)

                # Expert dropout: randomly make one route unattractive during
                # training so the gate cannot let alternatives die.
                if rng.random() < args.expert_dropout:
                    j = rng.randrange(ex.shape[1])
                    gl = gl.clone()
                    gl[:,j] = gl[:,j] - 6.0

                out,w = fuse(ex,gl)
                ce = F.cross_entropy(out,y)

                # Weak entropy floor: penalize only when entropy is too low.
                ent = -(w*w.clamp_min(1e-8).log()).sum(1).mean()
                target_ent = args.entropy_floor * math.log(ex.shape[1])
                ent_pen = F.relu(torch.tensor(target_ent, device=device)-ent)
                loss = ce + args.entropy_lambda*ent_pen

                inner.zero_grad(set_to_none=True)
                loss.backward()
                inner.step()

        xq,yq = next(iter(loader(Subset(ds,qids), args.query_size, False)))
        xq,yq = xq.to(device), yq.to(device)
        with torch.no_grad():
            exq = bank.forward_all(xq)
        gout = fast(xq,exq)
        out,w = fuse(exq,gout)
        qloss = F.cross_entropy(out,yq)

        fast.zero_grad(set_to_none=True)
        qloss.backward()
        outer.zero_grad(set_to_none=True)
        for p,fp in zip(gate.parameters(), fast.parameters()):
            if fp.grad is not None:
                p.grad = fp.grad.detach().clone()
        outer.step()

        if ep == 1 or ep % args.log_every == 0:
            print(f"[meta] {ep:04d}/{args.meta_episodes} corr={corr:12s} qloss={qloss.item():.4f}")


@torch.no_grad()
def evaluate(bank, gate, global_logits, dl, device, mode):
    bank.eval()
    if gate is not None:
        gate.eval()
    n=correct=0
    loss_sum=0.0
    wsum = torch.zeros(len(bank.names), device=device)

    for x,y in dl:
        x,y = x.to(device), y.to(device)
        ex = bank.forward_all(x)
        if mode == "uniform":
            gl = torch.zeros(x.shape[0], len(bank.names), device=device)
        elif mode == "global":
            gl = global_logits.unsqueeze(0).expand(x.shape[0],-1)
        elif mode == "adaptive":
            gl = gate(x,ex)
        else:
            raise ValueError(mode)
        out,w = fuse(ex,gl)
        loss_sum += F.cross_entropy(out,y,reduction="sum").item()
        correct += (out.argmax(1)==y).sum().item()
        n += y.numel()
        wsum += w.sum(0)

    return {
        "accuracy": correct/max(1,n),
        "loss": loss_sum/max(1,n),
        "mean_weights": (wsum/max(1,n)).detach().cpu().tolist(),
        "n": n,
    }


def adapt_gate(bank, gate, support_dl, device, steps, lr):
    g = copy.deepcopy(gate).to(device)
    opt = torch.optim.SGD(g.parameters(), lr=lr)
    cached = [(x.to(device),y.to(device)) for x,y in support_dl]
    for _ in range(steps):
        for x,y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)
            out,_ = fuse(ex,g(x,ex))
            loss = F.cross_entropy(out,y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return g


def oracle_fusion(bank, support_dl, device, steps, lr):
    logits = nn.Parameter(torch.zeros(len(bank.names), device=device))
    opt = torch.optim.Adam([logits], lr=lr)
    cached = [(x.to(device),y.to(device)) for x,y in support_dl]
    for _ in range(steps):
        for x,y in cached:
            with torch.no_grad():
                ex = bank.forward_all(x)
            out,_ = fuse(ex,logits.unsqueeze(0).expand(x.shape[0],-1))
            loss = F.cross_entropy(out,y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return logits.detach()


@dataclass
class Row:
    seed: int
    corruption: str
    method: str
    k_shot: int
    accuracy: float
    loss: float
    w_A_eye: float
    w_B_lower_geometry: float
    w_C_chroma: float
    A_eye_acc: float
    B_lower_geometry_acc: float
    C_chroma_acc: float


def make_row(seed, corr, method, k, metrics, expert_acc):
    w = metrics["mean_weights"]
    return Row(
        seed, corr, method, k,
        metrics["accuracy"], metrics["loss"],
        w[0], w[1], w[2],
        expert_acc["A_eye"], expert_acc["B_lower_geometry"], expert_acc["C_chroma"],
    )


def run_seed(args, seed, out_dir):
    print("\n" + "="*80)
    print(f"SEED {seed}")
    print("="*80)
    seed_everything(seed)
    device = pick_device(args.device)
    print(f"device={device}; torch={torch.__version__}; mps_available={torch.backends.mps.is_available()}")

    train = SyntheticEvidenceDataset(args.train_n,args.image_size,seed+10,"clean",args.cue_b_strength,args.cue_c_strength)
    val = SyntheticEvidenceDataset(args.val_n,args.image_size,seed+20,"clean",args.cue_b_strength,args.cue_c_strength)
    bank = ExpertBank().to(device)

    print("\nStage 1/4 — train isolated experts on clean data")
    train_experts(bank, loader(train,args.batch_size,True), loader(val,args.batch_size), device,args.expert_epochs,args.expert_lr)

    print("\nStage 2/4 — learn fixed global fusion")
    global_logits = train_global(bank,loader(train,args.batch_size,True),device,args.global_epochs,args.global_lr)

    gate = Gate(len(bank.names)).to(device)
    with torch.no_grad():
        gate.mlp[-1].bias.copy_(global_logits)

    print("\nStage 3/4 — meta-train adaptive gate (NO glasses)")
    meta_train_gate(bank,gate,args,device,seed)

    print("\nStage 4/4 — held-out glasses evaluation")
    rows = []

    clean = SyntheticEvidenceDataset(args.test_n,args.image_size,seed+30,"clean",args.cue_b_strength,args.cue_c_strength)
    glasses = SyntheticEvidenceDataset(args.test_n,args.image_size,seed+30,"glasses",args.cue_b_strength,args.cue_c_strength)

    for corr,ds in [("clean",clean),("glasses",glasses)]:
        dl = loader(ds,args.batch_size)
        eacc = standalone_accuracies(bank,dl,device)
        print(f"[{corr}] standalone experts: {eacc}")

        for method,mode in [("uniform","uniform"),("global","global"),("adaptive_zero_shot","adaptive")]:
            m = evaluate(bank,gate,global_logits,dl,device,mode)
            rows.append(make_row(seed,corr,method,0,m,eacc))
            print(f"[{corr}] {method:18s} acc={m['accuracy']:.4f} weights={np.round(m['mean_weights'],3).tolist()}")

        if corr == "glasses":
            for k in args.k_shots:
                sids = balanced_indices(ds,k,seed+k)
                qids = [i for i in range(len(ds)) if i not in set(sids)]
                sdl = loader(Subset(ds,sids), min(k,args.batch_size), True)
                qdl = loader(Subset(ds,qids),args.batch_size)
                qeacc = standalone_accuracies(bank,qdl,device)

                ag = adapt_gate(bank,gate,sdl,device,args.test_adapt_steps,args.test_adapt_lr)
                m = evaluate(bank,ag,global_logits,qdl,device,"adaptive")
                rows.append(make_row(seed,corr,"adaptive_few_shot",k,m,qeacc))
                print(f"[glasses] adaptive k={k:02d} acc={m['accuracy']:.4f} weights={np.round(m['mean_weights'],3).tolist()}")

                ogl = oracle_fusion(bank,sdl,device,args.oracle_steps,args.oracle_lr)
                # Evaluate oracle using global mode with learned condition logits.
                m = evaluate(bank,gate,ogl,qdl,device,"global")
                rows.append(make_row(seed,corr,"oracle_fusion",k,m,qeacc))
                print(f"[glasses] oracle   k={k:02d} acc={m['accuracy']:.4f} weights={np.round(m['mean_weights'],3).tolist()}")

    torch.save({
        "seed": seed,
        "bank": bank.state_dict(),
        "gate": gate.state_dict(),
        "global_logits": global_logits.detach().cpu(),
        "config": vars(args),
        "expert_names": bank.names,
    }, out_dir/f"checkpoint_seed_{seed}.pt")

    return rows


def summarize(rows: List[Row]) -> str:
    lines = []
    seeds = sorted(set(r.seed for r in rows))
    lines.append(f"seeds={seeds}")
    lines.append("")

    methods = ["uniform","global","adaptive_zero_shot","adaptive_few_shot","oracle_fusion"]
    for corr in ["clean","glasses"]:
        lines.append(corr.upper())
        for method in methods:
            subset = [r for r in rows if r.corruption==corr and r.method==method]
            if not subset:
                continue
            by_k = sorted(set(r.k_shot for r in subset))
            for k in by_k:
                ss = [r for r in subset if r.k_shot==k]
                accs = [r.accuracy for r in ss]
                ws = np.array([[r.w_A_eye,r.w_B_lower_geometry,r.w_C_chroma] for r in ss],dtype=float)
                lines.append(
                    f"{method:18s} k={k:2d} "
                    f"acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
                    f"weights={np.mean(ws,axis=0).round(3).tolist()}"
                )
        lines.append("")

    # Core diagnostics.
    g = [r for r in rows if r.corruption=="glasses" and r.method=="global"]
    fs = [r for r in rows if r.corruption=="glasses" and r.method=="adaptive_few_shot"]
    if g and fs:
        gmean = float(np.mean([r.accuracy for r in g]))
        best_k = None
        best_mean = -1
        for k in sorted(set(r.k_shot for r in fs)):
            m = float(np.mean([r.accuracy for r in fs if r.k_shot==k]))
            if m > best_mean:
                best_mean, best_k = m, k
        lines.append(f"CORE: glasses global mean accuracy = {gmean:.4f}")
        lines.append(f"CORE: best adaptive few-shot mean accuracy = {best_mean:.4f} at k={best_k}")
        lines.append(f"CORE: recovery over global = {best_mean-gmean:+.4f}")

    return "\n".join(lines)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=["auto","mps","cpu","cuda"],default="auto")
    p.add_argument("--output",default="results")
    p.add_argument("--seeds",type=int,nargs="+",default=[42])
    p.add_argument("--image-size",type=int,default=64)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--train-n",type=int,default=2600)
    p.add_argument("--val-n",type=int,default=700)
    p.add_argument("--test-n",type=int,default=1000)

    p.add_argument("--cue-b-strength",type=float,default=0.72)
    p.add_argument("--cue-c-strength",type=float,default=0.66)

    p.add_argument("--expert-epochs",type=int,default=6)
    p.add_argument("--expert-lr",type=float,default=0.002)
    p.add_argument("--global-epochs",type=int,default=4)
    p.add_argument("--global-lr",type=float,default=0.08)

    p.add_argument("--meta-episodes",type=int,default=300)
    p.add_argument("--support-size",type=int,default=20)
    p.add_argument("--query-size",type=int,default=40)
    p.add_argument("--inner-steps",type=int,default=1)
    p.add_argument("--inner-lr",type=float,default=0.06)
    p.add_argument("--meta-lr",type=float,default=0.001)
    p.add_argument("--expert-dropout",type=float,default=0.35)
    p.add_argument("--entropy-floor",type=float,default=0.42)
    p.add_argument("--entropy-lambda",type=float,default=0.06)
    p.add_argument("--log-every",type=int,default=25)

    p.add_argument("--k-shots",type=int,nargs="+",default=[2,4,8,16,32])
    p.add_argument("--test-adapt-steps",type=int,default=12)
    p.add_argument("--test-adapt-lr",type=float,default=0.03)
    p.add_argument("--oracle-steps",type=int,default=80)
    p.add_argument("--oracle-lr",type=float,default=0.08)
    return p.parse_args()


def main():
    args=parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)

    all_rows=[]
    for s in args.seeds:
        all_rows.extend(run_seed(args,s,out))

    with open(out/"results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(all_rows[0]).keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))

    payload={
        "config":vars(args),
        "rows":[asdict(r) for r in all_rows],
        "summary":summarize(all_rows),
    }
    (out/"results.json").write_text(json.dumps(payload,indent=2))
    (out/"summary.txt").write_text(payload["summary"]+"\n")

    print("\n"+"="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(payload["summary"])
    print(f"\nSaved to {out.resolve()}")


if __name__=="__main__":
    main()
