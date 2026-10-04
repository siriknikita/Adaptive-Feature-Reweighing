
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


# ---------------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------

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
    def _maybe_flip(rng: random.Random, p_correct: float, y: int) -> int:
        return y if rng.random() < p_correct else 1 - y

    def _base(self, idx: int) -> Tuple[Image.Image, int]:
        s = self.s
        y = idx % 2
        rng = random.Random(self.seed + idx * 7919)
        rng_np = np.random.default_rng(self.seed * 100003 + idx)

        arr = np.full((s, s, 3), 118.0, dtype=np.float32)
        arr += rng_np.normal(0, 10.0, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # weak face scaffold
        draw.ellipse(
            [int(.18*s), int(.08*s), int(.82*s), int(.92*s)],
            outline=(150, 150, 150),
            width=max(1, s//64)
        )

        # A: dominant fragile eye cue
        eye_y = int(.34*s)
        r = max(2, s//20)
        ax = int((.34 if y == 0 else .66) * s)
        dx = int((.66 if y == 0 else .34) * s)
        draw.ellipse([ax-r, eye_y-r, ax+r, eye_y+r], fill=(246,246,246))
        draw.ellipse([dx-r, eye_y-r, dx+r, eye_y+r], fill=(72,72,72))

        # B: weaker stable lower geometry
        by = self._maybe_flip(rng, self.cue_b_strength, y)
        cx, cy = int(.50*s), int(.70*s)
        span = int(.18*s)
        if by == 0:
            draw.line(
                [cx-span, cy+span//2, cx, cy-span//2, cx+span, cy+span//2],
                fill=(222,222,222), width=max(2,s//28)
            )
        else:
            draw.line(
                [cx-span, cy-span//2, cx, cy+span//2, cx+span, cy-span//2],
                fill=(222,222,222), width=max(2,s//28)
            )

        # C: weaker stable chroma
        cy_label = self._maybe_flip(rng, self.cue_c_strength, y)
        arr = np.asarray(img).astype(np.float32)
        delta = 8.0
        if cy_label == 0:
            arr[...,0] += delta
            arr[...,2] -= delta
        else:
            arr[...,2] += delta
            arr[...,0] -= delta
        return Image.fromarray(np.clip(arr,0,255).astype(np.uint8)), y

    def _corrupt(self, img: Image.Image, idx: int) -> Image.Image:
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
                fill=(rng.randint(80,150),)*3
            )
            return out

        if kind == "lower_patch":
            draw.rectangle(
                [int(.25*s), int(.56*s), int(.75*s), int(.86*s)],
                fill=(rng.randint(80,150),)*3
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
            # HELD OUT FROM META-TRAINING.
            lens_y0, lens_y1 = int(.27*s), int(.42*s)
            lens_w = int(.23*s)
            gap = int(.055*s)
            lx0 = s//2 - gap//2 - lens_w
            rx0 = s//2 + gap//2
            dark = (20,20,20)
            frame = (3,3,3)

            draw.rounded_rectangle(
                [lx0,lens_y0,lx0+lens_w,lens_y1],
                radius=max(2,s//40),
                fill=dark, outline=frame, width=max(1,s//64)
            )
            draw.rounded_rectangle(
                [rx0,lens_y0,rx0+lens_w,lens_y1],
                radius=max(2,s//40),
                fill=dark, outline=frame, width=max(1,s//64)
            )
            mid = (lens_y0+lens_y1)//2
            draw.line([lx0+lens_w,mid,rx0,mid], fill=frame, width=max(1,s//48))
            draw.line([int(.10*s),mid,lx0,mid], fill=frame, width=max(1,s//64))
            draw.line([rx0+lens_w,mid,int(.90*s),mid], fill=frame, width=max(1,s//64))
            return out

        raise ValueError(f"Unknown corruption: {kind}")

    def __getitem__(self, idx):
        img, y = self._base(idx)
        img = self._corrupt(img, idx)
        x = TF.to_tensor(img.resize((self.s,self.s), Image.Resampling.BILINEAR))
        return x, y


# ---------------------------------------------------------------------------
# Feature-isolated experts
# ---------------------------------------------------------------------------

def view_a(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    h = x.shape[-2]
    out[..., int(.20*h):int(.48*h), :] = x[..., int(.20*h):int(.48*h), :]
    return out


def view_b(x: torch.Tensor) -> torch.Tensor:
    g = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
    out = torch.zeros_like(g)
    h = x.shape[-2]
    out[..., int(.50*h):, :] = g[..., int(.50*h):, :]
    return out


def view_c(x: torch.Tensor) -> torch.Tensor:
    r,g,b = x[:,0:1],x[:,1:2],x[:,2:3]
    return torch.cat([r-g,b-g,r-b], dim=1)


FEATURES = {
    "A_eye": (view_a,3),
    "B_lower_geometry": (view_b,1),
    "C_chroma": (view_c,3),
}


class TinyExpert(nn.Module):
    def __init__(self, in_ch: int, width: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,width,5,stride=2,padding=2),
            nn.GELU(),
            nn.Conv2d(width,width*2,3,stride=2,padding=1),
            nn.GELU(),
            nn.Conv2d(width*2,width*3,3,stride=2,padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width*3,2)

    def forward(self,x):
        return self.head(self.net(x).flatten(1))


class ExpertBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.names = list(FEATURES)
        self.experts = nn.ModuleDict({
            name: TinyExpert(FEATURES[name][1]) for name in self.names
        })

    def forward_all(self,x):
        outs = []
        for name in self.names:
            fn,_ = FEATURES[name]
            outs.append(self.experts[name](fn(x)))
        return torch.stack(outs,dim=1)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Gate(nn.Module):
    def __init__(self,n_experts:int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,12,5,stride=2,padding=2),
            nn.GELU(),
            nn.Conv2d(12,20,3,stride=2,padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(20+3*n_experts,48),
            nn.GELU(),
            nn.Linear(48,n_experts),
        )

    def forward(self,x,expert_logits):
        z = self.encoder(x).flatten(1)
        p = expert_logits.softmax(-1)
        conf = p.max(-1).values
        margin = (p[...,1]-p[...,0]).abs()
        entropy = -(p*p.clamp_min(1e-8).log()).sum(-1)
        return self.mlp(torch.cat([z,conf,margin,entropy],dim=1))


# ---------------------------------------------------------------------------
# Routing mechanisms
# ---------------------------------------------------------------------------

METHODS = [
    "softmax",
    "temperature_softmax",
    "epsilon_floor",
    "logit_clip",
    "residual_router",
]


def route_weights(
    method: str,
    gate_logits: torch.Tensor,
    global_logits: torch.Tensor,
    args,
) -> torch.Tensor:
    if method == "softmax":
        return gate_logits.softmax(1)

    if method == "temperature_softmax":
        return (gate_logits / args.temperature).softmax(1)

    if method == "epsilon_floor":
        base = gate_logits.softmax(1)
        e = base.shape[1]
        eps = args.epsilon_floor
        if eps * e >= 1.0:
            raise ValueError("epsilon_floor must satisfy epsilon * n_experts < 1")
        return (1.0 - e*eps)*base + eps

    if method == "logit_clip":
        clipped = gate_logits.clamp(-args.logit_clip, args.logit_clip)
        return clipped.softmax(1)

    if method == "residual_router":
        # bounded residual around global logits
        residual = torch.tanh(gate_logits) * args.residual_scale
        gl = global_logits.unsqueeze(0).expand_as(gate_logits)
        return (gl + residual).softmax(1)

    raise ValueError(method)


def fuse_with_weights(expert_logits, weights):
    return (expert_logits * weights.unsqueeze(-1)).sum(1)


def entropy_of_weights(w: torch.Tensor) -> torch.Tensor:
    return -(w * w.clamp_min(1e-8).log()).sum(1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_loader(ds,batch,shuffle=False):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0)


@torch.no_grad()
def standalone_accuracies(bank,dl,device):
    bank.eval()
    correct = torch.zeros(len(bank.names),dtype=torch.long)
    n=0
    for x,y in dl:
        x,y=x.to(device),y.to(device)
        ex=bank.forward_all(x)
        pred=ex.argmax(-1)
        correct += (pred.cpu()==y.cpu().unsqueeze(1)).sum(0)
        n += y.numel()
    return {name:correct[i].item()/max(1,n) for i,name in enumerate(bank.names)}


def train_experts(bank,train_dl,val_dl,device,epochs,lr):
    opt=torch.optim.AdamW(bank.parameters(),lr=lr,weight_decay=1e-4)
    best=None
    best_score=-1
    for epoch in range(1,epochs+1):
        bank.train()
        total_loss=total=0
        for x,y in train_dl:
            x,y=x.to(device),y.to(device)
            ex=bank.forward_all(x)
            loss=sum(F.cross_entropy(ex[:,i],y) for i in range(ex.shape[1]))/ex.shape[1]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()*y.numel()
            total += y.numel()
        accs=standalone_accuracies(bank,val_dl,device)
        score=sum(accs.values())/len(accs)
        print(f"[experts] epoch={epoch:02d} loss={total_loss/max(1,total):.4f} val={accs}")
        if score > best_score:
            best_score=score
            best={k:v.detach().cpu().clone() for k,v in bank.state_dict().items()}
    if best is not None:
        bank.load_state_dict(best)


def train_global(bank,train_dl,device,epochs,lr):
    for p in bank.parameters():
        p.requires_grad_(False)
    logits=nn.Parameter(torch.zeros(len(bank.names),device=device))
    opt=torch.optim.Adam([logits],lr=lr)
    for epoch in range(1,epochs+1):
        for x,y in train_dl:
            x,y=x.to(device),y.to(device)
            with torch.no_grad():
                ex=bank.forward_all(x)
            w=logits.softmax(0).unsqueeze(0).expand(x.shape[0],-1)
            out=fuse_with_weights(ex,w)
            loss=F.cross_entropy(out,y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"[global] epoch={epoch:02d} weights={logits.softmax(0).detach().cpu().numpy().round(3).tolist()}")
    return logits.detach().clone()


def balanced_indices(ds,k,seed):
    by={0:[],1:[]}
    for i in range(len(ds)):
        _,y=ds[i]
        by[int(y)].append(i)
    rng=random.Random(seed)
    each=max(1,k//2)
    out=[]
    for c in (0,1):
        rng.shuffle(by[c])
        out += by[c][:each]
    return out[:k]


def meta_train_one_method(bank,base_gate,args,device,seed,method,global_logits):
    meta_corrs=["clean","eye_patch","lower_patch","desaturate","blur","jpeg"]
    rng=random.Random(seed+5000+hash(method)%10000)
    gate=copy.deepcopy(base_gate).to(device)
    outer=torch.optim.AdamW(gate.parameters(),lr=args.meta_lr,weight_decay=1e-4)

    for p in bank.parameters():
        p.requires_grad_(False)

    for ep in range(1,args.meta_episodes+1):
        corr=rng.choice(meta_corrs)
        ds=SyntheticEvidenceDataset(
            args.train_n,args.image_size,seed+100,corr,
            args.cue_b_strength,args.cue_c_strength
        )
        ids=list(range(len(ds)))
        rng.shuffle(ids)
        sids=ids[:args.support_size]
        qids=ids[args.support_size:args.support_size+args.query_size]

        fast=copy.deepcopy(gate).to(device)
        inner=torch.optim.SGD(fast.parameters(),lr=args.inner_lr)

        x,y=next(iter(make_loader(Subset(ds,sids),args.support_size,True)))
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            ex=bank.forward_all(x)
        gl=fast(x,ex)

        if rng.random() < args.expert_dropout:
            j=rng.randrange(ex.shape[1])
            gl=gl.clone()
            gl[:,j]-=6.0

        w=route_weights(method,gl,global_logits,args)
        out=fuse_with_weights(ex,w)
        ce=F.cross_entropy(out,y)

        ent=entropy_of_weights(w).mean()
        target=args.entropy_floor*math.log(ex.shape[1])
        ent_pen=F.relu(torch.tensor(target,device=device)-ent)
        loss=ce + args.entropy_lambda*ent_pen

        inner.zero_grad(set_to_none=True)
        loss.backward()
        inner.step()

        xq,yq=next(iter(make_loader(Subset(ds,qids),args.query_size,False)))
        xq,yq=xq.to(device),yq.to(device)
        with torch.no_grad():
            exq=bank.forward_all(xq)
        qgl=fast(xq,exq)
        qw=route_weights(method,qgl,global_logits,args)
        qout=fuse_with_weights(exq,qw)
        qloss=F.cross_entropy(qout,yq)

        fast.zero_grad(set_to_none=True)
        qloss.backward()
        outer.zero_grad(set_to_none=True)
        for p,fp in zip(gate.parameters(),fast.parameters()):
            if fp.grad is not None:
                p.grad=fp.grad.detach().clone()
        outer.step()

        if ep==1 or ep%args.log_every==0:
            print(f"[meta:{method}] {ep:04d}/{args.meta_episodes} corr={corr:12s} qloss={qloss.item():.4f}")

    return gate


@torch.no_grad()
def evaluate(bank,gate,global_logits,dl,device,args,mode,method=None):
    bank.eval()
    if gate is not None:
        gate.eval()

    n=correct=0
    loss_sum=0.0
    wsum=torch.zeros(len(bank.names),device=device)
    entropy_sum=0.0
    saturated=0

    for x,y in dl:
        x,y=x.to(device),y.to(device)
        ex=bank.forward_all(x)

        if mode=="uniform":
            w=torch.full(
                (x.shape[0],len(bank.names)),
                1.0/len(bank.names),
                device=device
            )
        elif mode=="global":
            w=global_logits.softmax(0).unsqueeze(0).expand(x.shape[0],-1)
        elif mode=="adaptive":
            gl=gate(x,ex)
            w=route_weights(method,gl,global_logits,args)
        else:
            raise ValueError(mode)

        out=fuse_with_weights(ex,w)
        loss_sum += F.cross_entropy(out,y,reduction="sum").item()
        correct += (out.argmax(1)==y).sum().item()
        n += y.numel()
        wsum += w.sum(0)
        entropy_sum += entropy_of_weights(w).sum().item()
        saturated += (w.max(1).values > args.saturation_threshold).sum().item()

    return {
        "accuracy":correct/max(1,n),
        "loss":loss_sum/max(1,n),
        "mean_weights":(wsum/max(1,n)).detach().cpu().tolist(),
        "mean_entropy":entropy_sum/max(1,n),
        "saturation_rate":saturated/max(1,n),
        "n":n,
    }


def adapt_gate(bank,gate,global_logits,support_dl,device,args,method):
    g=copy.deepcopy(gate).to(device)
    opt=torch.optim.SGD(g.parameters(),lr=args.test_adapt_lr)
    cached=[(x.to(device),y.to(device)) for x,y in support_dl]

    for _ in range(args.test_adapt_steps):
        for x,y in cached:
            with torch.no_grad():
                ex=bank.forward_all(x)
            gl=g(x,ex)
            w=route_weights(method,gl,global_logits,args)
            out=fuse_with_weights(ex,w)
            loss=F.cross_entropy(out,y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return g


def oracle_fusion(bank,support_dl,device,args):
    logits=nn.Parameter(torch.zeros(len(bank.names),device=device))
    opt=torch.optim.Adam([logits],lr=args.oracle_lr)
    cached=[(x.to(device),y.to(device)) for x,y in support_dl]

    for _ in range(args.oracle_steps):
        for x,y in cached:
            with torch.no_grad():
                ex=bank.forward_all(x)
            w=logits.softmax(0).unsqueeze(0).expand(x.shape[0],-1)
            out=fuse_with_weights(ex,w)
            loss=F.cross_entropy(out,y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return logits.detach()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class Row:
    seed:int
    method:str
    corruption:str
    k_shot:int
    accuracy:float
    loss:float
    mean_entropy:float
    saturation_rate:float
    w_A_eye:float
    w_B_lower_geometry:float
    w_C_chroma:float
    A_eye_acc:float
    B_lower_geometry_acc:float
    C_chroma_acc:float
    release_A:float
    recovery_BC:float


def make_row(seed,method,corr,k,m,eacc,zero_weights=None):
    w=m["mean_weights"]
    if zero_weights is None:
        release=0.0
        recover=0.0
    else:
        release=zero_weights[0]-w[0]
        recover=(w[1]+w[2])-(zero_weights[1]+zero_weights[2])

    return Row(
        seed=seed,
        method=method,
        corruption=corr,
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
        release_A=release,
        recovery_BC=recover,
    )


def evaluate_oracle(bank,oracle_logits,dl,device,args):
    return evaluate(bank,None,oracle_logits,dl,device,args,mode="global")


def run_seed(args,seed,out_dir):
    print("\n"+"="*88)
    print(f"SEED {seed}")
    print("="*88)

    seed_everything(seed)
    device=pick_device(args.device)
    print(f"device={device}; torch={torch.__version__}; mps_available={torch.backends.mps.is_available()}")

    train=SyntheticEvidenceDataset(args.train_n,args.image_size,seed+10,"clean",args.cue_b_strength,args.cue_c_strength)
    val=SyntheticEvidenceDataset(args.val_n,args.image_size,seed+20,"clean",args.cue_b_strength,args.cue_c_strength)
    bank=ExpertBank().to(device)

    print("\nStage 1 — train experts")
    train_experts(
        bank,
        make_loader(train,args.batch_size,True),
        make_loader(val,args.batch_size,False),
        device,args.expert_epochs,args.expert_lr
    )

    print("\nStage 2 — learn global fusion")
    global_logits=train_global(
        bank,
        make_loader(train,args.batch_size,True),
        device,args.global_epochs,args.global_lr
    )

    clean=SyntheticEvidenceDataset(args.test_n,args.image_size,seed+30,"clean",args.cue_b_strength,args.cue_c_strength)
    glasses=SyntheticEvidenceDataset(args.test_n,args.image_size,seed+30,"glasses",args.cue_b_strength,args.cue_c_strength)

    clean_dl=make_loader(clean,args.batch_size)
    glasses_dl=make_loader(glasses,args.batch_size)

    clean_eacc=standalone_accuracies(bank,clean_dl,device)
    glasses_eacc=standalone_accuracies(bank,glasses_dl,device)

    rows=[]

    # diagnostics: uniform/global
    for corr,dl,eacc in [("clean",clean_dl,clean_eacc),("glasses",glasses_dl,glasses_eacc)]:
        um=evaluate(bank,None,global_logits,dl,device,args,"uniform")
        rows.append(make_row(seed,"uniform",corr,0,um,eacc))
        gm=evaluate(bank,None,global_logits,dl,device,args,"global")
        rows.append(make_row(seed,"global",corr,0,gm,eacc))

    print("\nStage 3 — train router ablations")
    method_gates={}
    for method in METHODS:
        base=Gate(len(bank.names)).to(device)
        with torch.no_grad():
            base.mlp[-1].bias.copy_(global_logits)
        gate=meta_train_one_method(bank,base,args,device,seed,method,global_logits)
        method_gates[method]=gate

    print("\nStage 4 — held-out glasses recoverability")
    for method,gate in method_gates.items():
        clean_m=evaluate(bank,gate,global_logits,clean_dl,device,args,"adaptive",method)
        rows.append(make_row(seed,method,"clean",0,clean_m,clean_eacc))

        zero=evaluate(bank,gate,global_logits,glasses_dl,device,args,"adaptive",method)
        rows.append(make_row(seed,method,"glasses",0,zero,glasses_eacc))
        zero_w=zero["mean_weights"]

        print(
            f"[{method}] glasses zero-shot acc={zero['accuracy']:.4f} "
            f"sat={zero['saturation_rate']:.3f} "
            f"weights={np.round(zero_w,3).tolist()}"
        )

        for k in args.k_shots:
            sids=balanced_indices(glasses,k,seed+k)
            qids=[i for i in range(len(glasses)) if i not in set(sids)]
            sdl=make_loader(Subset(glasses,sids),min(k,args.batch_size),True)
            qdl=make_loader(Subset(glasses,qids),args.batch_size,False)
            qeacc=standalone_accuracies(bank,qdl,device)

            adapted=adapt_gate(bank,gate,global_logits,sdl,device,args,method)
            m=evaluate(bank,adapted,global_logits,qdl,device,args,"adaptive",method)
            rows.append(make_row(seed,method,"glasses",k,m,qeacc,zero_w))

            print(
                f"[{method}] k={k:02d} acc={m['accuracy']:.4f} "
                f"sat={m['saturation_rate']:.3f} "
                f"releaseA={zero_w[0]-m['mean_weights'][0]:+.3f} "
                f"weights={np.round(m['mean_weights'],3).tolist()}"
            )

    # oracle
    for k in args.k_shots:
        sids=balanced_indices(glasses,k,seed+k)
        qids=[i for i in range(len(glasses)) if i not in set(sids)]
        sdl=make_loader(Subset(glasses,sids),min(k,args.batch_size),True)
        qdl=make_loader(Subset(glasses,qids),args.batch_size,False)
        qeacc=standalone_accuracies(bank,qdl,device)
        ol=oracle_fusion(bank,sdl,device,args)
        m=evaluate_oracle(bank,ol,qdl,device,args)
        rows.append(make_row(seed,"oracle","glasses",k,m,qeacc))

    torch.save(
        {
            "seed":seed,
            "bank":bank.state_dict(),
            "global_logits":global_logits.detach().cpu(),
            "router_states":{m:g.state_dict() for m,g in method_gates.items()},
            "config":vars(args),
        },
        out_dir/f"checkpoint_seed_{seed}.pt"
    )

    return rows


def summarize(rows:List[Row]) -> Tuple[str,List[dict]]:
    lines=[]
    methods=["softmax","temperature_softmax","epsilon_floor","logit_clip","residual_router"]
    summary_rows=[]

    global_glasses=[r for r in rows if r.method=="global" and r.corruption=="glasses" and r.k_shot==0]
    global_mean=float(np.mean([r.accuracy for r in global_glasses])) if global_glasses else float("nan")

    lines.append(f"Global glasses baseline accuracy: {global_mean:.4f}")
    lines.append("")

    for method in methods:
        lines.append(method.upper())
        z=[r for r in rows if r.method==method and r.corruption=="glasses" and r.k_shot==0]
        if z:
            zacc=float(np.mean([r.accuracy for r in z]))
            zsat=float(np.mean([r.saturation_rate for r in z]))
            zw=np.mean(np.array([[r.w_A_eye,r.w_B_lower_geometry,r.w_C_chroma] for r in z]),axis=0)
            lines.append(
                f"zero-shot acc={zacc:.4f} sat={zsat:.3f} "
                f"weights={zw.round(3).tolist()}"
            )

        for k in sorted(set(r.k_shot for r in rows if r.method==method and r.k_shot>0)):
            ss=[r for r in rows if r.method==method and r.corruption=="glasses" and r.k_shot==k]
            if not ss:
                continue
            acc=np.array([r.accuracy for r in ss],dtype=float)
            sat=np.array([r.saturation_rate for r in ss],dtype=float)
            rel=np.array([r.release_A for r in ss],dtype=float)
            rec=np.array([r.recovery_BC for r in ss],dtype=float)
            w=np.array([[r.w_A_eye,r.w_B_lower_geometry,r.w_C_chroma] for r in ss],dtype=float)

            lines.append(
                f"k={k:2d} acc={acc.mean():.4f}±{acc.std():.4f} "
                f"sat={sat.mean():.3f} releaseA={rel.mean():+.3f} "
                f"recoverBC={rec.mean():+.3f} "
                f"weights={w.mean(0).round(3).tolist()}"
            )

            summary_rows.append({
                "method":method,
                "k_shot":k,
                "accuracy_mean":float(acc.mean()),
                "accuracy_std":float(acc.std()),
                "gain_over_global":float(acc.mean()-global_mean),
                "saturation_mean":float(sat.mean()),
                "release_A_mean":float(rel.mean()),
                "recovery_BC_mean":float(rec.mean()),
                "w_A_mean":float(w[:,0].mean()),
                "w_B_mean":float(w[:,1].mean()),
                "w_C_mean":float(w[:,2].mean()),
            })
        lines.append("")

    # ranking by best mean gain over global
    ranking=[]
    for method in methods:
        candidates=[r for r in summary_rows if r["method"]==method]
        if candidates:
            best=max(candidates,key=lambda x:x["accuracy_mean"])
            ranking.append(best)
    ranking.sort(key=lambda x:x["accuracy_mean"],reverse=True)

    lines.append("BEST METHOD PER ABLATION")
    for r in ranking:
        lines.append(
            f"{r['method']:20s} best_k={r['k_shot']:2d} "
            f"acc={r['accuracy_mean']:.4f} "
            f"gain={r['gain_over_global']:+.4f} "
            f"sat={r['saturation_mean']:.3f} "
            f"releaseA={r['release_A_mean']:+.3f}"
        )

    return "\n".join(lines),summary_rows


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=["auto","mps","cpu","cuda"],default="auto")
    p.add_argument("--output",default="results")
    p.add_argument("--seeds",type=int,nargs="+",default=[11,22,33,44,55])

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

    p.add_argument("--meta-episodes",type=int,default=260)
    p.add_argument("--support-size",type=int,default=20)
    p.add_argument("--query-size",type=int,default=40)
    p.add_argument("--inner-lr",type=float,default=0.06)
    p.add_argument("--meta-lr",type=float,default=0.001)
    p.add_argument("--expert-dropout",type=float,default=0.30)
    p.add_argument("--entropy-floor",type=float,default=0.35)
    p.add_argument("--entropy-lambda",type=float,default=0.04)
    p.add_argument("--log-every",type=int,default=25)

    p.add_argument("--temperature",type=float,default=2.0)
    p.add_argument("--epsilon-floor",type=float,default=0.05)
    p.add_argument("--logit-clip",type=float,default=3.0)
    p.add_argument("--residual-scale",type=float,default=2.0)
    p.add_argument("--saturation-threshold",type=float,default=0.99)

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
    for seed in args.seeds:
        all_rows.extend(run_seed(args,seed,out))

    if not all_rows:
        raise RuntimeError("No result rows produced.")

    with open(out/"results.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(all_rows[0]).keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))

    summary_text,summary_rows=summarize(all_rows)

    with open(out/"per_method_summary.csv","w",newline="") as f:
        if summary_rows:
            w=csv.DictWriter(f,fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)

    payload={
        "config":vars(args),
        "rows":[asdict(r) for r in all_rows],
        "summary_rows":summary_rows,
        "summary":summary_text,
    }
    (out/"results.json").write_text(json.dumps(payload,indent=2))
    (out/"summary.txt").write_text(summary_text+"\n")

    print("\n"+"="*88)
    print("FINAL SUMMARY")
    print("="*88)
    print(summary_text)
    print(f"\nSaved to {out.resolve()}")


if __name__=="__main__":
    main()
