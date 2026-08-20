"""
Tibia OT Flow Matching Network
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Same architecture as femur OT-FM — reused directly.
N_LM is 8 (tibia) instead of 9 (femur), handled via cfg.py.

Architecture:
  BoundEnc  : tibial boundary PLY → dense feature seq (PT blocks)
  LMEnc     : 8 tibia landmarks   → condition vector
  CondFuse  : boundary + landmarks → fused 512-dim condition
  VFNet     : OT flow matching vector field (PT + CrossAttn, AdaLN time emb)
  FlowMatch : full model with Heun ODE solver at inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import math
import cfg


def chunked_knn(pos, k, chunk=512):
    B, N, _ = pos.shape
    out  = torch.zeros(B, N, k, dtype=torch.long, device=pos.device)
    p32  = pos.detach().float()
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        _, idx = torch.cdist(p32[:, s:e], p32).topk(k+1, dim=-1, largest=False)
        out[:, s:e] = idx[:, :, 1:k+1]
    return out


def gather_nb(x, idx):
    B, N, k = idx.shape
    D    = x.shape[-1]
    flat = idx.reshape(B, -1)
    return torch.gather(x, 1, flat.unsqueeze(-1).expand(-1,-1,D)).reshape(B,N,k,D)


def fps(pos, n):
    B, N, _ = pos.shape
    dev = pos.device
    if N <= n:
        return torch.arange(N, device=dev).unsqueeze(0).expand(B, -1)
    idx  = torch.zeros(B, n, dtype=torch.long, device=dev)
    dist = torch.full((B, N), 1e10, device=dev)
    far  = torch.randint(0, N, (B,), device=dev)
    bi   = torch.arange(B, device=dev)
    idx[:, 0] = far
    for i in range(1, n):
        d    = (pos.float() - pos[bi, far].unsqueeze(1).float()).pow(2).sum(-1)
        dist = torch.min(dist, d)
        far  = dist.argmax(-1)
        idx[:, i] = far
    return idx


def fps_g(x, idx):
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1,-1,x.shape[-1]))


class PTBlock(nn.Module):
    """Point Transformer block with float32 attention for stability."""
    def __init__(self, dim, heads=4, k=16, ffn=2, dropout=0.05):
        super().__init__()
        self.k     = k
        self.hd    = max(dim // heads, 1)
        self.heads = heads
        self.q     = nn.Linear(dim, dim)
        self.kk    = nn.Linear(dim, dim)
        self.v     = nn.Linear(dim, dim)
        self.pe    = nn.Sequential(nn.Linear(3,dim), nn.GELU(), nn.Linear(dim,dim))
        self.am    = nn.Sequential(nn.Linear(dim,dim), nn.GELU(), nn.Linear(dim,dim))
        self.out   = nn.Linear(dim, dim)
        self.n1    = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim,dim*ffn), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(dim*ffn,dim))
        self.n2    = nn.LayerNorm(dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, pos):
        res = x; x = self.n1(x)
        ki  = chunked_knn(pos, self.k)
        kn  = gather_nb(self.kk(x), ki)
        vn  = gather_nb(self.v(x),  ki)
        pe  = self.pe(gather_nb(pos, ki) - pos.unsqueeze(2))
        at  = self.am(self.q(x).unsqueeze(2).expand_as(kn) - kn + pe)
        at  = F.softmax(at.float().clamp(-30,30) / math.sqrt(self.hd), dim=2).to(x.dtype)
        o   = self.drop((at * (vn + pe)).sum(2))
        x   = res + self.out(o)
        return x + self.drop(self.ff(self.n2(x)))


class CrossAttnBlock(nn.Module):
    """Cross attention: tibia points attend to boundary condition features."""
    def __init__(self, dim, cond_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.hd    = dim // heads
        self.q     = nn.Linear(dim, dim)
        self.k     = nn.Linear(cond_dim, dim)
        self.v     = nn.Linear(cond_dim, dim)
        self.out   = nn.Linear(dim, dim)
        self.n1    = nn.LayerNorm(dim)
        self.n2    = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim,dim*2), nn.GELU(), nn.Linear(dim*2,dim))

    def forward(self, x, ctx):
        res = x; x = self.n1(x)
        B, N, D = x.shape
        M = ctx.shape[1]
        q = self.q(x).reshape(B, N, self.heads, self.hd).transpose(1,2)
        k = self.k(ctx).reshape(B, M, self.heads, self.hd).transpose(1,2)
        v = self.v(ctx).reshape(B, M, self.heads, self.hd).transpose(1,2)
        at = torch.matmul(q, k.transpose(-2,-1)).float() / math.sqrt(self.hd)
        at = F.softmax(at.clamp(-30,30), dim=-1).to(x.dtype)
        o  = torch.matmul(at, v).transpose(1,2).reshape(B, N, D)
        x  = res + self.out(o)
        return x + self.ff(self.n2(x))


class BoundEnc(nn.Module):
    """Encodes tibial boundary → global vector + per-point features."""
    def __init__(self):
        super().__init__()
        d = cfg.ENC_DIM
        self.fp     = cfg.ENC_FPS
        self.proj   = nn.Sequential(nn.Linear(3,d), nn.GELU(), nn.Linear(d,d))
        self.blocks = nn.ModuleList([
            PTBlock(d, cfg.ENC_HEADS, cfg.ENC_K, cfg.FFN_EXP, dropout=0.05)
            for _ in range(cfg.ENC_DEPTH)])
        self.pool_proj = nn.Sequential(nn.Linear(d*2, cfg.ENC_DIM), nn.GELU(),
                                       nn.Linear(cfg.ENC_DIM, cfg.ENC_DIM))

    def forward(self, pts):
        if pts.shape[1] > self.fp:
            fi  = fps(pts, self.fp)
            pts = fps_g(pts, fi)
        x = self.proj(pts)
        for b in self.blocks:
            x = b(x, pts)
        gv = self.pool_proj(torch.cat([x.max(1)[0], x.mean(1)], -1))
        return gv, x  # (B,D), (B,N,D)


class LMEnc(nn.Module):
    """Encodes 8 tibia landmarks → condition vector."""
    def __init__(self):
        super().__init__()
        d = cfg.LM_DIM
        self.emb    = nn.Sequential(nn.Linear(3,d), nn.GELU(), nn.Linear(d,d))
        self.le     = nn.Parameter(torch.randn(1, cfg.N_LM, d) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d,4,d*3,0.1,'gelu',batch_first=True)
            for _ in range(cfg.LM_DEPTH)])
        self.pool   = nn.Sequential(nn.Linear(d,d), nn.GELU(), nn.Linear(d,d))

    def forward(self, lm):
        x = self.emb(lm) + self.le
        for l in self.layers: x = l(x)
        return self.pool(x.max(1)[0]), x  # (B,D), (B,N_LM,D)


class CondFuse(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = cfg.ENC_DIM + cfg.LM_DIM
        self.f = nn.Sequential(
            nn.Linear(in_dim, cfg.COND_DIM), nn.GELU(),
            nn.Linear(cfg.COND_DIM, cfg.COND_DIM), nn.GELU(),
            nn.Linear(cfg.COND_DIM, cfg.COND_DIM))

    def forward(self, bf, lf):
        return self.f(torch.cat([bf, lf], -1))


class VFNet(nn.Module):
    """Vector field network: PT blocks + cross-attention + AdaLN time embedding."""
    def __init__(self):
        super().__init__()
        d = cfg.VF_DIM
        self.fp      = cfg.VF_FPS
        self.te_freq = nn.Parameter(torch.randn(64) * 0.02)
        self.te      = nn.Sequential(nn.Linear(128, d*2), nn.GELU(), nn.Linear(d*2, d))
        self.proj    = nn.Sequential(nn.Linear(3+d+cfg.COND_DIM, d), nn.GELU(),
                                     nn.Linear(d, d))
        self.pt_blocks    = nn.ModuleList([
            PTBlock(d, cfg.VF_HEADS, cfg.VF_K, cfg.FFN_EXP)
            for _ in range(cfg.VF_DEPTH)])
        self.cross_blocks = nn.ModuleList([
            CrossAttnBlock(d, cfg.ENC_DIM, heads=4)
            for _ in range(cfg.VF_DEPTH)])
        self.adaln = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(d, d*2))
            for _ in range(cfg.VF_DEPTH)])
        self.out   = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,d),
                                   nn.GELU(), nn.Linear(d,3))

    def time_emb(self, t):
        freq = self.te_freq.abs() * 2 * math.pi
        a    = t.unsqueeze(-1) * freq.unsqueeze(0)
        return self.te(torch.cat([a.cos(), a.sin()], -1))

    def forward(self, xt, t, cond, bound_feats):
        B, N, _ = xt.shape
        te  = self.time_emb(t)
        ce  = cond.unsqueeze(1).expand(B, N, -1)
        tee = te.unsqueeze(1).expand(B, N, -1)

        if N > self.fp:
            fi  = fps(xt, self.fp)
            xd  = fps_g(xt, fi)
            cd  = fps_g(ce, fi)
            tde = fps_g(tee, fi)
        else:
            xd, cd, tde, fi = xt, ce, tee, None

        feat = self.proj(torch.cat([xd, tde, cd], -1))

        for pt_blk, ca_blk, ada in zip(self.pt_blocks, self.cross_blocks, self.adaln):
            ada_out = ada(te)
            scale, shift = ada_out.chunk(2, dim=-1)
            feat = feat * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            feat = cp.checkpoint(pt_blk, feat, xd, use_reentrant=False)
            feat = ca_blk(feat, bound_feats)

        vd = self.out(feat).clamp(-10.0, 10.0)

        if fi is not None and N > self.fp:
            d      = torch.cdist(xt.float(), xd.float()).to(xt.dtype)
            _, idx = d.topk(3, -1, largest=False)
            kd     = torch.gather(d, 2, idx)
            kv     = gather_nb(vd, idx)
            w      = 1.0 / (kd + 1e-8)
            w      = w / w.sum(-1, keepdim=True)
            return (w.unsqueeze(-1) * kv).sum(2)
        return vd


class FlowMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.benc = BoundEnc()
        self.lenc = LMEnc()
        self.fuse = CondFuse()
        self.vf   = VFNet()

    def encode(self, bound, lm):
        bf, bfeats = self.benc(bound)
        lf, _      = self.lenc(lm)
        return self.fuse(bf, lf), bfeats

    def forward(self, tibia, bound, lm):
        tibia = torch.nan_to_num(tibia, 0.0)
        bound = torch.nan_to_num(bound, 0.0)
        lm    = torch.nan_to_num(lm,    0.0)

        bf, bfeats = self.benc(bound)
        lf, _      = self.lenc(lm)
        cond       = self.fuse(bf, lf)

        if torch.any(torch.isnan(cond)):
            return torch.tensor(0.0, device=tibia.device, requires_grad=True), {'loss': 0.0}

        B  = tibia.shape[0]
        t  = torch.rand(B, device=tibia.device).clamp(1e-5, 1-1e-5)
        x0 = torch.randn_like(tibia)
        te = t.view(B, 1, 1)

        xt     = (1 - (1 - cfg.SIGMA_MIN) * te) * x0 + te * tibia
        target = tibia - (1 - cfg.SIGMA_MIN) * x0

        pred = self.vf(xt, t, cond, bfeats)

        if torch.any(torch.isnan(pred)):
            return torch.tensor(0.0, device=tibia.device, requires_grad=True), {'loss': 0.0}

        loss = F.mse_loss(pred, target)
        if torch.isnan(loss):
            return torch.tensor(0.0, device=tibia.device, requires_grad=True), {'loss': 0.0}

        return loss, {'loss': loss.item()}

    @torch.no_grad()
    def generate(self, bound, lm, n_pts=None, steps=cfg.INF_STEPS, solver='heun'):
        if n_pts is None: n_pts = cfg.N_TIBIA
        B    = bound.shape[0]
        dev  = bound.device
        cond, bfeats = self.encode(bound, lm)
        x    = torch.randn(B, n_pts, 3, device=dev)
        dt   = 1.0 / steps

        for i in range(steps):
            t  = torch.full((B,), i * dt, device=dev)
            v1 = self.vf(x, t, cond, bfeats)
            if solver == 'euler':
                x = x + v1 * dt
            else:  # heun
                t2 = torch.full((B,), (i+1) * dt, device=dev)
                x2 = x + v1 * dt
                v2 = self.vf(x2, t2, cond, bfeats)
                x  = x + 0.5 * (v1 + v2) * dt
        return x


def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
