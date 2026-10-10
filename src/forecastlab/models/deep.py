"""Global neural forecasters (PyTorch, CPU).

All models are *global*: one network for all 30,490 series, trained on random windows
(series, cut-off day) from the training period. Each window has

    context  C days of past sales + covariates   (what the model may see)
    horizon  H days of known-future covariates   (calendar, SNAP, events, price) and the H targets

Sales are divided by a per-window scale (1 + mean of the context) so a best-seller and a slow mover look alike
to the network; the output is multiplied back. Loss = Poisson negative log-likelihood (non-negative counts).

    LSTM / GRU      recurrent encoder over the context -> MLP head that emits all H days at once (direct, "MQ-RNN" style)
    Transformer     self-attention over context + horizon tokens (horizon tokens carry covariates, not sales)
    DeepAR          autoregressive LSTM with a negative-binomial likelihood; forecasts by sampling paths
                    -> a full predictive distribution, not just a mean
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from forecastlab.data.load import Dataset

log = logging.getLogger(__name__)
CONTEXT = 112
N_COV = 11  # 7 weekday one-hot + snap + event + log relative price + on-shelf


@dataclass
class Tensors:
    """Inputs aligned with ds.Y, kept compact; covariates are assembled per batch."""
    Y: np.ndarray         # [n, T] float32 sales
    logrel: np.ndarray    # [n, T] float32 log(price / 365-day mean price)
    on: np.ndarray        # [n, T] float32 on-shelf flag
    day: np.ndarray       # [T, 8] float32 weekday one-hot + event flag
    snap: np.ndarray      # [3, T] float32 SNAP per state
    state: np.ndarray     # [n] state code
    static: np.ndarray    # [n, 4] int64: store, dept, cat, item

    def cov(self, rows: np.ndarray, days: np.ndarray) -> np.ndarray:
        """[len(rows), days.shape[1], N_COV] covariates for a block of (row, day) pairs."""
        c = np.empty((*days.shape, N_COV), dtype=np.float32)
        c[..., :7] = self.day[days, :7]
        c[..., 7] = self.snap[self.state[rows][:, None], days]
        c[..., 8] = self.day[days, 7]
        c[..., 9] = self.logrel[rows[:, None], days]
        c[..., 10] = self.on[rows[:, None], days]
        return c


def prepare(ds: Dataset) -> Tensors:
    wd = np.eye(7, dtype=np.float32)[ds.cal["wday"].to_numpy() - 1]
    ev = (ds.cal["event_name_1"] != "").to_numpy(np.float32)
    snap = np.stack([ds.cal[f"snap_{s}"].to_numpy(np.float32) for s in ["CA", "TX", "WI"]])
    st = pd.Categorical(ds.series["state_id"], categories=["CA", "TX", "WI"]).codes.astype(np.int64)
    P = pd.DataFrame(ds.P.T).ffill().bfill()
    rel = np.nan_to_num(np.log((P / P.rolling(365, min_periods=1).mean()).to_numpy().T)).astype(np.float32)
    static = np.stack([pd.factorize(ds.series[c])[0] for c in ["store_id", "dept_id", "cat_id", "item_id"]], axis=1)
    return Tensors(ds.Y.astype(np.float32), rel, ds.on_shelf.astype(np.float32),
                   np.concatenate([wd, ev[:, None]], axis=1), snap, st, static.astype(np.int64))


def batch(t: Tensors, rows: np.ndarray, t0: np.ndarray, H: int, with_target: bool = True):
    """Windows whose context ends at day t0 (inclusive); horizon = t0+1 .. t0+H."""
    ctx = t0[:, None] + np.arange(-CONTEXT + 1, 1)[None]
    fut = t0[:, None] + np.arange(1, H + 1)[None]
    y_ctx = t.Y[rows[:, None], ctx]
    scale = 1.0 + y_ctx.mean(axis=1, keepdims=True)
    out = {
        "y_ctx": torch.from_numpy(y_ctx / scale),
        "cov_ctx": torch.from_numpy(t.cov(rows, ctx)),
        "cov_fut": torch.from_numpy(t.cov(rows, fut)),
        "static": torch.from_numpy(t.static[rows]),
        "scale": torch.from_numpy(scale.astype(np.float32)),
    }
    if with_target:
        out["y_fut"] = torch.from_numpy(t.Y[rows[:, None], fut])
        out["mask"] = torch.from_numpy(t.on[rows[:, None], fut])
    return out


class Static(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, d) for n, d in [(10, 4), (7, 4), (3, 2), (3049, 6)]])
        self.dim = sum(e.embedding_dim for e in self.embs)

    def forward(self, s):
        return torch.cat([e(s[:, i]) for i, e in enumerate(self.embs)], dim=-1)


class RNNForecaster(nn.Module):
    def __init__(self, H: int, cell: str = "lstm", hidden: int = 64):
        super().__init__()
        self.H, self.static = H, Static()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(1 + N_COV + self.static.dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden + H * N_COV + self.static.dim, 256), nn.ReLU(), nn.Linear(256, H))

    def forward(self, b):
        st = self.static(b["static"])
        x = torch.cat([b["y_ctx"][..., None], b["cov_ctx"], st[:, None].expand(-1, CONTEXT, -1)], dim=-1)
        _, hN = self.rnn(x)
        h = (hN[0] if isinstance(hN, tuple) else hN)[-1]
        z = torch.cat([h, b["cov_fut"].flatten(1), st], dim=-1)
        return nn.functional.softplus(self.head(z)) * b["scale"]


class TransformerForecaster(nn.Module):
    def __init__(self, H: int, d: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.H, self.static = H, Static()
        self.inp = nn.Linear(2 + N_COV + self.static.dim, d)   # sales, is_future flag, covariates, static
        self.pos = nn.Parameter(torch.randn(CONTEXT + H, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, 1)

    def forward(self, b):
        B = b["y_ctx"].shape[0]
        st = self.static(b["static"])[:, None].expand(-1, CONTEXT + self.H, -1)
        y = torch.cat([b["y_ctx"], torch.zeros(B, self.H)], dim=1)[..., None]
        flag = torch.cat([torch.zeros(B, CONTEXT), torch.ones(B, self.H)], dim=1)[..., None]
        cov = torch.cat([b["cov_ctx"], b["cov_fut"]], dim=1)
        z = self.enc(self.inp(torch.cat([y, flag, cov, st], dim=-1)) + self.pos)
        return nn.functional.softplus(self.out(z[:, CONTEXT:]).squeeze(-1)) * b["scale"]


class DeepAR(nn.Module):
    """Autoregressive LSTM; at each step emits negative-binomial (mean, dispersion) for the next day."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.static = Static()
        self.cell = nn.LSTM(1 + N_COV + self.static.dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.mu = nn.Linear(hidden, 1)
        self.alpha = nn.Linear(hidden, 1)

    def _params(self, h, scale):
        mu = nn.functional.softplus(self.mu(h)).squeeze(-1) * scale + 1e-4
        alpha = nn.functional.softplus(self.alpha(h)).squeeze(-1) / scale.sqrt() + 1e-4
        return mu, alpha

    def nll(self, b):
        """Teacher forcing over context + horizon: input y_{t-1}, predict y_t."""
        st = self.static(b["static"])
        H = b["y_fut"].shape[1]
        y_all = torch.cat([b["y_ctx"] * b["scale"], b["y_fut"]], dim=1)            # raw scale
        x_y = torch.cat([torch.zeros_like(y_all[:, :1]), y_all[:, :-1]], dim=1) / b["scale"]
        cov = torch.cat([b["cov_ctx"], b["cov_fut"]], dim=1)
        x = torch.cat([x_y[..., None], cov, st[:, None].expand(-1, cov.shape[1], -1)], dim=-1)
        h, _ = self.cell(x)
        mu, alpha = self._params(h, b["scale"])
        y = y_all
        r = 1.0 / alpha
        ll = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1) + r * torch.log(r / (r + mu))
              + y * torch.log(mu / (r + mu)))
        mask = torch.cat([b["cov_ctx"][..., 10], b["mask"]], dim=1)
        w = torch.ones_like(y)
        w[:, -H:] = 3.0  # emphasise the forecast horizon
        return -(ll * mask * w).sum() / (mask * w).sum()

    @torch.no_grad()
    def sample(self, b, H: int, n_samples: int = 50) -> torch.Tensor:
        st = self.static(b["static"])
        B = b["y_ctx"].shape[0]
        x_y = torch.cat([torch.zeros(B, 1), b["y_ctx"][:, :-1]], dim=1)
        x = torch.cat([x_y[..., None], b["cov_ctx"], st[:, None].expand(-1, CONTEXT, -1)], dim=-1)
        _, state = self.cell(x)
        rep = lambda z: z.repeat_interleave(n_samples, dim=0)  # noqa: E731
        state = tuple(rep(s.transpose(0, 1)).transpose(0, 1).contiguous() for s in state)
        scale, stn = rep(b["scale"]), rep(st)
        prev = rep(b["y_ctx"][:, -1:])
        out = []
        for k in range(H):
            xk = torch.cat([prev[..., None], rep(b["cov_fut"][:, k: k + 1]), stn[:, None]], dim=-1)
            h, state = self.cell(xk, state)
            mu, alpha = self._params(h[:, -1:], scale)
            r = 1.0 / alpha
            lam = torch.distributions.Gamma(r, r / mu).sample()
            y = torch.poisson(lam) * rep(b["cov_fut"][:, k: k + 1, 10])
            out.append(y)
            prev = y / scale
        return torch.cat(out, dim=1).view(B, n_samples, H)


def poisson_loss(pred, b):
    nll = pred - b["y_fut"] * torch.log(pred + 1e-6)
    return (nll * b["mask"]).sum() / b["mask"].sum().clamp(min=1)


def train(kind: str, t: Tensors, origin: int, H: int, start: int, steps: int = 3000, batch_size: int = 512,
          lr: float = 2e-3, seed: int = 0) -> nn.Module:
    """Train on windows whose horizon ends on or before `origin` (no test data)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = {"LSTM": lambda: RNNForecaster(H, "lstm"), "GRU": lambda: RNNForecaster(H, "gru"),
             "Transformer": lambda: TransformerForecaster(H), "DeepAR": lambda: DeepAR()}[kind]()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    n = len(t.Y)
    lo, hi = max(start, CONTEXT), origin - H
    # sample series proportional to sqrt(volume) so best-sellers are seen more often, but slow movers still appear
    vol = np.sqrt(t.Y[:, lo:hi].mean(axis=1) + 0.05)
    p = vol / vol.sum()
    model.train()
    t_start, run = time.time(), 0.0
    for step in range(steps):
        rows = rng.choice(n, batch_size, p=p)
        t0 = rng.integers(lo, hi + 1, batch_size)
        on = t.on[rows, t0] > 0
        b = batch(t, rows[on], t0[on], H)
        loss = model.nll(b) if kind == "DeepAR" else poisson_loss(model(b), b)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % 500 == 0:
            log.info("  %s step %d/%d loss %.4f (%.0fs)", kind, step + 1, steps, run, time.time() - t_start)
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, t: Tensors, origin: int, H: int, rows: np.ndarray | None = None,
            n_samples: int = 50, quantiles: tuple[float, ...] = ()) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(t.Y)) if rows is None else rows
    F = np.zeros((len(rows), H), dtype=np.float32)
    Q = np.zeros((len(quantiles), len(rows), H), dtype=np.float32)
    bs = 2048 if not isinstance(model, DeepAR) else 256
    for i in range(0, len(rows), bs):
        r = rows[i: i + bs]
        b = batch(t, r, np.full(len(r), origin), H, with_target=False)
        b["mask"] = b["cov_fut"][..., 10]
        if isinstance(model, DeepAR):
            s = model.sample(b, H, n_samples)
            F[i: i + bs] = s.mean(dim=1).numpy()
            for j, q in enumerate(quantiles):
                Q[j, i: i + bs] = torch.quantile(s, q, dim=1).numpy()
        else:
            F[i: i + bs] = (model(b) * b["mask"]).numpy()
    F *= t.on[rows, origin + 1: origin + 1 + H]
    return (F, Q) if quantiles else F


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

