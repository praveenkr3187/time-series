"""Pretrained time-series foundation models, used zero-shot (no training on our data).

    Chronos-Bolt (small, 48M params)  univariate: sees only the series' own history
    Chronos-2    (120M params)        can also read past and known-future covariates (SNAP, events, price)

Both are downloaded from Hugging Face on first use (~180 MB / ~460 MB) and run on CPU.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch

log = logging.getLogger(__name__)
MODELS = {"Chronos-Bolt small": "amazon/chronos-bolt-small", "Chronos-2": "amazon/chronos-2"}


@lru_cache(maxsize=2)
def pipeline(name: str):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(MODELS[name], device_map="cpu", torch_dtype=torch.float32)


@torch.no_grad()
def forecast(name: str, Y: np.ndarray, h: int, quantiles: list[float], past_cov: dict[str, np.ndarray] | None = None,
             future_cov: dict[str, np.ndarray] | None = None, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Y [n, C] context. Covariates (Chronos-2 only): past [n, C], future [n, h] per name.

    Returns mean [n, h] and quantiles [len(quantiles), n, h].
    """
    p = pipeline(name)
    n = len(Y)
    mean = np.zeros((n, h), dtype=np.float32)
    Q = np.zeros((len(quantiles), n, h), dtype=np.float32)
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        if name == "Chronos-2":
            inputs = []
            for j in range(sl.start, min(sl.stop, n)):
                d = {"target": torch.from_numpy(Y[j].astype(np.float32))}
                if past_cov:
                    d["past_covariates"] = {k: torch.from_numpy(v[j].astype(np.float32)) for k, v in past_cov.items()}
                    d["future_covariates"] = {k: torch.from_numpy(v[j].astype(np.float32)) for k, v in future_cov.items()}
                inputs.append(d)
            q, m = p.predict_quantiles(inputs, prediction_length=h, quantile_levels=quantiles)
            mean[sl] = torch.stack([x[0] for x in m]).numpy()
            Q[:, sl] = torch.stack([x[0] for x in q]).permute(2, 0, 1).numpy()
        else:
            q, m = p.predict_quantiles([torch.from_numpy(y.astype(np.float32)) for y in Y[sl]], prediction_length=h,
                                       quantile_levels=quantiles)
            mean[sl] = m.numpy()
            Q[:, sl] = q.permute(2, 0, 1).numpy()
        if (i // batch_size) % 20 == 0:
            log.info("  %s %d/%d", name, min(i + batch_size, n), n)
    return np.clip(mean, 0, None), np.clip(Q, 0, None)
