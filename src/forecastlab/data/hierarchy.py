"""The M5 product/location hierarchy as a sparse summing matrix.

    S [n_aggregate_series, n_bottom_series]  with  y_aggregate = S @ y_bottom

Twelve levels, 42,840 series in total (M5 definition):

    1 total            5 dept               9  store x dept
    2 state            6 state x cat        10 item
    3 store            7 state x dept       11 item x state
    4 category         8 store x cat        12 item x store  (= the 30,490 bottom series)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

LEVELS: list[tuple[str, list[str]]] = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("category", ["cat_id"]),
    ("department", ["dept_id"]),
    ("state x category", ["state_id", "cat_id"]),
    ("state x department", ["state_id", "dept_id"]),
    ("store x category", ["store_id", "cat_id"]),
    ("store x department", ["store_id", "dept_id"]),
    ("item", ["item_id"]),
    ("item x state", ["item_id", "state_id"]),
    ("item x store", ["item_id", "store_id"]),
]


@dataclass
class Hierarchy:
    S: sp.csr_matrix          # [n_agg, n_bottom]
    keys: pd.DataFrame        # one row per aggregate series: level (1-12), level_name, key

    def level_rows(self, level: int) -> np.ndarray:
        return np.flatnonzero(self.keys["level"].to_numpy() == level)

    def aggregate(self, X: np.ndarray, level: int | None = None) -> np.ndarray:
        """Sum bottom-level rows into every aggregate series (or one level)."""
        S = self.S if level is None else self.S[self.level_rows(level)]
        return np.asarray(S @ X)


def build_hierarchy(series: pd.DataFrame, levels: list[tuple[str, list[str]]] = LEVELS) -> Hierarchy:
    blocks, keys = [], []
    n = len(series)
    for lvl, (name, cols) in enumerate(levels, start=1):
        if cols:
            key = series[cols].astype(str).agg("/".join, axis=1)
        else:
            key = pd.Series(["total"] * n)
        codes, uniq = pd.factorize(key, sort=False)
        blocks.append(sp.csr_matrix((np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(len(uniq), n)))
        keys.append(pd.DataFrame({"level": lvl, "level_name": name, "key": uniq}))
    return Hierarchy(sp.vstack(blocks).tocsr(), pd.concat(keys, ignore_index=True))


@lru_cache(maxsize=1)
def m5_hierarchy() -> Hierarchy:
    from forecastlab.data.load import load_dataset
    return build_hierarchy(load_dataset().series)
