"""Inference for v35: averaged multi-alpha scorer with the integer rule."""
from fractions import Fraction
from pathlib import Path

import numpy as np

from common import parse_answer
from protocol_features import build_features
from verifier_v31 import build_rows

SCHEMA = 'embedding-plus-parse-conflict-plus-arithmetic-verifier-v31'


def _legal(parsed):
    if parsed is None:
        return False
    try:
        v = Fraction(parsed)
    except (ValueError, ZeroDivisionError):
        return False
    return v.denominator == 1 and v != 0


def _zrow(s, idx):
    out = np.zeros_like(s)
    for q in range(s.shape[0]):
        ii = idx[q]
        if len(ii) < 2:
            continue
        v = s[q, ii]
        sd = float(v.std())
        out[q, ii] = 0.0 if sd < 1e-12 else (v - v.mean()) / sd
    return out


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Provide --checkpoint pointing to the v35 bundle')
    with np.load(Path(checkpoint) / 'model.npz', allow_pickle=False) as saved:
        if str(saved['feature_schema'].item()) != SCHEMA:
            raise ValueError('Unsupported feature schema')
        alphas = [int(a) for a in saved['alphas']]
        weights = [(np.asarray(saved[f'w_{a}'], dtype=np.float64), float(saved[f'b_{a}']))
                   for a in alphas]

    features = np.concatenate([build_features(bank, arrays['embeddings']),
                               build_rows(bank)], axis=2)
    if features.shape[-1] != weights[0][0].size:
        raise ValueError('Feature width does not match the checkpoint')

    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    idx = [np.array([i for i, p in enumerate(ps) if p is not None], dtype=int) for ps in parsed]
    fused = np.zeros(features.shape[:2])
    for w, b in weights:
        fused += _zrow(features @ w + b, idx)
    fused /= len(weights)

    choices = []
    for q, ps in enumerate(parsed):
        if not len(idx[q]):
            choices.append(0)
            continue
        legal = [i for i in idx[q] if _legal(ps[i])]
        pool = legal if legal else list(idx[q])
        choices.append(int(max(pool, key=lambda i: fused[q, i])))
    return choices
