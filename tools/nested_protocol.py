"""Nested selection over the training split: score the procedure, not the winner.

Outer folds are held out; inside each outer fold every candidate design is
cross-fitted on the remaining inner folds, the best combination is chosen on
that inner evidence only, refitted on all inner data, then scored on the outer
fold. The pooled outer score is the only number that may be quoted for the
method. Validation and test are never opened.

Usage:
    python tools/nested_protocol.py --designs A B C E --folds 3 --seed 17
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))

from common import load_bank, load_inputs, parse_answer, read_jsonl  # noqa: E402
from protocol_features import build_features  # noqa: E402

DESIGNS = {}


def design(name):
    def register(fn):
        DESIGNS[name] = fn
        return fn
    return register


def _stats(x):
    flat = x.reshape(-1, x.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


@design('A')
def pairwise_ridge(x, y, valid, contest, alpha=100.0):
    mu, sd = _stats(x)
    z = (x - mu) / sd
    rows = []
    for q in np.where(contest)[0]:
        for ci in [i for i in range(4) if valid[q, i] and y[q, i] == 1]:
            for wi in [i for i in range(4) if valid[q, i] and y[q, i] == 0]:
                rows.append(z[q, ci] - z[q, wi])
    d = np.asarray(rows)
    w = np.linalg.solve(d.T @ d + alpha * np.eye(d.shape[1]), d.sum(0))
    return lambda xx: ((xx - mu) / sd) @ w


@design('B')
def pointwise_ridge(x, y, valid, contest, alpha=100.0):
    mu, sd = _stats(x)
    z = (x - mu) / sd
    rows, targ = [], []
    for q in np.where(contest)[0]:
        for i in range(4):
            if valid[q, i]:
                rows.append(z[q, i])
                targ.append(y[q, i])
    a = np.hstack([np.asarray(rows), np.ones((len(rows), 1))])
    pen = np.eye(a.shape[1]) * alpha
    pen[-1, -1] = 0.0
    w = np.linalg.solve(a.T @ a + pen, a.T @ np.asarray(targ))
    return lambda xx: np.hstack([((xx - mu) / sd).reshape(-1, xx.shape[-1]),
                                 np.ones((xx.shape[0] * xx.shape[1], 1))]).reshape(
        xx.shape[0], xx.shape[1], -1) @ w


def _listwise(x, y, valid, contest, l2, steps=400):
    import torch
    mu, sd = _stats(x)
    z = (x - mu) / sd
    zt = torch.tensor(z, dtype=torch.float64)[contest]
    yt = torch.tensor(y, dtype=torch.float64)[contest]
    vt = torch.tensor(valid, dtype=torch.bool)[contest]
    w = torch.zeros(x.shape[-1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=steps, tolerance_grad=1e-10,
                            tolerance_change=1e-14, history_size=30,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        s = (zt @ w).masked_fill(~vt, -1e9)
        loss = -(torch.log_softmax(s, dim=1) * yt).sum() / yt.sum().clamp(min=1)
        loss = loss + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    coef = w.detach().numpy()
    return lambda xx: ((xx - mu) / sd) @ coef


DESIGNS['C'] = lambda x, y, v, c: _listwise(x, y, v, c, 1e-3)
DESIGNS['E'] = lambda x, y, v, c: _listwise(x, y, v, c, 1e-1)


def zrow(s):
    m, sd = s.mean(1, keepdims=True), s.std(1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (s - m) / sd


def score_of(s, valid):
    ch = np.where(valid, s, -1e18).argmax(1)
    return ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--designs', nargs='+', default=['A', 'B', 'C', 'E'])
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args()
    for d in args.designs:
        if d not in DESIGNS:
            raise SystemExit(f'unknown design {d}; have {sorted(DESIGNS)}')

    bank, arrays = load_inputs('train')
    gold = read_jsonl(CODE / 'data' / 'gsm8k_train.jsonl')
    x = build_features(bank, arrays['embeddings'])
    y = np.zeros((len(bank), 4))
    valid = np.zeros((len(bank), 4), dtype=bool)
    parses = []
    for q, (row, g) in enumerate(zip(bank, gold)):
        ref = parse_answer(g['answer'])
        p = [parse_answer(c['text']) for c in row['candidates']]
        parses.append(p)
        for i in range(4):
            valid[q, i] = p[i] is not None
            y[q, i] = float(p[i] == ref)
    nd = np.array([len({p for p in parses[q] if p is not None}) for q in range(len(bank))])
    contest = nd >= 2

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(bank))
    folds = np.empty(len(bank), dtype=int)
    folds[perm] = np.arange(len(bank)) % args.folds

    ok_all = np.zeros(len(bank), dtype=bool)
    seen = np.zeros(len(bank), dtype=bool)
    for f in range(args.folds):
        outer = folds == f
        inner = ~outer
        inner_ids = np.where(inner)[0]
        inner_folds = folds[inner_ids]
        full = {d: np.zeros((len(bank), 4)) for d in args.designs}
        for g in sorted(set(inner_folds.tolist())):
            gmask = np.zeros(len(bank), dtype=bool)
            gmask[inner_ids[inner_folds == g]] = True
            tr = ~outer & ~gmask
            for d in args.designs:
                full[d][gmask] = DESIGNS[d](x[tr], y[tr], valid[tr], contest[tr])(x[gmask])

        best, best_c = None, -1
        combos = [c for r in range(1, len(args.designs) + 1)
                  for c in itertools.combinations(args.designs, r)]
        for combo in combos:
            s = np.mean([zrow(full[d]) for d in combo], axis=0)
            c = int(sum(y[q, score_of(s, valid)[q]] == 1
                        for q in np.where(inner & contest)[0]))
            if c > best_c:
                best, best_c = combo, c

        out = np.zeros((len(bank), 4))
        for d in best:
            out[outer] = DESIGNS[d](x[inner], y[inner], valid[inner], contest[inner])(x[outer])
        s = zrow(out)
        ch = score_of(s, valid)
        ok = np.array([y[q, ch[q]] == 1 for q in range(len(bank))])
        ok_all |= ok & (outer & contest)
        seen |= outer & contest
        print(f'outer {f}: chose {"+".join(best)} (inner {best_c}) '
              f'-> outer {int((ok & outer & contest).sum())}/{int((outer & contest).sum())}')

    print(f'nested honest score: {int(ok_all.sum())}/{int(seen.sum())} '
          f'= {100*ok_all.sum()/seen.sum():.2f}%')


if __name__ == '__main__':
    main()
