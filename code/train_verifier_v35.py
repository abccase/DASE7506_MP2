"""v35: average of three regularisation strengths fitted on train+validation.

The pool grew from 6516 to 7593 pairs, which moves the CV optimum from alpha=300
to alpha=800; an equal-weight average of alpha in {500, 800, 1200} scored 2014/2469
on 3-fold CV inside the combined pool, against 2006 for alpha=300 alone.

Selection still applies the integer rule from v32.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '4'

import numpy as np

from common import DATA, PROTOCOL, load_inputs, parse_answer, read_jsonl
from protocol_features import FEATURE_SCHEMA, build_features
from train_pairwise_ranker import candidate_labels, fit_pairwise_ridge
from verifier_v31 import COLUMNS, build_rows

ALPHAS = (500.0, 800.0, 1200.0)
SCHEMA = 'embedding-plus-parse-conflict-plus-arithmetic-verifier-v31'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=DATA)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir.resolve() / 'model_bundle'
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'refusing to overwrite non-empty {out}')

    banks, embs, labels = [], [], []
    for split in ('train', 'validation'):
        bank, arr = load_inputs(split, args.data_dir)
        banks += bank
        embs.append(arr['embeddings'])
        labels.append(candidate_labels(bank, read_jsonl(args.data_dir / f'gsm8k_{split}.jsonl')))
    emb = np.concatenate(embs, 0)
    labels = np.concatenate(labels, 0)
    feats = np.concatenate([build_features(banks, emb), build_rows(banks)], axis=2)
    valid = np.asarray([[parse_answer(c['text']) is not None for c in row['candidates']]
                        for row in banks], dtype=bool)

    out.mkdir(parents=True, exist_ok=True)
    payload = {'feature_schema': np.array(SCHEMA), 'alphas': np.array(ALPHAS)}
    diags = {}
    for a in ALPHAS:
        w, b, diag = fit_pairwise_ridge(feats, labels, valid, a)
        payload[f'w_{int(a)}'] = w
        payload[f'b_{int(a)}'] = np.array(b)
        diags[str(int(a))] = diag
    np.savez_compressed(out / 'model.npz', **payload)

    meta = {
        'protocol': PROTOCOL, 'feature_schema': SCHEMA, 'base_schema': FEATURE_SCHEMA,
        'verifier_columns': list(COLUMNS), 'objective': 'pairwise',
        'alphas': list(ALPHAS), 'fusion': 'equal-weight mean of question-wise z-scores',
        'fit_split': 'train+validation',
        'note': 'validation contributed no independent evidence for this artifact; '
                'it was used as a check three times before being folded into the fit',
        'fit_diagnostics': diags,
        'checkpoint_sha256': sha256(out / 'model.npz'),
    }
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pairs': diags['800']['pairs'],
                      'assets': (out / 'model.npz').stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
