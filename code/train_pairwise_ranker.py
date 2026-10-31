"""Reproduce the retained pairwise ridge model with one train-only alpha=100 fit.

Uses the supplied train bank and train references. No model selection or scoring
is performed. Inference uses the unchanged protocol_all.py runtime.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Set numerical-library thread limits before importing NumPy.
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_variable] = '4'

import numpy as np

from common import DATA, PROTOCOL, check_data, load_inputs, parse_answer, read_jsonl
from protocol_features import FEATURE_SCHEMA, build_features


ALPHA = 100.0


def candidate_labels(bank, references):
    """Return [N,4] final-answer correctness labels from aligned references."""
    if len(bank) != len(references) or any(
        row['id'] != ref['id'] or row['question'] != ref['question']
        for row, ref in zip(bank, references)
    ):
        raise ValueError('Reference length, IDs and questions must match')
    labels = []
    for row, ref in zip(bank, references):
        gold = parse_answer(ref['answer'])
        if gold is None or len(row['candidates']) != 4:
            raise ValueError('Expected a valid numeric reference and four candidates')
        correct = [parse_answer(c['text']) == gold for c in row['candidates']]
        if not any(correct):
            raise ValueError('Expected at least one correct candidate')
        labels.append(correct)
    return np.asarray(labels, dtype=np.float64).reshape(len(bank), 4)


def pairwise_design(features, labels, valid):
    """Build standardized, within-question correct-minus-wrong differences.

    The mean/std are fitted on ALL original training candidates, matching the
    pointwise control. Only valid parsed candidates participate in pairs.
    Each positive/negative pair occurs once; there is no target centering.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    valid = np.asarray(valid)
    if (x.ndim != 3 or x.shape[0] < 1 or x.shape[1] != 4 or x.shape[2] < 1
            or y.shape != x.shape[:2] or valid.shape != y.shape
            or valid.dtype != np.bool_ or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all()):
        raise ValueError('Expected finite [N,4,D] features, binary labels and a boolean valid mask')
    positive = y.astype(bool)
    if np.any(positive & ~valid) or not np.all(positive.any(axis=1)):
        raise ValueError('Every question needs a valid correct candidate')
    negative = valid & ~positive
    indices = np.argwhere(positive[:, :, None] & negative[:, None, :])
    if not len(indices):
        raise ValueError('No valid correct-wrong pairs available')
    flattened = x.reshape(-1, x.shape[-1])
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    q, correct, wrong = indices.T
    design = (x[q, correct] - x[q, wrong]) / scale
    return design, mean, scale, q


def fit_pairwise_ridge(features, labels, valid, alpha):
    """Minimize sum((D@u - 1)**2) + alpha*sum(u**2), no intercept.

    Fold the original train-candidate scaling into raw inference weights.
    The fixed mean-centering bias has no effect on within-question ordering.
    Scores are ranking values, not calibrated correctness probabilities.
    """
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive')
    design, mean, scale, groups = pairwise_design(features, labels, valid)
    gram = design.T @ design
    gram.flat[::gram.shape[0] + 1] += alpha
    rhs = design.sum(axis=0)
    standardized_weights = np.linalg.solve(gram, rhs)
    weights = standardized_weights / scale
    bias = -float(mean @ weights)
    diagnostics = {
        'pairs': len(design),
        'contributing_questions': len(np.unique(groups)),
        'normal_equation_relative_residual': float(
            np.linalg.norm(gram @ standardized_weights - rhs) / max(1.0, np.linalg.norm(rhs))),
        'objective_value': float(np.square(design @ standardized_weights - 1).sum()
                                 + alpha * (standardized_weights @ standardized_weights)),
    }
    return weights, bias, diagnostics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    code = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA,
                        help='Frozen data directory; only the train split is read')
    parser.add_argument('--output-dir', type=Path, default=code / 'retrained_model',
                        help='Empty destination for a new model_bundle')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('Use an empty output directory to protect existing model files')
    data = args.data_dir.resolve()
    started = time.perf_counter()
    manifest = check_data(data, ('train',))
    bank, arrays = load_inputs('train', data)
    labels = candidate_labels(bank, read_jsonl(data / 'gsm8k_train.jsonl'))
    features = build_features(bank, arrays['embeddings'])
    valid = np.asarray([[parse_answer(candidate['text']) is not None
                         for candidate in row['candidates']] for row in bank], dtype=bool)
    weights, bias, diagnostics = fit_pairwise_ridge(features, labels, valid, ALPHA)

    bundle = output / 'model_bundle'
    bundle.mkdir(parents=True)
    np.savez(bundle / 'model.npz', weights=weights, bias=np.array(bias),
             feature_schema=np.array(FEATURE_SCHEMA))
    data_names = ('candidates/gsm8k_train.jsonl', 'candidates/gsm8k_train.npz',
                  'gsm8k_train.jsonl')
    source_names = ('train_pairwise_ranker.py', 'common.py', 'protocol_features.py',
                    'protocol_all.py')
    metadata = {
        'protocol': PROTOCOL, 'feature_schema': FEATURE_SCHEMA,
        'objective': 'pairwise', 'alpha': ALPHA, 'fit_split': 'train',
        'train_questions': len(bank), 'train_candidates': int(labels.size),
        'pair_objective': 'sum((D@u-1)^2)+alpha*sum(u^2); no free intercept',
        'standardization': 'population mean/std of all original train candidates; std < 1e-12 -> 1; folded into raw coefficients',
        'inference': 'protocol_all.predict', 'scores_are_probabilities': False,
        'randomness': 'none; deterministic linear solve',
        'fit_diagnostics': diagnostics,
        'data_sha256': {name: manifest['sha256'][name] for name in data_names},
        'source_sha256': {name: sha256(code / name) for name in source_names},
        'checkpoint_sha256': sha256(bundle / 'model.npz'),
        'python': platform.python_version(), 'numpy': np.__version__,
        'thread_limits': {name: os.environ[name] for name in
                          ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'elapsed_seconds': time.perf_counter() - started,
    }
    (bundle / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'checkpoint': str(bundle), 'objective': 'pairwise', 'alpha': ALPHA,
                      'pairs': diagnostics['pairs'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
                      'elapsed_seconds': metadata['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
