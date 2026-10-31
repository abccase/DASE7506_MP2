"""Add one label-free protocol consistency feature to supplied embeddings.

This inference utility does not import diagnostic tools or reference data.
The official parser remains the authority for voting and evaluation.
"""
import re

import numpy as np

from common import parse_answer


FEATURE_SCHEMA = 'embedding-plus-parse-conflict-v1'
BOX = re.compile(r'\\boxed\{([^{}]+)\}')


def parsing_conflict(text):
    """Flag disagreement with the last simple numeric box, not correctness."""
    if '####' not in text:
        return False
    boxes = BOX.findall(text)
    if not boxes:
        return False
    boxed = parse_answer('#### ' + boxes[-1])
    return boxed is not None and parse_answer(text) != boxed


def build_features(bank, embeddings):
    """Return [N,4,D+1], appending a 0/1 conflict flag in the final column."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if (embeddings.ndim != 3 or embeddings.shape[:2] != (len(bank), 4)
            or embeddings.shape[-1] < 1 or not np.isfinite(embeddings).all()):
        raise ValueError('Expected finite embeddings with shape [N,4,D]')
    flags = np.zeros((len(bank), 4, 1), dtype=np.float64)
    for q, row in enumerate(bank):
        if len(row['candidates']) != 4:
            raise ValueError('Expected four candidates per question')
        for j, candidate in enumerate(row['candidates']):
            flags[q, j, 0] = parsing_conflict(candidate['text'])
    return np.concatenate((embeddings, flags), axis=-1)
