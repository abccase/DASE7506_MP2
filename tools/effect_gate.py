"""Paired effect gate for two frozen prediction files.

Prints discordance, net difference, its standard error and z-score, and the
minimum detectable effect implied by the observed discordance. Refuses to call
an improvement below the declared z threshold.

Usage:
    python tools/effect_gate.py --split validation \
        --baseline release/neural-v9-eval/evaluation/validation-predictions.json \
        --challenger release/blend-v29/evaluation/validation-predictions.json \
        --min-z 2.0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))

from common import load_bank, parse_answer, read_jsonl, score_choices  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='validation')
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--challenger', type=Path, required=True)
    ap.add_argument('--min-z', type=float, default=2.0)
    args = ap.parse_args()

    base = json.loads(args.baseline.read_text())
    chal = json.loads(args.challenger.read_text())
    if base['ids'] != chal['ids']:
        raise SystemExit('prediction files are not row-aligned')

    bank = load_bank(args.split)
    gold = read_jsonl(ROOT / 'code' / 'data' / f'gsm8k_{args.split}.jsonl')
    ok_b = np.zeros(len(bank), dtype=bool)
    ok_c = np.zeros(len(bank), dtype=bool)
    for q, (row, g) in enumerate(zip(bank, gold)):
        ref = parse_answer(g['answer'])
        pb = parse_answer(row['candidates'][base['selected'][q]]['text'])
        pc = parse_answer(row['candidates'][chal['selected'][q]]['text'])
        ok_b[q] = pb is not None and pb == ref
        ok_c[q] = pc is not None and pc == ref

    cb, cc = int(ok_b.sum()), int(ok_c.sum())
    b = int((ok_c & ~ok_b).sum())
    c = int((ok_b & ~ok_c).sum())
    net = b - c
    se = float(np.sqrt(max(b + c, 1)))
    z = net / se
    n = len(bank)
    print(f'split {args.split}  n={n}')
    print(f'  baseline   {cb}/{n} = {100*cb/n:.2f}%')
    print(f'  challenger {cc}/{n} = {100*cc/n:.2f}%')
    print(f'  fixed by challenger {b}, broken {c}, net {net:+d} ({100*net/n:+.2f} pp)')
    print(f'  discordance {b+c}, SE {se:.1f} questions, z = {z:+.2f}')
    print(f'  minimum detectable effect at z={args.min_z}: '
          f'{args.min_z*se/n*100:.2f} pp ({int(np.ceil(args.min_z*se))} questions)')
    if z < args.min_z:
        print('  verdict: BELOW GATE - not an improvement, even if the sign is positive')
    else:
        print('  verdict: passes the declared gate')


if __name__ == '__main__':
    main()
