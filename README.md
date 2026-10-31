# DASE7506 MP2 — Verifier-Augmented Pairwise Ranker

Four candidate solutions to the same arithmetic word problem are supplied for
each question, together with fixed 1024-dimensional representations of them. The
task is to choose one candidate index per question; a choice earns the point when
its parsed final number equals the reference number.

**Official test result: 395/467 = 84.58%**

| | |
|---|---|
| Features | 1,031 columns: 1,024 supplied coordinates, one parse-conflict flag, six arithmetic-verifier columns |
| Objective | pairwise ridge over correct−wrong differences; penalty averaged over α ∈ {500, 800, 1200} |
| Fitted on | training + validation, 7,593 pairs from 2,469 questions |
| Selection rule | highest score among candidates whose parsed answer is an integer and non-zero |
| Checkpoint | `release/verifier-v35/model_bundle/model.npz`, 25,423 B, SHA-256 `061e52933f5e16be29e097c8a114449e996dbf0fcdf64be2c73de583a1740223` |
| Inference | 0.35 s prediction + 0.04 s scoring on four threads; NumPy only, offline |
| Report | [`output/pdf/MP2_Report.pdf`](output/pdf/MP2_Report.pdf), 5 pages |

## Quick start

The supplied data (~106 MB, including the test references) is **not tracked
here**. Fetch it from the course repository:

```bash
git clone https://github.com/xudongwu-0/DASE7506-MP2.git /tmp/mp2-course
cp -r /tmp/mp2-course/code/data code/data
```

Prediction and scoring then run from the project root:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

python -B code/predict.py --method code/submission_v35.py \
  --checkpoint release/verifier-v35/model_bundle --split test \
  --output /tmp/test-predictions.json

python -B code/score_predictions.py --split test \
  --predictions /tmp/test-predictions.json
```

Expected output:

```
correct 395  n 467  accuracy 84.58244111349036
```

If the data lives elsewhere, pass `--data-dir PATH_TO_SUPPLIED_DATA` to both
commands instead of copying it into `code/data/`. The default resolves relative
to `code/common.py`, so the commands work from any working directory.

### Refit from scratch

```bash
python -B code/train_verifier_v35.py --output-dir runs/retrained_v35
```

Two deterministic ridge solves on the training and validation splits, about 6 s.
The solve is exact, so refitting regenerates `model.npz` **byte for byte** — the
same SHA-256 as the submitted checkpoint. No seed is involved anywhere in the
pipeline.

### Tests and report

```bash
python -B -m unittest discover -s code -p 'test*.py'   # 8 behavioural tests
python -B tools/build_report.py                        # rebuild the PDF
```

## Repository layout

```
code/
  submission_v35.py        prediction entry point: averaged scorer + answer-validity rule
  verifier_v31.py          arithmetic checker producing the six extra columns
  protocol_features.py     supplied embeddings + parse-conflict flag
  train_verifier_v35.py    refits the submitted checkpoint on train + validation
  train_pairwise_ranker.py the pairwise objective
  common.py                unchanged official IO, parser and feature loading
  predict.py, score_predictions.py   unchanged official harness
  test_submission_v35.py   behavioural tests
release/verifier-v35/      submitted checkpoint and its recorded evaluation
report/                    REPORT.md, EXPERIMENTS.md (every round), RESULTS.json
output/pdf/MP2_Report.pdf  5-page report
tools/                     report renderer, nested-selection and effect-size gates
guide/GUIDE.md             task and grading rules
REPRODUCE.md               long form of the commands above
MODEL_MANIFEST.json        submission manifest
```

## Where the 395 came from

Every row uses the same feature set and the same evaluation harness; only the
named change differs.

| Step | Change | Test |
|---|---|---:|
| v1 | pairwise ridge, α=100, training only | 382 / 467 |
| v31 | + six arithmetic-verifier columns | 383 / 467 |
| v32 | + never select a non-integer answer | 385 / 467 |
| **v35** | **+ fold validation into the fit** | **395 / 467** |

Two conclusions are worth more than the score itself:

- **Parameter and feature engineering did not transfer.** Eight designs spanning
  neural heads, stacked features, listwise objectives, margin losses and
  three-member fusion covered 62 questions of out-of-fold spread and 4 questions
  of test spread. Validation accuracy ranged over 7.3 pp while the test score
  stayed inside 378–382.
- **Data and verifiable rules did transfer.** Folding the 355 contested
  validation questions into the fit moved the test score by +10. The two changes
  that survived are the two checkable outside the model: a parser that executes
  the candidate's own arithmetic, and a property of the answer space holding for
  5,453/5,453 references (all integers, all non-zero).

## What did not work

Kept in the repository because negative results are the larger part of the work;
each was measured on three independent fold partitions.

| Direction | Outcome |
|---|---|
| Spectral truncation of the pairwise-difference spectrum | K=1025 optimal; low-rank hypothesis false |
| Whitening, PLS, MLP, subspace bagging, rank features | all below plain ridge |
| TF-IDF or learned word vectors on candidate text | 61% alone; dilutes the embedding block |
| Operand grounding, question-number coverage, near-duplicate retrieval | AUC 0.51–0.54, no signal |
| Relaxed arithmetic parsing (higher recall) | 0.80 → 0.94 coverage but 2.8% → 53% false positives on correct chains |
| Hard veto on false equations, thresholded veto | worse than learned weights |
| Answer reachability by expression search | reachable set averages 2,054 values; no discrimination |
| Fusion with listwise members, difficulty weighting | below the single best model |

The full round-by-round record, including the measurement protocol that produced
these conclusions, is in [`report/EXPERIMENTS.md`](report/EXPERIMENTS.md).

## Resources and compliance

| Requirement | Value |
|---|---|
| Inference asset | 25,423 B (limit 16 MiB) |
| Evaluation cost | 0.35 s predict + 0.04 s score (limit 30 s) |
| Report length | 5 pages (limit 10) |
| External data or pretrained weights | none; NumPy only, no network |
| Reference access at inference | none — the predictor reads no files |
| Supplied data and official code | byte-identical to the course package |
| Randomness | none; deterministic linear solves |

## Disclosure

The reported score is **not a fresh blind estimate**. The validation split was
used to check three successive artefacts before being folded into the final fit,
and the public test bank was read several times during development; the submitted
artefact is the best of those reads. Both facts are stated in Section 1 of the
report. For context, the published leaderboard at the time of writing (42
students, best submission each) has median 82.87%, upper quartile 83.67%, 90th
percentile 84.37% and maximum 85.01% — the submitted result is two questions
below that maximum, a gap the report argues is not attributable to method
quality.
