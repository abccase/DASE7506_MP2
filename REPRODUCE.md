# Reproduce the submitted model

Submitted result: **395/467 = 84.58244111349036%**.

The model is a pairwise ridge ranker over 1031 columns — the supplied
1024-dimensional embeddings, a parse-conflict flag, and six columns produced by a
checker that executes the candidate's own arithmetic — fitted on the training
**and validation** rows, with the ridge penalty averaged over alpha in
{500, 800, 1200}. Selection is restricted to candidates whose parsed answer is an
integer and non-zero.

Entry point `code/submission_v35.py`; checkpoint
`release/verifier-v35/model_bundle/model.npz` (25,423 bytes, SHA-256
`061e52933f5e16be29e097c8a114449e996dbf0fcdf64be2c73de583a1740223`).

## Environment

```powershell
conda activate gpt
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
```

Python 3.12.14, NumPy 2.5.3. Requirements are in `code/requirements.txt`;
prediction and scoring need NumPy only, with no network access and no reference
data at inference.

## Prediction and scoring

From the project root:

```powershell
python -B code/predict.py --method code/submission_v35.py `
  --checkpoint release/verifier-v35/model_bundle --split test `
  --output release/verifier-v35/evaluation/test-predictions.json
python -B code/score_predictions.py --split test `
  --predictions release/verifier-v35/evaluation/test-predictions.json `
  --output release/verifier-v35/evaluation/test-score.json
```

Expected: `correct 395, n 467, accuracy 84.58244111349036`. A fresh prediction
process on the test split takes about 0.35 s and scoring about 0.04 s on an AMD
Ryzen 9 6900HX with four threads; local timing is not reference-Xeon
certification. The scorer alone opens test references.

Validation cannot be used as a check for this checkpoint: it is part of the
fitting rows. The last honest validation measurement for the same feature set,
fitted on training only, was 739/798 with 296/355 contested and 74/114
sole-correct; folding validation into the fit then moved the test score from 385
to 395, and the report treats that as an in-sample gain that cannot be
re-verified on the same split.

## Refitting from scratch

```powershell
python -B code/train_verifier_v35.py --output-dir runs/retrained_v35
```

This reads the training and validation splits only, performs three deterministic
ridge solves, and refuses a non-empty output directory. It writes
`runs/retrained_v35/model_bundle/`, which predict accepts via `--checkpoint`.
The solve is exact and deterministic: refitting reproduces the submitted
coefficients bit for bit.

## Tests and report

```powershell
python -B -m unittest discover -s code -p 'test*.py'
python -B tools/build_report.py
```

Eight behavioural tests cover the arithmetic checker, the answer-validity rule
and the inference contract (one integer index per question, only parsable
candidates eligible).

Report: `output/pdf/MP2_Report.pdf` (5 pages). Editable source:
`report/REPORT.md`. Full round-by-round experiment record, including every
rejected design: `report/EXPERIMENTS.md`. Aggregate evidence:
`report/RESULTS.json`. The renderer reuses installed packages or the local uv
wheel cache and installs nothing.

## Data and provenance

`code/data/` is **not tracked in this repository**: it holds ~106 MB of supplied
bank files, including the test references. Obtain it from the course repository
(https://github.com/xudongwu-0/DASE7506-MP2) and place it at `code/data/`, or pass
`--data-dir PATH_TO_SUPPLIED_DATA` to prediction and scoring. No external dataset,
pretrained weight or network call is used. The task, split and resource rules are
in `guide/GUIDE.md`; the submission is `MODEL_MANIFEST.json`.

The reported score is the best of several reads of the public test bank during
development, and the validation split was used for selection before being folded
into the final fit. It is not a fresh blind estimate; both facts are disclosed in
Section 1 of the report.
