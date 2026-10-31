# MP2 submitted model

Start with [REPRODUCE.md](../REPRODUCE.md) for the exact evaluation command. The
report is [MP2_Report.pdf](../output/pdf/MP2_Report.pdf); the full round-by-round
experiment record is [report/EXPERIMENTS.md](../report/EXPERIMENTS.md); the task
and grading rules are in [GUIDE.md](../guide/GUIDE.md).

Submitted result: **395/467 = 84.58244111349036%**. The model is a pairwise ridge
ranker over 1031 columns — the supplied 1024-dimensional embeddings, a
parse-conflict flag, and six columns from a checker that executes the candidate's
own arithmetic — fitted on the training and validation rows with the penalty
averaged over alpha in {500, 800, 1200}. Selection is restricted to candidate
answers that are integers and non-zero.

| Path | Purpose |
|---|---|
| submission_v35.py | Prediction entry point: averaged scorer plus the answer-validity rule. |
| verifier_v31.py | Arithmetic checker producing the six extra columns. |
| protocol_features.py | Supplied embeddings plus the parse-conflict flag. |
| train_verifier_v35.py | Refits the submitted checkpoint on train + validation. |
| train_pairwise_ranker.py | The pairwise objective, shared by the trainer. |
| common.py, predict.py, score_predictions.py | Unchanged official course IO, parser and evaluator. |
| test_submission_v35.py | Behavioural tests for the verifier, the rule and the inference contract. |
| data/ | Complete supplied data, manifest, licence and attribution. |
| ../release/verifier-v35/ | Submitted checkpoint (25,423 bytes) and its recorded evaluation. |

Prediction, scoring and fitting all run offline on NumPy alone. Keep the
four-thread settings from `REPRODUCE.md`.
