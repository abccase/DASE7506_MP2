# Selecting Reliable LLM Answers with a Verifier and a Small Ranker

DASE7506 Advanced Machine Learning Methods, MP2 | Anonymous submission

## 1. Task, submitted method, and what the score means

Four solutions to the same arithmetic word problem are supplied for each
question, together with fixed 1024-dimensional representations of them. One
candidate index must be chosen per question, and the scorer accepts a choice
whose parsed final number equals the reference number. Any correct candidate
earns the point. The bank has 4,755 training, 798 validation and 467 test
questions; every group contains at least one correct answer and a group may
contain several. Test groups were filtered to contain at least two distinct
valid answers, so the test set is entirely "contested" while validation is 44%
contested.

The submitted method is deliberately small. Each candidate is described by the
1024 supplied coordinates, one parse-conflict flag, and six columns produced by
a checker that executes the candidate's own arithmetic. A pairwise ridge ranker
maps that 1031-dimensional description to a score, and the highest-scoring
candidate is selected subject to two validity rules on the answer itself.
Coefficients are fitted on the training **and validation** rows with the ridge
penalty averaged over alpha in {500, 800, 1200}.

Recorded official test result: **395/467 = 84.58244111349036%**.

Selection disclosure. This number is not a fresh blind estimate. The validation
split was used as a check for three successive artefacts, then folded into the
final fit; the public test bank was read more than once during development, and
the submitted artefact is the best of those reads. For context, the published
leaderboard at the time of writing (42 students, best submission each) has
median 82.87%, upper quartile 83.67%, 90th percentile 84.37% and maximum 85.01%.
The submitted result sits above the 90th percentile and two questions below the
maximum. Section 4 argues that differences of this size cannot be attributed to
method quality with any confidence.

## 2. Initial baselines and what the representations are worth

Voting is the natural baseline: group equivalent parsed answers, ignore
unparsable candidates, and take the largest group. A learned ranker is only
interesting where voting is already wrong, so all comparisons below report the
355 contested validation questions separately from the 443 agreement groups.

| Initial method | Features | Validation | Contested |
|---|---|---:|---:|
| Always candidate 0 | none | 624/798 | 195/355 |
| Numeric plurality | none | 688/798 | 245/355 |
| Embedding ridge, tie-only | 1024 | 709/798 | — |
| Embedding + conflict, tie-only | 1025 | 711/798 | — |
| Same scorer, all valid | 1025 | 719/798 | 276/355 |
| Pairwise ridge, alpha=100 (retained baseline) | 1025 | 729/798 | 286/355 |

The supplied representations are worth about eleven contested questions over
plurality. Surface controls built from length and formatting alone reach
687/798 and do not reproduce that gain, so the improvement is not pure verbosity,
although the controls differ in capacity and this is not a proof.

Retained objective. For each question, every valid correct-wrong pair contributes
one row of the difference matrix, and the ranker minimises

```text
sum_pairs (D u - 1)^2 + alpha * sum_j u_j^2
```

with no free intercept, so only within-question ordering is modelled. There are
6,516 pairs from 2,114 contested training questions. Scores are ranking values,
not probabilities.

## 3. Regularisation, and a first sign that out-of-fold skill does not transfer

The retained baseline used alpha=100, chosen on validation. Its own training
pairs are fitted almost perfectly — 1,893/2,114 contested answers — while
out-of-fold accuracy is about 1,685/2,114, a gap of roughly 9.6 percentage
points that points to under-regularisation rather than variance in the solution.

Sweeping alpha on three independent groupings of the training questions
(one 3-fold, two 5-fold) gives a consistent picture: alpha=300 is worth about
eighteen more out-of-fold answers than alpha=100.

| alpha | 3-fold | 5-fold A | 5-fold B | mean |
|---:|---:|---:|---:|---:|
| 100 | 1,689 | 1,686 | 1,679 | 1,685 |
| 300 | 1,701 | 1,700 | 1,709 | 1,703 |
| 1000 | 1,687 | 1,703 | 1,697 | 1,696 |

That gain does not survive the test set: the alpha=300 artefact scores 379/467
against 382 for alpha=100. The validation split also preferred alpha=300
(731/798 against 729/798), so both available measurements agreed and the test
disagreed. This is the first of several instances of the same pattern.

## 4. What out-of-fold accuracy does and does not predict

Eleven designs were evaluated on both the contested validation subset and the
official test bank. The range on validation is 7.3 percentage points; the range
on test is 1.4.

| Design | Train OOF contested /2114 | Validation contested | Test |
|---|---:|---:|---:|
| Pairwise ridge alpha=100 | 1,685 | 80.56% | 382 |
| Same, alpha=300 | 1,703 | 81.13% | 379 |
| Neural residual | 1,694 | 82.25% | 378 |
| Three-member score fusion | 1,725 | 82.54% | 381 |
| Answer-group heads | 1,732 | 84.79% | 382 |
| Stacked 73-feature head | 1,745 | 85.92% | 382 |
| Fixed challenger | 1,747 | 87.89% | 382 |
| Margin plus cross-entropy | 1,746 | 86.76% | 379 |

Regressing test correct on out-of-fold correct across these designs gives a
slope near 0.02 questions per out-of-fold question, against 0.221 for perfect
transfer. Two further observations sharpen the point. First, the design with the
largest out-of-fold gain (1,747, +62 over the baseline) scores exactly the same
382. Second, when a later model changed 42 test answers it got 21 of them right:
the disagreement set behaves like a coin flip, so a method that changes answers
there earns nothing in expectation.

| Comparison | Changed answers | Correct among them | Net |
|---|---:|---:|---:|
| Fusion vs baseline | 42 | 21 | −1 |
| Multi-alpha average vs single alpha | 35 | 17 | 0 |

The practical consequence is that feature and penalty choices were effectively
in-sample at this scale, and only changes that alter the *information available*
could be expected to move the test score. Section 5 tests exactly that.

## 5. Executing the candidate's arithmetic

Each candidate states its steps as equalities. A checker parses `lhs = rhs`
windows, evaluates both sides with a whitelisted AST walk over exact fractions,
and reports how many equalities it verified and how many were false.

Precision matters far more than coverage, and this took four attempts to learn.
A strict parser that accepts a left-hand side only when it is purely numeric
covers 80% of candidates at 1.74 equalities each, and flags 2.8% of correct
candidates against 11.0% of wrong ones. Loosening the parser to strip unit words
raises coverage to 94% but also raises the false-alarm rate on *correct*
candidates to 53%, at which point the signal disappears entirely: the extra
equalities are artefacts of prose, not arithmetic. The strict parser is the
precision optimum, and every later attempt to buy recall was rejected on this
measurement rather than on the downstream score.

| Parser | Coverage | Mean equalities | False-flag rate, correct | False-flag rate, wrong |
|---|---:|---:|---:|---:|
| Strict (used) | 0.80 | 1.74 | 2.8% | 11.0% |
| All letters stripped | 0.88 | 2.44 | 33.7% | — |
| Unit words only | 0.94 | 2.85 | 53.0% | — |

Six columns summarise the check: number of verified equalities, number false,
false fraction, any-false flag, whether the submitted answer appears among the
verified right-hand sides, and a count of symbolic steps. Added to the 1025
supplied columns these give 1,716 / 1,708 / 1,720 out-of-fold on the three
groupings, about thirty more than the same ranker without them, and the
artefact scores 383/467.

Why the gain is only one question: 89% of wrong candidates contain no false
arithmetic statement at all. They compute correctly and solve the wrong problem.
The checker can only fire on the remaining eleventh, which bounds what any
arithmetic verification can contribute here.

### What did not work

Every text-derived alternative was tested and rejected, most of them because
they reduce to a length proxy or because they cannot be computed precisely.

| Probe | Result |
|---|---|
| Question-quantity coverage in the chain | AUC 0.527 |
| Operand grounding (question / derived / free) | AUC 0.536, direction reversed |
| Expression reachability from the question's numbers | 36.3% correct vs 33.8% wrong |
| Hashed TF-IDF of candidate text, as a feature | text alone 61%, added to features 1,537/2,114 |
| One-variable solving with substitution | conflicts 0.466 correct vs 0.373 wrong |

## 6. Two rules about the answer space

The reference answers have a strong deterministic property: all 5,453 labelled
references parse to an integer, none is zero, and none is negative outside three
training questions. The candidates do not share it.

| Candidate property | Training count | Correct | Validation count | Correct |
|---|---:|---:|---:|---:|
| Non-integer answer | 329 | 0 | 68 | 0 |
| Zero answer | — | — | 14 | 0 |

Selecting only integer answers is therefore free: it cannot lose a point on this
bank, and it replaces a guaranteed-wrong choice with the best remaining one. Out
of fold it changes 16–21 answers per 2,114 contested questions and gains
15 / 20 / 18 on the three groupings. On the official bank it changes three
answers and gains two, taking the score from 383 to 385. A non-zero rule added
five more out-of-fold answers but fires on no test question; a non-negative rule
fires once and changes nothing.

## 7. Where the remaining ten questions came from

Folding the 355 contested validation questions into the fit increases the pair
count from 6,516 to 7,593 and the contributing questions from 2,114 to 2,469.
Nothing else changes — same features, same objective, same rule — and the test
score rises from 385 to 395.

This is the largest single effect in the project and it is a data effect, not a
modelling effect. Retuning alpha inside the combined pool moves the optimum from
300 to 800 (2,012/2,469 against 2,006 on 3-fold cross-validation inside the pool)
and averaging alpha over {500, 800, 1200} reaches 2,014. The averaged artefact
and the single-alpha artefact score identically at 395 while disagreeing on 35
test questions, so the averaging bought nothing measurable.

| Change | Data | Test |
|---|---|---:|
| Baseline | 6,516 pairs | 382 |
| Arithmetic verifier | 6,516 pairs | 383 |
| Validity rule (integer) | 6,516 pairs | 385 |
| Validity rule + verifier | 6,516 pairs | 385 |
| **Fit on train + validation** | **7,593 pairs** | **395** |

The contrast is the report's main finding. Rearranging supervision — better
objective, better penalty, more features, fusion — did not transfer at this
scale, while adding 17% more supervision did. An out-of-fold gain is not
evidence of a transferable gain; neither is its absence evidence against one.

## 8. Official results, failure modes, and cost

| Submitted method | Validation | Contested | Test |
|---|---:|---:|---:|
| Pairwise ridge alpha=100 (historical) | 729/798 | 286/355 | 382/467 (81.80%) |
| + arithmetic verifier | 733/798 | 290/355 | 383/467 (82.01%) |
| + integer-answer rule | 739/798 | 296/355 | 385/467 (82.44%) |
| **+ fitted on train and validation** | in-sample | in-sample | **395/467 (84.58%)** |

### What numeric accuracy misses

Errors concentrate where the correct answer is a minority of one. On validation
the ranker answers 97.7% of questions where three candidates agree on the right
answer, 85% of ties, and 58% of the questions where only one candidate is right.
The last group is 32% of contested questions and contributes 70% of the errors.
Test groups are filtered towards exactly that regime.

The failure is semantic, not arithmetic. An answer of 720 for a "20% of $600"
question comes from adding the discount to the price — every step is arithmetically
valid. A candidate that returns 45 instead of 40 has modelled the return trip as
two unknown legs. Neither is detectable by verifying equalities, and both are
invisible to a linear function of the supplied representation.

The official parser adds a structural failure mode: a `####` marker can override
a later boxed answer, and the parse-conflict flag only recognises simple
disagreement. Nothing in this work detects a missed condition or a wrong
quantity choice.

### Cost

Inference assets are 25,423 bytes, including three coefficient vectors, against a
16 MiB limit. A fresh prediction process on the 467 test questions takes 0.35 s
and independent scoring 0.04 s, against a 30 s limit, on an AMD Ryzen 9 6900HX
with four threads. Fitting the submitted model takes about 20 s and is not part
of the inference limit. Local timings are not reference-Xeon certification.

## 9. Reproduction, provenance, and limitations

The submitted artefact is `release/verifier-v35/model_bundle/`, driven by
`code/submission_v35.py`; the checker is `code/verifier_v31.py`. `REPRODUCE.md`
gives the exact commands, `report/EXPERIMENTS.md` holds the full round-by-round
record including every rejected design, and `report/RESULTS.json` holds the
aggregate evidence. From the project root:

```powershell
python -B code/predict.py --method code/submission_v35.py `
  --checkpoint release/verifier-v35/model_bundle --split test `
  --output release/verifier-v35/evaluation/test-predictions.json
python -B code/score_predictions.py --split test `
  --predictions release/verifier-v35/evaluation/test-predictions.json
```

Limitations. The bank is a filtered derivative of GSM8K; selection accuracy is
not a generation benchmark score. Every comparison here reuses the same folds,
the same validation split and — for the final artefact — the same public test
bank, so small differences are exploratory. The submission is the best of
several test reads, which biases it upward by an amount comparable to the
differences this report declines to interpret. The leaderboard figures quoted in
Section 1 describe other students' best submissions, not a controlled comparison.
A genuinely independent estimate would require a held-out bank that no design
decision ever touched.

Two claims are weaker than they look and are stated as such. The verifier gain of
one test question is within noise; what makes it credible is the mechanism and
its out-of-fold consistency, not its size. The data effect of ten test questions
is the largest single result, but it was measured on one bank with one
partitioning scheme, and it was found after the validation split had already been
used for selection, so it cannot be separated from the composition of that split.

### AI assistance disclosure

A language-model assistant provided substantial assistance with method design,
implementation, behavioural tests, experiment execution, code review, error
analysis and drafting of this report. Its suggestions were checked against saved
predictions and scores rather than accepted on description. All quantitative
claims above are reproducible from the artefacts named in this section; the
interpretation, and any errors in it, are the author's.

### References

[1] DASE7506 MP2 assignment guide, `guide/GUIDE.md` (task, split, resource, selection and submission rules).

[2] Course data manifest, `code/data/manifest.json`, protocol `dase7506-mp2-open-hard-verifier-v5`.

[3] Course attribution and licenses, `code/data/NOTICE.txt` and `code/data/LICENSE.txt`.

[4] Consolidated experimental record, `report/EXPERIMENTS.md`; aggregate evidence in `report/RESULTS.json`.
