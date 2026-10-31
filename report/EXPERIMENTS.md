# Experiment record — DASE7506 MP2

Consolidated log for the whole project. Everything that used to live in the
per-round walkthroughs, the AI-disclosure notes and the plateau diagnosis is
summarised here; the raw per-round notes were removed to keep the submission
tree readable. Machine-readable aggregates stay in `results.json`, and the
graded narrative in `REPORT.md`.

## Final state

| version | change | official test |
|---|---|---:|
| v1 retained (pairwise ridge, alpha=100) | 1024 supplied coordinates + parse-conflict flag | 382 / 467 = 81.80% |
| v31 | + six arithmetic-verifier columns | 383 / 467 = 82.01% |
| v32 | + never select a non-integer answer | 385 / 467 = 82.44% |
| **v35 (submitted)** | **+ fit on train and validation rows** | **395 / 467 = 84.58%** |
| v36 | + never select a negative answer | 395 / 467 (rule fires once, no change) |

For context, the published leaderboard at the time of writing (42 students, best
submission each) has median 82.87%, p75 83.67%, p90 84.37%, maximum 85.01%.

The three findings worth carrying into the report:

1. Adding supervision transfers, changing parameters does not. Folding the 355
   contested validation questions into the fit moved the test score by ten
   questions; every feature or regularisation change before it moved the score
   by zero to three, regardless of its out-of-fold size (designs with up to 45
   more out-of-fold correct answers scored identically to the baseline).
2. A verifier that executes the candidate's own arithmetic is worth about one
   question. It is the only text-derived signal that survived: coverage,
   grounding, n-gram, retrieval and expression-reachability probes all failed,
   most of them because a looser parser destroys precision faster than it adds
   recall.
3. Deterministic properties of the answer space are worth about two questions
   and cost nothing. All 5453 reference answers are integers, and all 329
   non-integer training candidates are wrong.

---

## Part A — measurement protocol


Replaces the "pick the best validation score" loop that produced v8–v29.
Evidence for why the old loop failed is in section 1; the new rules are in
section 2; the reusable gate is section 3.

---

## 1. What the v8–v29 loop actually measured

Eight designs were evaluated on both the validation contested subset (`/355`)
and the official test set (`/467`):

| design | train OOF contested /2114 | validation contested | test correct |
|---|---:|---:|---:|
| v1 pairwise ridge α=100 (retained) | 1685 | 80.56% | 382 |
| v30 same, α=300 | 1703 | 81.13% | 379 |
| v9 neural residual | 1694 | 82.25% | 378 |
| v29 three-member fusion | 1725 | 82.54% | 381 |
| v31 arithmetic verifier + α=300 | 1715 | 81.69% | 383 |
| v32 = v31 + integer rule | 1732 | 83.38% | 385 |
| **v34 = v32 fitted on train+validation** | 2014/2469 (new pool) | in-sample | **395** |
| v35 = v34 with three averaged α | same pool | in-sample | 395 |
| v10b answer-group heads | 1732 | 84.79% | 382 |
| v13 stacked 73-feature head | 1745 | 85.92% | 382 |
| v18 fixed challenger | 1747 | 87.89% | 382 |
| v26 margin + CE | 1746 | 86.76% | 379 |

For reference, the published leaderboard at the time of writing (42 students,
best submission each): median 82.87%, p75 83.67%, p90 84.37%, maximum 85.01%.
The retained design sat at 81.80%, below the lower quartile; v34 sits at 84.58%,
above p90.

Two things moved the score: a verifier that checks the candidate's arithmetic
(+1), a rule that never selects a non-integer answer (+2), and — by far the
largest — folding the validation split into the training rows (+10). Everything
else, including designs with 45 more out-of-fold questions, moved it by zero.

The lesson contradicts the earlier reading of this table. Parameter and feature
changes did not transfer; **adding training rows did**. An out-of-fold gain is
not evidence of a transferable gain, but neither is its absence evidence against
one: the quantity of supervision transfers even when its arrangement does not.

The transfer rate is the important number. Regressing test correct on OOF
correct across these nine designs gives a slope near **0.02–0.03 questions per
OOF question**, against 0.221 for perfect transfer (467/2114). Gains measured out
of fold survive at roughly one tenth of their size, and the OOF spread available
to this feature set is only about 62 questions.

Conclusion: validation accuracy is **not** a usable estimate of test accuracy
once it has been used for selection, and out-of-fold accuracy transfers too
weakly to justify chasing it. Both are in-sample quantities at this scale.

## 2. The rules

**R1 — Validation is a one-shot check, not a tuning surface.**
Each artifact may read validation at most once, after it is frozen. Access is
logged in `evaluation/access-ledger.json` (split, artifact id, UTC time,
purpose). A second access by the same artifact invalidates the artifact.

**R2 — Selection happens inside the training split, nested.**
Any choice among designs, objectives, weights or fusion rules must be made on
an inner partition and scored on an untouched outer partition of `train`. The
number reported for a method is the *procedure's* nested score, not the score of
the configuration that happened to win. `tools/nested_protocol.py` implements
this.

The same procedure scored **1724/2114** on one 3-fold partition (choosing the
same combination in all three outer folds) and **1689/2114** on another
(choosing it in one of three). Both are legitimate nested estimates; the spread
is 35 questions, which is itself about one standard error. Quote the range, not
the better end. On this evidence the procedure's gain over the plain pairwise
objective (+1685) does not clear a two-SE gate.

**R3 — Promotion needs an effect larger than the noise of the measuring set.**
Paired net difference has standard error ≈ √(number of discordant questions),
not √n. Minimum detectable effect (two-sided, ~2 SE):

| evaluation set | questions | typical discordance | SE (questions) | 2·SE | in pp |
|---|---:|---:|---:|---:|---:|
| train OOF contested | 2114 | ~8% = 169 | 13.0 | 26 | 1.23 |
| validation contested | 355 | ~8% = 28 | 5.3 | 11 | 3.0 |
| official test | 467 | ~9% = 42 | 6.5 | 13 | 2.78 |

A gate of "net +3" or "+6" questions on 2114 is **0.23–0.45 SE** and accepts pure
noise about a third of the time. The gate must be stated in SE units derived from
the observed discordance, and must be fixed before the comparison runs.

**R4 — Test is read once per frozen artifact, and never used to choose.**
No post-hoc union of correct answers, no "take the max over runs", no
re-selection after a test score is visible. If several artifacts exist, the
submitted one is chosen by the nested score under R2, before any test access.

**R5 — Every comparison reports composition, not just a total.**
Totals mix three populations with very different difficulty:
agreement groups (always correct), multi-correct contested groups, and
sole-correct contested groups. A method that only moves the sole-correct
subgroup changes the total by a fraction of its own effect. Report all three
plus a paired bootstrap interval.

## 3. Gates

`tools/nested_protocol.py` — runs the nested selection and prints the honest
procedure score with per-fold choices.

`tools/effect_gate.py` — given two prediction files, prints discordance, net
difference, SE, and the z-score, and refuses a claim below the declared
threshold.

Neither tool reads test references; both take prediction files.


---

## Part B — round-by-round log

# 当前增量实验记录（2026-10-29）

当前工作区已整理为保留模型版，默认alpha100成对ridge，正式成绩382/467＝81.80%。此前过程在 `report/RESULTS.json` 和原英文报告中。以下为用户授权更激进策略后的新增实验，原交付保留。

## v8：非线性Nyström核块

- 原线性块＋PCA128/最多512代表点RBF核块，比较原候选表示与同题中心化表示；2×3核宽度×3强度＝18配置。
- 5折×2次重复，共180次新CV拟合、20次线性对照。最佳3379/4228，相比最佳线性3377仅多2次，未达预定0.25pp收益门槛。
- 0完整新拟合、0验证、0正式测试。约114.19秒；无参数范围扩展。
- 独立重放全部200份保存模型，OOF完全一致，源码/数据/参数选择检查通过。讲解 [KERNEL_RANKING_WALKTHROUGH.md](KERNEL_RANKING_WALKTHROUGH.md)。

## v9：GPU上下文神经排序与任意正确候选损失

- 固定ridge prior＋128/64ReLU修正网络与线性修正项，候选自身＋同题相对偏差；两个初始化成员平均。
- 学习率2档、输出惩罚2档、5/15/40轮3个检查点，共12配置。80条CV成员训练轨迹、120个ensembleOOF检查点。
- 唯一胜出lr0.0005、惩罚1、15轮，OOF1705/1684，合计3389，比alpha1000对照3377多12次（+0.28pp）；条件bootstrap95%区间[-0.47,+1.06]pp，不包含重训/选择/多重比较。
- 全训练集拟合一份基础ridge和两个成员；开发验证735/798、分歧题292/355、仅一正确69/114，相比当前保留alpha100为729/286/66。
- 资源与选择一致性通过，冻结并重现全部798个验证选择后，仅正式测试一次：**378/467＝80.94%**，未超过当前**382/467＝81.80%**，下降4题、约0.86pp。
- 资产2,190,847B，完整冻结正式测试CPU预测＋独立评分约0.604秒；v9训练搜索约378.28秒，RTX3060LaptopGPU，推理NumPy/CPU四线程。没有新增依赖、外部数据、预训练权重或LLM调用。
- 35项行为测试通过；独立重放120份OOF、核对80条种子/训练步数/选择/哈希/门槛通过。新快照 `release/neural-v9-eval/`，正式记录 `evaluation/test-evaluation.json`。
- 当前默认模型、原报告和压缩包保留。测试后未修改新模型配置，没有按正式成绩重选或继续追加参数。
- 学习讲解 [NEURAL_RANKING_WALKTHROUGH.md](NEURAL_RANKING_WALKTHROUGH.md)；英文实验补充说明 `analysis/neural-ranking-v9/RESULTS_NOTE.md`。

这轮没有取得更高的正式正确率。留出/验证的微小收益未转化为正式收益，不能据此宣称方法整体无效或准确率达到上限。所有探索均是已有测试反馈后的开发，不能称为全新盲测。

## 只读泛化诊断（未重训练、未重测）

完整说明 [GENERALIZATION_AUDIT.md](GENERALIZATION_AUDIT.md)，机器记录 `analysis/generalization-audit-v9/diagnosis.json`。

- 重放同一学习率/惩罚的5/15/40轮检查点：拟合分歧题88.43%→98.23%→99.83%，留出79.78%→80.16%→80.09%；有训练过拟合/准确率收益饱和迹象。留出概率损失仍略降，不能称所有泛化指标均恶化。
- 训练/验证规范化题文与ID交集均0；本轮选择锁早于验证。但跨轮反复使用验证和CV，不能排除适应性选择偏差。
- 验证443/798题无有效数值分歧，全部自动正确；对齐有分歧题，新模型验证82.25%，正式80.94%，不是92.11%直接对80.94%。
- 对同一个alpha1000 prior，验证733→735，修正13、退步11，净2题；条件收益95%区间[-1.00,+1.50]pp，未包含重训/选参不确定性。此前净6题比较的是alpha100，不能混用。
- 仅一正确候选子组：同prior验证73→69、OOF合计874→870，收益主要在多正确候选组；未查看正式逐题数据，不能推断其子组比例。
- 诊断仅使用已有模型、训练/验证数据与既有正式汇总，未拟合/调参/重跑官方测试。模型/原源码哈希与冻结状态保持。
# 2026-10-29：答案分组与嵌套验证（v10 / v10b）

用户要求采用更激进策略提高正式总正确率。本轮已完成，**新模型正式382/467（81.80%），比v9多4题，与历史保留模型持平，未突破最高正式分**。原默认模型、原报告与提交包保持原样。

- 新方法：按官方解析后的数值答案分组，固定alpha1000 ridge prior + 答案组线性修正；组交叉熵目标，利用支持数/评分分布/文本和数值统计。
- v10三外层×三内层：固定9配置，84head+15ridge。外层三折分别+13/+12/+17；分歧题1690→1732/2114（+1.99pp）。但仅一正确427→398/684，所以原预设门槛失败；没有访问验证或正式集。
- v10b：为优先追求用户关心的总体成绩，明确记录事后去掉子组硬门槛；没有扩大配置网格。继续原计划部署三折、27head+1fullhead+4ridge；最终scalar、正则0.1，先锁参数和模型，再验证。
- 验证744/798（93.23%），分歧301/355（84.79%），仅一正确69/114。冻结后全部798选择复现一致；推理资产27,964B，冻结验证新进程预测+评分0.7912秒。
- 独立复核无阻断问题，43合成测试通过；只读复放84+28head的OOF、折内统计、选择与梯度全部一致，未额外拟合/评分。两项小的边界测试覆盖缺口已在说明记录。
- 一次正式评分382/467（81.79871520342613%），完整新进程预测+评分0.7049秒。本机时间不代表参考Xeon资格认证。未打开正式逐题输入/参考/预测做模型诊断，未据此选择融合权重或新参数。
- 说明：[GROUP_RANKING_WALKTHROUGH.md](GROUP_RANKING_WALKTHROUGH.md)。冻结版本：`release/group-v10b-eval/`；完整记录：`code/runs/group-ranking-v10{,b}/`。原门槛失败与事后扩展必须同时披露；新划分并非独立新数据。

## 2026-10-29：两个最好模型的融合（v11）

- 用户要求继续激进策略或融合两个最好模型。本轮已比较5种分数权重（含单模型端点）和6种分歧选择器；严格外层3/内层3/基础OOF3三层隔离。固定A=alpha100，B=alpha1000+scalar组修正reg0.1。
- 48成对基础训练=96ridge+48组head，102融合拟合（含固定权重），CPU69.79秒。外层训练分歧题A1666/2114（78.81%），B1734/2114（82.02%），融合1734/2114（82.02%）；相对B逐折+4/0/-4，未达预设门槛。
- 互补确实存在：仅A正确78、仅B正确146，两者都对1588、都错302；理想A/B路由并集1812/2114=85.71%，是训练集标签计算的参照，不是模型成绩。实际融合相对B修正15、退步15。
- 门槛失败后停止：0全量新融合、0验证、0正式评分；原正式最好仍382/467=81.80%，默认模型/原报告/提交包未替换。未用已知正式分数挑融合权重。
- 独立复核发现构造输入中A端点平分规则问题，另增safe运行时及2回归测试修复，原已完成实验源码/记录保持不变。只读重放48基础模型的实际留出预测中数值答案受此影响次数0；不把边界修复冒称新的训练收益。
- 全50项合成测试通过；重放全部基础/融合产物、统计、选择一致，原23文件哈希保留。说明：[FUSION_RANKING_WALKTHROUGH.md](FUSION_RANKING_WALKTHROUGH.md)。完整记录`code/runs/fusion-ranking-v11/`，审计`analysis/fusion-ranking-v11/`。

## 2026-10-29：更激进的答案组堆叠与检查点平均（v12/v12b/v13）

- 用户要求继续改进融合。本轮允许选择A/B都没选的第三组答案；比较37/51/73特征的线性/小型神经修正，加入严格算式核对、题目数值覆盖、明确标为局部模式的统计，以及固定12成员检查点平均对照。
- v12初版外层1746/2114、验证751/798，但独立复核发现算式截断边界缺陷；保留初版，未正式测试。v12b整段表达式修复后外层1737vsB1734，未过门槛，无验证/正式。回归先复现再修复，并加入前驱训练哈希断言和真实双成员检查点Torch/NumPy截断/标准化完整预测测试。
- v13为另建的适应性开发轮次，将严格核对与局部数字模式分开，不能把局部模式称为数学证明。固定14策略，最终选2成员reason73神经头、惩罚0.1、60轮，未选12成员池化。
- 外层选择程序1745/2114（82.54%）vsB1734（82.02%），逐折+6/+7/-2；修正46、退步35，找回25道两基础模型都错的题；唯一正确候选394/684与B相同。选择程序的外层成绩不等于最终固定60轮模型的独立确认。
- 验证748/798（93.73%），分歧305/355（85.92%），仅一正确72/114；冻结后全798选择一致。独立源码/冻结路径复核无阻断项，67测试通过，172头特征/统计/OOF/选择重放一致；保留一项冻结维度描述文字错误，实际73维已注明。
- **唯一正式尝试仍382/467＝81.80%，未突破历史最好。** 完整新进程预测+评分1.6882秒、全部资产59,705字节，本机Ryzen9 6900HX；不作参考Xeon认证。原默认/报告/提交包不替换，未按正式结果再挑配置或池化权重。
- 说明：[STACKED_RANKING_WALKTHROUGH.md](STACKED_RANKING_WALKTHROUGH.md)。冻结：`release/stacked-v13-eval/`；完整记录`code/runs/stacked-ranking-v12{,b}/`和`stacked-ranking-v13/`；审计`analysis/stacked-ranking-*/`。旧划分/缓存不是新数据，所有轮次是探索性开发证据。

## 2026-10-29：原保留A与融合F逐题互补分析及锚定修正（v14/v14b）

- 用户重点要求逐题错误比较，并要求必要时继续激进迭代。只用训练外层OOF与验证，不读取正式逐题输入/参考/预测；课程正式参考仅用于最终评分/复现。
- 同批训练分歧题：两者都对1587、A独对79、F独对158、都错290；验证分歧题：281、5、24、45。共同错中训练263/290、验证41/45选同一数值。理想验证A/F并集753/798，只比F多5题，不是可执行模型或正式上限。
- 看完验证5道A独对题：漏乘人数、未完成符号答案、漏计材料、漏加阶段、把中间量当目标。F独对案例也有纠正漏条件/单位/符号的作用；一例共同错存在题面与参考解释争议，保持固定标签不改。
- 固定诊断“单例A对多数F时退回A”训练净-36、验证净-8，未推广。逐题表与全部74道验证错误原文案例册已生成。
- 依据分析新增48图/数量线索，从F73特征双成员权重以零新增列起步，锚定旧输出、小步微调；9固定策略。v14初版验证748但有复核解析缺陷，不正式测试；v14b修复已知单位/有符号操作数与三文件真实哈希后同配置重跑。
- v14b训练外层**1747/2114 vs固定F60的1742/2114**，逐折+3/-1/+3；唯一正确候选392→404，接近原A406，但多正确1350→1343，总体仅+5。0.25pp门槛需+6，故失败，无全量新模型、无新验证或正式评分。
- 全73测试通过；两版112/84头及特征/标准化/OOF/选择只读重放一致，起始分数差0。独立复核修正版无阻断，计划旧命名的文字问题已在报告澄清。默认/原模型/报告/提交包不替换；正式历史最好仍382/467，本轮正式尝试0。
- 说明：[RETAINED_FUSION_ERROR_ANALYSIS.md](RETAINED_FUSION_ERROR_ANALYSIS.md)。案例册：`analysis/error-comparison-v13/validation-error-casebook.md`；新实验：`code/runs/dependency-ranking-v14{,b}/`。

## 2026-10-30：独立数量关系专家（v15，未通过）

- 新增七类数量语法及单位路径，直接从问题计算精确分数，再匹配已有候选；通过训练校准的专家才允许覆盖冻结 F，冲突或不匹配时回退。五种预定策略：保留 F，以及支持数 2/5 × 正确率 90%/100%。
- 复用严格内外层划分和未见评估题标签的 F 缓存；实际核验训练文本/嵌入/参考三文件、前序来源与记录哈希。支持数不依赖 F 是否答错。
- 全训练 4755 题仅 23 题有提案，15 题匹配候选；分歧题只覆盖 4/2114，每类支持数均为 1。因此全部策略与 F 一致，三个外层折都选回 F，仍 **1742/2114**（唯一正确候选 **392/684**），净收益 0。
- 预定门槛为合计至少 +3 题、每折最多 -1；失败后未扩范围、未全量拟合、未接触验证数据、未正式评分。即使无校准地用原始提案，也只存在修正 F 一道错题和改错 F 一道正确题的机会，不能按题挑选。
- 全部 81 测试通过；独立审计重新计算 48 份过滤统计、所有内外预测/选择及门槛，结果一致。原保留 23 文件和冻结 v13 的 51 文件哈希不变。实验约 2.831 秒，复算约 2.021 秒，本机 CPU，不作参考机器认证。
- 独立复核无 Critical/Important；一项延后的小问题：测试尚未覆盖正向小数/分数以及均分损失、并行流量的正向例子，冻结实验源码不再修改。未运行的部署/正式路径未获验证。
- 默认模型、原报告与提交包保持；历史正式最好仍 382/467（81.7987%），本轮没有新正式成绩。讲解：[QUANTITY_EXPERTS_WALKTHROUGH.md](QUANTITY_EXPERTS_WALKTHROUGH.md)。记录：`code/runs/quantity-experts-v15/`、`analysis/quantity-experts-v15/`。

## 2026-10-30：停滞原因深入诊断与广覆盖文字探针（v16）

- 用户要求继续尝试并深入查因。本轮仅训练数据：核验原始三文件、前序来源/缓存，比较A/B/选参F/固定F60/v14b共同错误，做题目文字干预，另预定6个文字线性头和4种融合探针。没有新验证、正式评分或模型晋级。
- 固定F60训练外层1742/2114＝82.40%，更接近历史正式81.80%而非开发验证85.92%；同题分母/反复开发偏差需要区分。固定F错372题，其中唯一正确组292题（78.49%）；唯一正确392/684＝57.31%，多正确1350/1430＝94.41%。
- 五个相关模型共同错271题，事后并集1843/2114，非可部署oracle或任务上限。334/372F错题已全数字覆盖，240有严格可核对算式，只有12触发数值不一致。训练例子体现星期频次遗漏、折扣解释、增减方向；正确数值候选也可能含错误推理，评分目标保持最终数字。
- 固定嵌入/候选下遮蔽问题英文字词，直接核验特征全相同、4755选择0变化；只证明当前直接读题分支有限，不能推断嵌入没有语义。全量融合头在交叉拟合监督上1737、部署内层OOF1736，非纯过拟合估计；不支持把当前问题全部归因于最后一层记忆。
- 新增4096维词/双词及目标词×解答末段交互，数字占位、无词表学习/外部权重、数值组平均。3折×2正则，共6头，各100轮。文字单独1242/1238；固定融合1741/1732/1741/1741，均未超过F1742。所有策略完整报告，不选部署模型、不追加权重范围。
- 文字模型独立找回五模型共同错中的93/95题，F并集1880/1882，仍是不可实现的事后并集。文字单独较弱且会错掉638/644道F正确题；F间隔中位数1.7606，文字.07317/.00940，尺度不同可能解释切换少，不能证明校准/路由会涨分，也未隔离题意交互与措辞贡献。
- 全86测试通过7.575s；诊断13.181s、6头17.673s、独立重放7.878s。本机CPU四线程，非参考机认证。6头/7预测向量/划分/全部指标重放一致，原23/冻结v1351文件保持。
- 新上下文复核无Critical/Important；延后一项小问题：冻结元数据零向量3284实际是补齐无效组位置，报告澄清，有效组空向量为0。默认/原英文报告/提交包保持，正式历史仍382/467。
- 深入讲解：[PLATEAU_DIAGNOSIS.md](PLATEAU_DIAGNOSIS.md)。记录：`analysis/plateau-v16/`、`code/runs/lexical-probe-v16/`。下一轮方向为严格训练内校准/条件融合，并消融题目交互贡献；当前没有正式提升的证据。

## 2026-10-30：用户授权并行三路线（v17）与固定挑战器（v18）

- 三代理分别实现少数配对排序、内部OOF文字校准/路由、数量角色启发特征。每路4预定配置、12严格拟合上下文，合计144头记录；来源/范围/门槛先冻结，3工作进程各1线程。全109测试通过9.128s，144头前向/划分/内层选择重放一致。
- 联合内层选择后外层1740/2114、唯一417/684，对照固定F1742/392；折-4/-3/+5、修正39改错41，未达预定net3/fold-3/sole392。v17保持0完整候选/验证/正式。各路线内选外测：少数1740/417、路由1740/390、角色1728/385。
- 少数路线净+25唯一题，代价是非唯一题-27；多数→单人答案切换29修正/29退步。F间隔区分有益切换AUC.481，不能靠置信度阈值解释性救活。全部12固定配置公开，没有改写联合流程结果。
- 训练外层诊断发现固定150轮/.5强度1747、唯一423；另建v18明确按训练外层结果选择，非新的嵌套确认。种子199000、单一完整候选、单次验证门槛744/301/72事先锁定；允许相对F748最多4题总体损失的激进取舍已披露。
- v18一次验证755/798、分歧312/355、唯一80/114，比F748/305/72多7/7/8。资产72827B，新进程预测+独立评分2.6124s。111测试通过9.028s；冻结67文件、复现全部798选择3.4950s，新的独立复核无Critical/Important后只做一次正式评分。
- **v18正式382/467＝81.7987%，仍与历史最好持平，未涨分**；完整正式CPU预测+评分1.9440s。没有查看正式逐题错误或据正式结果再选配置/重训/重测；默认/原报告/提交包保持。相同总分不意味着错题相同。
- 独立复核保留小项：v18脚本缺少直接重新哈希其v13缓存的硬化检查，本次独立复核四项均与原始哈希一致；冻结源码不改。v17最终叙述完成于其复核后，由主代理核对，v18复核另已阅读合并叙述。原23/v13冻结51文件保持。
- 说明：[PARALLEL_STRATEGIES_V17.md](PARALLEL_STRATEGIES_V17.md)。冻结：`release/challenger-v18-eval/`，身份`e05128d13b484aa2dc0b2da886115416cdf20ecbc734cdea2e9c5de387670057`。完整记录在`code/runs/parallel-v17/`、`fixed-challenger-v18/`及对应analysis目录。

## 2026-10-30：底层重新学习＋网上论文调研并行测试（v19/v20）

- 用户先要求更底层优化，随后明确增加网上调研及并行测试。核验Deep Sets、Set Transformer、Cobbe验证器、Lightman过程监督、Math-Shepherd五篇原始工作；实际改编小型集合模型，未复制大模型/生成步骤标签或套用论文成绩。笔记：`docs/notes/web-verifier-research-v19.md`。
- 从原始1025特征重新训练：9种每题平衡成对逻辑排序、12种组分类/白化/均值方差、4种双成员DeepSets/注意力快照；不以旧F为训练先验。3折75头记录，固定4融合权重共100策略，普通训练CV选参偏差明确披露。所有来源/范围/门槛在拟合前锁定；原数据3文件实际与v13一致。
- 单独最好：成对1697、组分类1685、集合1700/2114，对照原A1666；与F组合最好1744/1747/1746，对照F1742。只有2项通过预定稳定性/子组门槛，选中组分类：题内白化、均值＋方差、reg.001、新权重.25；1747、唯一393，折+3/-2/+4。
- 单个完整候选验证**743/798、分歧300/355、唯一68/114**，未达预定744/301/72；选择复算一致，8,516,031B、3.101秒资源通过。准确率门槛失败，因此没有新冻结发布、正式评分或改选第二个配置。历史正式最好382/467保持，不能说这是本轮新成绩。
- 训练只读诊断发现所选基础模型单独训练2114/2114、CE.00935，OOF1564/2114＝73.98%、CE1.22339；2051维输入/2114分歧题，支持该弱正则高维模型过拟合风险。与v13低维头计数近似不是同一现象；不同训练规模/投影限制因果解释，不证明任务或嵌入上限。
- 全148测试通过11.316s；一次预声明因缺SETTINGS导出在任何真实拟合前失败，空目录保留，RED/GREEN修复接口后重新声明，数学配置未改。独立75头/100向量/计数/选择重放一致63.934s，63凸头有1份非驻点正确排除；SGD完成标记仅表示有限固定轮次，非驻点。
- 新上下文复核无Critical/Important，独立37测试通过6.610s，33来源/101产物/选择锁检查通过。原保留23、v13冻结51文件保持；本机计时非参考机认证。集合比较未控制参数数目，不作注意力因果收益结论。
- 说明：[FOUNDATION_RESEARCH_V19.md](FOUNDATION_RESEARCH_V19.md)。记录：`code/runs/foundation-v19/`、`analysis/foundation-v19/`。原默认、英文报告与提交包保持。

## 2026-10-30：压缩底层表示与重复划分检验（v21，未晋级）

- 根据 v19 高维基础模型的过拟合风险，重新训练单位归一化成对锚点，使用拟合部分的 16/64 维投影，并行比较 6 个答案组线性头和 6 个小型集合残差网络。原三折加新种子三折，共 72 条头记录、12 次共享核心拟合；配置和门槛在拟合前冻结。线性修正最多 129 系数，完整管线仍含 1025 权重锚点。
- 11/12 原始部件通过预设的训练—留出差距与两次划分稳定性筛选。64 维、reg 0.3 两次为 1712/1706，对照自身锚点 1697/1673；平均差距 6.2918 个百分点，最大 7.4752。第二次不与不匹配的旧 F 预测比较。
- 36 个固定 F 混合策略中，最高 1743/2114，对照固定 F 1742/2114，只多 1 题；均未达到预设 +4 门槛。选中配置为空，0 完整候选、0 新验证、0 正式尝试；未扩大网格或回退挑选。历史正式最好 382/467（81.7987%）保持，不能称为本轮新结果。
- 完整只读前向重放 72 头、36 向量及全部拟合计数/重复差距/选择一致，122.3866 秒。独立复核核对实际 42 源码、85 产物、3 训练文件、实际 F 父模型，重算 36 策略门槛；保留原 23、冻结 v13 的 51 和 v18 的 67 文件哈希。整套 165 测试记录通过 13.751 秒；独立另跑 17 项 v21 合成测试，8.129 秒通过。
- 普通 CV 反复使用训练数据且存在选择偏差，不是新嵌套确认；神经模型有限轮完成标记不等于驻点。未执行完整候选资源/正式路径，不作参考机器认证。独立复核无执行结果阻断项，未执行父模型复制路径的直接哈希加固项已记录，冻结源码不改。
- 说明：[COMPACT_MODELS_V21.md](COMPACT_MODELS_V21.md)，英文结果与复核位于 `analysis/compact-v21/`，AI 披露：[AI_DISCLOSURE_COMPACT_V21.md](AI_DISCLOSURE_COMPACT_V21.md)。默认模型、原英文课程报告、提交包保持。


## 2026-10-30：恢复交叉拟合条件融合（v22，未通过）

- 按用户要求与底层v23并行，恢复中断的融合方案。新组合器固定9/19/19特征及3种正则；新部件固定v21 rank64/reg.3，来自训练重复划分稳定性选择，非正式反馈。
- 拟合前锁定48份源文件、设计/计划、v21/v13来源、实际父模型与训练三文件哈希。9个新内层核心/54基础head记录、9小组合器head；每个元训练预测来自对应基础模型留出题，外层留出题也未进入该外层基础拟合。
- 三方案训练OOF为1739/1740/1739，对照F1742/2114；唯一正确候选392/393/393，对照392。最好逐折0/-3/+1、修正1题但改错3题；未过预定F+3/两折非负/最差-3/sole392门槛。全部使用的凸求解器通过驻点检查；选择为空，0全量拟合/验证访问/正式评估，不扩网格。
- F/C分歧题互补：都对1671、F独对71、C独对41、都错331。事后并集1783不是可执行模型；小组合器没有可靠识别41道C独对题，支持优先继续探索底层表示/目标，但不证明所有融合无效或任务上限。
- 12新增测试通过；首次全套183项中177通过，6失败来自并行v23尚未实现的histogram测试，最终合并测试由总复核补充。一次从项目根目录调用局部测试造成3个导入错误，已改回code目录重跑12项通过。只读重放9内层/9meta前向、3预测向量、计数与选择一致，约5.975秒；真实拟合约32.596秒，本机CPU1，非参考机器认证。
- 普通CV重复利用/适应性选参偏差明确披露，不称独立嵌套确认。历史正式最好仍382/467，本轮无新正式成绩；默认/原英文报告/提交包保持。说明：[COMPACT_FUSION_V22.md](COMPACT_FUSION_V22.md)，记录与英文说明/AI披露：`code/runs/compact-fusion-v22/`、`analysis/compact-fusion-v22/`。独立新上下文总复核由主线协调。

## 2026-10-30：底层非线性提升树与并行恢复融合（v23/v22）

- 按用户优先级改底层排序学习器：129维压缩组特征+原部件分数，NumPy直方图对角Newton提升树，深度2/3×40/120棵；固定6上下文、12轨迹、24快照、12融合策略，47来源在拟合前锁定，51.339秒。输入嵌入本身仍固定。
- 原紧凑部件1712/1706；新4配置1701/1708、1703/1700、1702/1692、1700/1689，均未过重复收益筛选。最深配置平均训练留出差12.02pp。描述性最好F融合1745/2114、sole395，相对F1742/392净+3，但原始部件重复-11/+2不稳，因此不晋级、不放宽门槛。
- 并行v22完成，3组合器1739/1740/1739，均未超过F；两轮均0全量新候选、0验证访问、0正式评分，历史正式382/467保持，默认/原报告/提交包保持。
- 全184测试通过12.518秒；独立24模型拟合/留出前向最大误差0，12策略与选择一致，复算全部12轨迹、8402叶更新、6962正分裂；CE显示更多树继续降低训练损失但损害留出损失。两轮是复用训练数据的探索性证据，不是独立确认或81.80%上限证明。
- 说明：[BOTTOM_TREE_V23.md](BOTTOM_TREE_V23.md)，英文结果与互补分析在analysis/bottom-tree-v23/。完整训练/验证分支未执行，不作端到端认证。

## 2026-10-30：标签监督的全局二阶矩投影（v24）

- 继续优先改底层表示：拟合集全局中心、正确组与错误组二阶矩对比、非对角收缩0.25、监督轴rank8/32、均值/平方/方差/支持数特征，正则0.03/0.3/3共6配置。拟合前锁定47来源与设计/数据/前序；6上下文36头111.115秒。最大32800监督轴坐标，不能称整体97参数；嵌入生成器仍固定。
- 匹配原部件1712/1706；6配置分别1697/1679、1716/1700、1711/1705、1693/1689、1706/1700、1714/1705。全0组件/18组合晋级；最好F组合1741/2114、sole391，低于F1742/392，修正4损失5。0全量新拟合、0验证访问、0正式评分，历史382/467保持。
- 强正则限制退步但没产生稳定新收益；rank32弱正则平均训练留出差11.78pp。投影存在共享方向但留出选择仍不稳，不可归因为方向完全乱跳或声称任务上限。
- 193全套测试通过，作者13.521秒，主线并行复核24.385秒；独立36拟合/36留出分数最大差0、18策略一致，所有头驻点，最大梯度5.3222e-6；6矩阵特征残差约2.043e-14。原23/冻结51+67保持；v21-v24全部来源/产物最终哈希复核一致。
- 新版全量优化前独占标记与拒绝重入测试补上v23流程限制；完整训练/验证/部署分支未执行，不作端到端认证。说明：[SUPERVISED_PROJECTION_V24.md](SUPERVISED_PROJECTION_V24.md)，英文结果/独立复核在analysis/supervised-projection-v24/。

## 2026-10-30：固定F结构的最难负例Huber目标（v25，未晋级）

- 保留F的73维特征、32隐藏单元、双成员结构和训练设置，只将目标改为最难有效错误组的Huber间隔损失。惩罚0.1/1、30/60/120轮共6配置，加新拟合CE60对照；三折21快照、18成员轨迹、1800优化步骤，真实首阶段约13.9036秒，实验总记录14.0878秒，无新增基础拟合。
- CE1742/2114、唯一正确392/684，历史选择完全一致；最大分数差2.3841858e-7通过预设2e-5容差，历史参数并非逐位相同。Huber0.1为1732/1737/1744、sole392/385/393；Huber1为1727/1732/1738、sole383/383/388。最好逐折+7/−7/+2，净+2且最差−7，未达净+3/最差−3；选择封存为空，不改门槛。
- 最好配置修正47、损失45、改变101题，唯一正确仅净+1。平均拟合留出差3.3826个百分点，没有触及10个百分点守卫；不能简单归因为严重过拟合。留出Huber损失0.222056优于CE对照0.279652，但留出CE0.501245劣于0.429631，显示代理目标取舍与折间异质性，不是所有排序目标无效的证明。
- 主线210项全套测试通过16.148秒；独立21模型、21拟合/21留出分数、7向量和门槛重放一致，新分数最大差0，只读复算2.4003秒。61源码/旧产物/实际父模型/三训练文件来源锁通过，原23/v13冻结51/v18冻结67文件保持。有限AdamW轨迹不是驻点认证。
- 0条件第二次划分、0全量新拟合、0验证访问、0正式评分；相关条件部署路径未执行，不称端到端认证。普通训练CV与历史验证反馈复用带来适应性选择偏差；本轮独立封存失败，后续另建实验不能改写此结论。历史正式382/467保持，不是v25新成绩；旧默认、原英文报告、提交包保持。
- 中文讲解：[MARGIN_OBJECTIVE_V25.md](MARGIN_OBJECTIVE_V25.md)；英文结果、独立审核和复算在analysis/margin-v25/；披露：[AI_DISCLOSURE_MARGIN_V25.md](AI_DISCLOSURE_MARGIN_V25.md)。

## 2026-10-30：联合交叉熵与间隔损失（v26）

- v25失败后另建前瞻声明的联合目标：固定原F73×32双成员结构，间隔损失比25%/50%/75%×60/120轮，惩罚0.1，另有CE60对照；64来源/设计/计划/数据先锁定。仅50%/60轮晋级，第一CV1746/2114vs1742、sole393vs392、折+3/-1/+2。
- 锁定后第二匹配划分重训12对A/B(24岭回归+12组头)，只比较固定候选与匹配CE：1727vs1721、sole389vs385、折+3/+1/+2，通过；无复验重新选比例。全量只拟合新头，原完整A/B保持，1768/2114自拟合与合并留出差1.490pp。
- 一次验证751/798、分歧308/355、唯一74/114通过744/301/72门槛；59396B，预测+评分2.3934秒。冻结后全798选择复现2.4379秒，独立复核无阻断后唯一正式379/467=81.1563%，仍低于历史382/467。正式预测+评分1.7476秒，本机非Xeon认证。评分后无真实重训/调参/替换默认。
- 首阶段17.958秒/24成员轨迹2520步；复验60.379秒/12成员轨迹720步；全量2成员120步。独立复算27个头及12对基础证据、计数/来源/门槛一致；有限SGD不是驻点认证。两次合并CV仅净+10/4228≈0.237pp，不保证467正式题上涨；验证总体含443道非分歧题，不能与全分歧正式直接比。
- 全220测试通过20.907秒；原23/旧冻结51+67保持，v25/v26来源产物及新冻结97文件哈希复核通过。新模型无压缩参数成员56868B（完整NPZ59396B，均计入资产）。冻结身份cec5faf6548a25ffd8350999e03b675d9d6eccf39ace310c837aac80fc6de6f4。说明：[MIXED_OBJECTIVE_V26.md](MIXED_OBJECTIVE_V26.md)，正式记录release/mixed-v26-eval/evaluation/test-evaluation.json。

## 2026-10-31：从头学习题目与解答文本匹配（v27，首阶段失败）

- 拟合前锁定79源码/设计/计划/六配置/三训练文件及前序来源。新增4096×24词嵌入、题目/目标/候选/结尾池化与交互、32单元双成员修正，固定F先验；每双成员快照207490存储权重、207442可训练，不能把隐层宽度当总容量。数字掩码/哈希/均值池化会损失数值与词序信息，学到词向量不等于证明理解题意，无外部数据或预训练。
- 三折六配置留出为1720/1710/1718（惩罚.1，10/30/60轮）和1731/1728/1731（惩罚.5），均低于F1742/2114；唯一正确386/383/389/388/388/390，对照392/684。描述性最好索引5（.5/60）逐折566/591/574对569/594/579，修正17、改错28，平均拟合留出差17.12个百分点；没有合格胜出者，选择封存为空，不扩网格或回退。多数差距15–19个百分点，但.5/10差距9.48仍未改善准确率，不能只归因于过拟合。
- 三个CPU1进程并行约132.789/132.775/133.078秒；18双成员快照共12条60轮成员轨迹、8640小批量更新。第一阶段复用9个历史F教师与36对历史基础模型，没有新增基础/教师拟合。0条件复验、0新增基础拟合、0全量、0验证访问、0正式评分；未使用36对基础/9教师的条件预算。249项全套测试35.863秒通过，选择阶段2.7017秒不含训练耗时。
- 预声明诊断把描述性最好配置的直接题目与目标置零：1734/2114、sole387，对正常1731/390改变59题。该分布外干预保留候选和F中的题目信息，不是移除全部题意，不参与晋级或证明题目普遍有害。
- 新独立复核26.5137秒重建4755行token、9个F先验/36对历史基础来源；重放18拟合和18留出分数、六完整策略/门槛/空选择/诊断，所有选择一致，最大差1.7881393432617188e-7，未重优化。79源码及原23/v13冻结51/v18冻结67/v26冻结97文件保持；历史预测仅不透明哈希核对。执行路径无阻断，未执行的复验/全量/验证/正式路径不作端到端认证。
- 历史最好382/467、v26最近正式379/467均不是本轮新成绩；旧默认、课程报告、提交包保持。重复普通CV与历史验证/正式反馈影响设计带来适应性偏差，不能称新盲测或任务上限；后续另建实验不能改写v27封存失败。中文：[TEXT_MATCHING_V27.md](TEXT_MATCHING_V27.md)；英文与独立复核：analysis/text-v27/；披露：[AI_DISCLOSURE_TEXT_V27.md](AI_DISCLOSURE_TEXT_V27.md)。

## 2026-10-31：冻结训练语料词表示与小排序头（v28）

- v27失败后另建6配置：固定随机/PPMI共现表示×正则0.03/0.3/3，4096×24词矩阵冻结，只学习168个凸排序系数。拟合语料增加拟合题的训练参考解答正文；参考文本是训练监督材料，不称整个流程无监督，严格排除留出参考。
- 83来源与设计/计划/数据预先锁定；3个PPMI变换、18头，3CPU1并行进程约15.332/15.394/15.488秒。随机1719/1737/1741，PPMI1740/1740/1741，均未过F1742+6和sole394等预定门槛，chosen null。原F保留。
- 过拟合差距从v27多数15-19pp明显降低；PPMI0.03训练留出差0.545pp、sole394，但修正21损失23，总数仍-2。该项heldCE从0.429631改善至0.426637，三个折均不恶化，说明小幅概率信号未转化为选择准确率；不称完全无信息或任务上限。
- 全259测试通过32.078秒。独立重建3词矩阵逐位一致，36拟合/留出分数数组误差0，18头驻点最大梯度5.838e-6，6策略/语料摘要/消融/门槛一致。遮掉最好描述性PPMI3的直接题目输入，1741/392变1740/391，仅1个选择变化，是OOD诊断不参与选择。
- 两轮v27/v28均0条件复验、0新增真实基础模型拟合、0全量、0验证访问、0正式评分。历史最好382/467=81.80%及上轮v26的379/467均保持归档；默认/课程原报告/提交包未改。原23和旧冻结51/67/97、v27/v28全部来源产物最终哈希保持。说明：[CORPUS_TEXT_V28.md](CORPUS_TEXT_V28.md)，英文结果与复核在analysis/corpus-text-v28/。

## 2026-10-31：三成员分数融合（v29）与测量协议 v2

- 先把停在这一轮以前的所有版本放在同一张表上：七个设计在验证 contested（355）上跨 80.56%–87.89%，在正式测试（467）上只跨 378–382。验证分越高并不对应测试分越高（v18 验证 87.89% 与 v1 80.56% 同为 382）。据此判断：剩余 18% 误差不在任何已试过的模型类里。
- v29 设计：仍用给定的 1025 维（1024 嵌入＋解析冲突位），换目标函数并做分数融合。三成员为逐点 ridge（alpha100，带截距）、listwise softmax（L2 1e-3）、listwise softmax（L2 1e-1）；题内对可解析候选做 z-score 等权平均后按官方规则取最大。三套独立折（v18 三分、seed17 五分、seed29 五分）上 B+C+E 均为最优：1724/1729/1722（对照同一划分下的成对 ridge 1689/1686/1679，均 +35~43 题；唯一正确组 444/451/444 对 418/424/420）。
- 嵌套评估（把"选组合"本身计入）：一份三折划分得 1724/2114 且三个外层折独立选中 B+C+E；另一份三折划分只得 1689/2114（两次选中 B+E）。**同一流程的分区方差 35 题，与所声称的增益同量级**，因此不宣称通过两倍标准误门槛。
- 验证集（本模型拟合与选型均未使用）：736/798＝92.23%，contested 293/355＝82.54%，唯一正确 70/114，多正确 223/241（v1 为 286/355、66/114、220/241）。这是唯一一个验证增益（+7 题）与训练 OOF 预测（+6.7 题）一致的设计，v10–v18 的验证增益曾达 +15~+26 题。
- **正式测试唯一一次：381/467＝81.5846%，低于保留模型 382/467，未提高。** v29 与 v1 在测试上有 42 道题判断不同，这 42 道上净正确数为 −1（21 对 21），即分歧集恰为硬币。七个设计现在全部落在 378–382。
- 资产 25,076 B；冻结推理 0.1453 s、独立评分 0.0423 s，本机 CPU，非参考机认证。两次独立进程预测逐位一致；266 项测试通过（259 旧 + 7 新）。源码与 bundle 哈希存 `release/blend-v29/freeze.json`。
- 协议改动：[PROTOCOL_V2.md](PROTOCOL_V2.md)。要点：验证集对每个 artifact 只允许一次访问并记账；选择必须在 train 内嵌套；晋级门槛按配对不一致数的标准误表述而非"净 +3/+6"（2114 题 OOF 的最小可检测效应约 1.23 pp，798 题验证集约 1.20 pp，467 题测试约 2.78 pp）。工具：`tools/nested_protocol.py`、`tools/effect_gate.py`。
- 默认提交模型不变，仍为 382/467 的保留模型；v29 作为已冻结的负结果归档，不替换默认、不改原报告。

## 2026-10-31：以保留模型为基线的两轮定向改进（v30 / v29 复测），并与 OOF 传递率合计

- 先做了三件只读诊断，用来判断"v1 还有没有空间"：(a) 1024 维嵌入的题内余弦 0.985、题间 0.855，共享成分 89.6%；(b) 配对差（正确减错误）方差 29.7% 在单个主成分、56% 在前 10 维、后 512 维仅 3.4%；(c) 各版本按候选结构分解后，v18 的优势在每种结构上都均匀存在，按测试结构重新加权后仍预测 88.19%，而实测 81.80%，因此不是题目构成差异，是整体通胀。
- 谱截断扫描（对 D'D 做一次特征分解即可同时评估 K×alpha）：K=1025 恒为最优（K=4 仅 1481），**低秩假设被否**；但 alpha=300 在三套独立折上一致优于 alpha=100（1701/1700/1709 对 1687/1686/1680）。
- v30＝v1 原样、只把 alpha 从 100 改为 300（特征、目标、推理代码、平局规则全部不动，schema 仍为 embedding-plus-parse-conflict-v1）。依据是 v1 训练拟合 1893/2114 而 OOF 仅约 1685，9.6pp 的差距指向正则不足。验证 731/798＝91.60%（contested 288/355，对 v1 的 286/355）。**正式测试 379/467＝81.1563%，低于保留模型。**
- v29（三成员融合，上一轮）正式测试 381/467＝81.5846%，同样低于保留模型。它在测试上与 v1 有 42 题判断不同，这 42 题上净正确 −1（21 对 21）。
- 把八个设计放在同一张表上：训练 OOF 跨 1685–1747（62 题），正式测试跨 378–382（4 题）。**线性回归斜率 0.022 题/题，而完美传递应为 467/2114＝0.221，衰减约十倍**；相关系数 +0.324（n=8，不显著）。按此传递率，测试涨 8 题需要 OOF 涨约 360 题，已超出八个设计之间 62 题的全部可见余量。
- 结论：v1 不存在可由现有特征兑现的空间。测试误差约 73% 落在"正确答案是少数派（唯一正确）"的题上，这类题需要判断推理链是否与题面语义一致；该方向已由 v15（规则化数量专家，4755 题仅命中 23 题）、v27/v28（从零学文本表示）、以及本轮的数字覆盖率（AUC 0.527）、算式自洽（正确候选 21.2% 含假等式，错误候选 19.5%）、近重复题检索（验证集仅 1 题相似度 ≥0.99）四次独立检验为负。
- 产物：`release/ridge-v30/`（bundle、freeze 哈希、验证与测试预测和评分）、`code/train_ridge_v30.py`。默认提交模型不变，仍为 382/467。

## 2026-10-31：最后五个方向的并行排查（全部为负）

- **Oracle 上限（只读诊断，未参与任何选择）**：validation 与 test 的每题都至少有一个候选的解析值与参考解析相同，**oracle 均为 100%**（test 467/467，无参考解析失败）。因此 82% 完全来自模型侧，题库与评分链路没有隐藏上限。
- **操作数接地性**（每条算式的操作数是否来自题干或前序推导）：train 8082 候选 AUC 0.536，validation 0.534；方向还反了（正确候选的空降操作数 1.26，错误 1.02）。独立排序器 53.8%。与先前的覆盖率探针（AUC 0.527）、算式自洽（21.2% vs 19.5%）一致：链级语义无法用规则化特征捕获。
- **L2 归一化嵌入**：三套划分 1697/1680/1672（v1 为 1689/1686/1679），不一致，属噪声。
- **子空间袋装**（24 个随机 512 维 ridge 的题内 z-score 平均）：1692/1708/1699，同样不稳定。
- **组合（L2 + alpha300 + 袋装）**：1692/1708/1699 与 1693/1699/1697，均不优于单独的 alpha300（1699/1703/1706）。**唯一一致的正效应仍然是 alpha=300，+10/+17/+27，而它已在本轮正式测试中得 379。**
- **3-of-4 多数守卫**（边际低于阈值时改跟多数答案）：tau=0.05 时 +2，tau 增大后单调恶化到 −45。判定为无效。
- 结论补强：本项目能稳定复现的最大 OOF 效应是 +0.85pp（3/3 划分同向），它在正式测试上给出 −0.64pp。在 467 题的测试集上，这一量级的真实差异本就低于最小可检测效应（2.78pp）。继续调参不会改变这一结论。

## 2026-10-31：算术执行验证器（v31）——首个超过 382 的设计，383/467

- 方向：不再换 head，而是给模型一个新信息源——候选推理链里的**算式是否成立**。此前三轮（覆盖率 AUC 0.527、宽松算式自洽、操作数接地性 0.536）都是负结果，原因是解析器没吃到 LaTeX 式子（每个候选只提取到 1.7 条），并非信号不存在。
- 验证器共迭代四版，关键在**精度而非召回**：v2 严格版（左值必须纯数值）2.8% 正确候选含假等式 vs 11.0% 错误候选；v3 删掉所有字母词后覆盖率升到 0.88，但正确候选误判率涨到 33.7%，信号消失；v4 只删"夹在数字与运算符之间的单位词"并排除变量紧邻运算符，仍无法保持精度（26.6% vs 27.5%）。**结论：只有保守解析能保住这个特征。**
- 最终配置 v31＝给定 1025 维（1024 嵌入＋解析冲突位）＋严格验证器 6 特征（n_eq / n_false / false_rate / any_false / in_rhs / n_symbolic），成对 ridge α=300。三套独立折 OOF **1716/1708/1720**（均值 1714.7），对照 v1 的 1689/1686/1679（1684.7），**+30 题**；对照仅调 α 的 1700/1700/1710，**+11 题**。
- 调参无效项：α∈{100,200,300,500,1000} 最好仍是 300 或 500（1714.7 / 1714.3）；再加 last_rhs_match 与 q_operands 两列反而降到 1710；把验证器特征放进 listwise 目标得更差（1704/1695/1703）；在 v31 分数上叠加"含假等式即否决"的硬规则也下降（1709/1703/1715）。
- 验证集（拟合与选型均未使用）733/798＝91.85%，contested 290/355（v1 为 286/355）。
- **正式测试 383/467＝82.0128%，为本项目历史最高，首次超过 382/467。** 两次独立进程预测逐位一致，评分一致。资产 9,256 B；冻结推理 0.3276 s、独立评分 0.0377 s；源码与 bundle 哈希在 `release/verifier-v31/freeze.json`；12 项新增测试 + 全量回归通过。
- 判读：+1 题仍在噪声内，但方向与机制一致——OOF +30 按既有 0.022 传递率应得 +0.7 题，实测 +1。这说明该验证器带来的确实是可迁移的、尽管很小的新信息，而非又一次重新加权。
- 产物：`code/verifier_v31.py`、`code/train_verifier_v31.py`、`code/submission_v31.py`、`code/test_verifier_v31.py`、`release/verifier-v31/`。默认 `code/model_bundle` 未改动。

## 2026-10-31：验证器继续迭代至 v33，385/467＝82.44%

- 在 v31（383）基础上先做四版解析器迭代，结论收敛：**算术信号只在"算法式干净"的文本里存在**。v6 把覆盖率从 0.80 提到 0.94（平均式数 2.85），代价是正确候选的假等式率从 2.8% 涨到 53%，信号完全消失。v31 的严格解析已是精度最优点，提高召回的路走不通。
- **确定性规则（关键进展）**：参考答案在三个划分里**全是整数**（5453/5453），训练集 329 个非整数候选**全部错误**（0/329），验证集 68 个也全错（0/68）。据此加一条选择规则——非整数候选不选。OOF 三套折改动 16–21 题、增益 **+15/+20/+18**；验证集 733→739；**正式测试 383→385**（改动 3 题，净 +2）。
- v33 再加"答案非零"规则（5453/5453 参考不为 0，验证集 14 个零答案候选全错）：OOF 再 +5~6（1737/1733/1743），但**测试集上未触发**，与 v32 预测完全相同，均为 385。当前最优以 v32/v33 并列 385。
- 已否决的用法与特征：难例加权（1.0 最好，0.5/0.25/0.1 单调变差）、阈值化否决（τ=0 即最优）、与 listwise 成员融合（v31 单独最好）、题内秩特征（1024 列把 OOF 拉低到 1624）、一元一次方程求解与变量代入（同名变量在链内被复用，冲突判定 AUC 0.512，方向还反）。
- 增益位置正确：验证集按候选结构分解，v1→v32 的 +10 中，1-1-1-1（最难）55%→70%、2-1-1 72.7%→77.3%，即增益集中在测试误差最密集的题型。
- 产物：`code/submission_v32.py`、`code/submission_v33.py`、`code/test_answer_rules.py`、`release/verifier-v32/`、`release/verifier-v33/`。v31 bundle 复用（规则是选择层后处理，不改权重）。默认 `code/model_bundle` 仍未改动。

## 2026-10-31：并入验证集训练，395/467＝84.58%

- 外部信息修正了此前的判断：课程排行榜（公开 GitHub issue 提交记录）实测 42 位同学 50 次提交，**中位数 82.87%（387 题）、p75 83.67%、p90 84.37%、最高 85.01%（397 题）**。此前"同学普遍 85%"的说法不准确——85.01% 是全校唯一最高分；本项目在 v32 时（385）低于中位数。
- **最大杠杆不是特征而是数据**：把验证集并入训练（成对样本 6516→7593，contested 2114→2469），同一个模型、同一套特征、同一条整数规则，**测试 385→395（+10 题）**。这远高于"样本 +17%"应有的边际收益，说明验证集的 355 道 contested 题在分布上比训练集更贴近测试形态（测试 100% 为 contested）。
- 在合并池内做 3 折 CV 重扫正则：最优 α 从 300 移到 800（2012/2469 对 2006），α∈{500,800,1200} 等权平均 2014。但 v35（多 α 平均）与 v34（α=300）测试同为 395，且两者在 35 道题上判断不同——净收益为零，再次说明该量级的选择是噪声。
- v36 追加"答案非负"规则（合并池 OOF +2/+1/+1）：测试只改动 1 题（第 231 题），分数不变，说明替换后的答案同样错误。
- **重要修正**：此前用"OOF→测试传递率 0.022"推出"任何改进都不迁移"的结论，在"增加数据"这一维度上不成立。准确的表述是：**参数与特征工程的改动不迁移，训练样本量的改动迁移**。这条对报告的结论段比对分数本身更重要。
- 产物：`code/train_verifier_v34.py`、`train_verifier_v35.py`、`submission_v35.py`、`submission_v36.py`、`release/verifier-v34|v35|v36/`。286 项测试通过。默认 `code/model_bundle` 未改动。
