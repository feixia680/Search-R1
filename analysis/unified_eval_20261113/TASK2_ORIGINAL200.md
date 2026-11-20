# 原200题：两模型统一评测与搜索预算对照

仅报告原来用于验证和 checkpoint 选择的 200 道题。模型均为 step-500 checkpoint；没有训练。

两个模型共用同一评测执行器、检索服务、tokenizer、greedy 解码和上下文限制。结果属于原验证集复测。

## 结果

| 策略 | 模型 | EM | 检索/题 | 证据覆盖率 | 第二跳有新增证据 | 第二跳平均覆盖增量 | 重复查询（轨迹/事件） | 缺答案（题数/率） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_one | baseline | 52/200 (26.0%) | 1.000 | 60.09% | 0/0 | 0.00% | 0/0 | 0/200 (0.0%) |
| fixed_two | baseline | 58/200 (29.0%) | 2.000 | 64.34% | 17/200 | 4.25% | 131/131 | 0/200 (0.0%) |
| autonomous | baseline | 54/200 (27.0%) | 1.000 | 59.17% | 0/0 | 0.00% | 0/0 | 0/200 (0.0%) |
| fixed_one | evidence | 46/200 (23.0%) | 1.000 | 55.29% | 0/0 | 0.00% | 0/0 | 0/200 (0.0%) |
| fixed_two | evidence | 60/200 (30.0%) | 2.000 | 76.99% | 85/200 | 21.70% | 9/9 | 0/200 (0.0%) |
| autonomous | evidence | 79/200 (39.5%) | 1.925 | 78.43% | 104/185 | 26.29% | 3/3 | 16/200 (8.0%) |

## 配对比较

下表为 evidence 减 baseline；CI 为按问题配对 bootstrap 10,000 次。

| 题目集/策略 | EM差值 | 95% CI | 相对EM变化 | 错→对 | 对→错 |
|---|---:|---:|---:|---:|---:|
| original200/fixed_one | -3.0 pp | [-8.0, +1.5] pp | -11.5% | 9 | 15 |
| original200/fixed_two | +1.0 pp | [-5.0, +6.5] pp | +3.4% | 18 | 16 |
| original200/autonomous | +12.5 pp | [+6.0, +19.5] pp | +46.3% | 37 | 12 |

## 第二次检索的答题收益

同模型的固定两次减固定一次，共享第一跳查询和返回内容；第二跳后重新作答。

| 题目集/模型 | EM差值 | 95% CI | 错→对 | 对→错 | 新证据且纠正答案 |
|---|---:|---:|---:|---:|---:|
| original200/baseline/fixed_two_minus_one | +3.0 pp | [-1.0, +7.0] pp | 11 | 5 | 6.0 |
| original200/evidence/fixed_two_minus_one | +7.0 pp | [+2.0, +12.5] pp | 22 | 8 | 17.0 |

## 第二跳的证据收益

“有新增证据”指第二跳的 gold 支持事实覆盖增量大于 0；“平均覆盖增量”按全部 200 题计算，因此未执行第二跳的题记为 0。重复查询列为出现至少一次重复的轨迹数 / 重复查询事件总数。缺答案按 final_answer 为空统计。

| 模型/策略 | 第二跳调用 | 有新增证据 | 零新增证据 | 平均覆盖增量 | 第一跳已全覆盖仍继续 |
|---|---:|---:|---:|---:|---:|
| baseline/fixed_two | 200/200 | 17 | 183 | 4.25% | 56 |
| baseline/autonomous | 0/200 | 0 | 0 | 0.00% | 0 |
| evidence/fixed_two | 200/200 | 85 | 115 | 21.70% | 40 |
| evidence/autonomous | 185/200 | 104 | 81 | 26.29% | 24 |

## 解释范围

固定一次/两次使用外部动作前缀约束，两个模型的协议完全相同；自主策略直接使用模型原生动作。模型在相同固定预算下的差异，可以检验收益是否超出调用数变化；自主与固定策略的差异也包含前缀约束的影响，不能全部归因于预算。

固定两次与固定一次的第一跳查询及返回内容逐题核验一致。支持事实严格按原有 token 可见性规则评分，模型自行生成的文本不计为检索证据。覆盖率为逐题平均。检索 token 统计每次 observation 截断后 token 数，不含生成 token，也不等于整段累计上下文长度。

这 200 题已用于验证和 checkpoint 选择，因此结果用于同题配对诊断，不代表独立测试集泛化。

- One training seed per method; old baseline shuffle seed not recorded.
- Forced policies enforce action prefixes and closing tags. Autonomous uses native actions; protocol differs.
- Forced fixed policies share first-hop query tokens. No wall-clock latency claim.
- Full gold support coverage is not proof that the answer is correct.
- The original 200 questions were used for validation/checkpoint selection; these results are a controlled replay, not a heldout test.
- Paired question bootstrap measures evaluation-sample uncertainty, not training-seed variability. Secondary p-values are uncorrected for multiple comparisons.

完整逐题轨迹、冻结 manifest、协议、CSV、配对统计保存在同目录。analysis/ 不推送。
