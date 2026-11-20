# 搜索策略受控评估（已完成）

200道既有验证题，同一step500模型、首跳和检索语料。固定前缀控制查询/答案生成；按需条件为首跳答案token几何平均概率<0.8或答案为空，不使用标准答案。第二跳分支共享计算，因此不报告实测按需延迟。

|策略|EM|平均检索次数|平均支持事实覆盖|
|---|---:|---:|---:|
|fixed_one|25.0%|1.00|59.5%|
|fixed_two|28.5%|2.00|64.3%|
|adaptive|26.5%|1.30|61.3%|

固定两次新增证据出现在19/200题，10题由错变对、3题由对变错。第二次query与第一次完全重复132/200；重搜经常无新增信息。

按需触发60/200题，7题获得新增证据；4题由错变对、1题由对变错。相对固定一次的EM差值95%配对bootstrap区间为-0.5至+4.0个百分点，尚不能断言稳定改善。

这些题曾用于历史检查点选择；属于验证集对照。结果不等于独立测试集提升。

{
  "paired_comparisons": {
    "fixed_one_vs_fixed_two": {
      "delta_em": 0.035,
      "paired_95ci": [
        0.0,
        0.07
      ],
      "wrong_to_right": 10,
      "right_to_wrong": 3
    },
    "fixed_one_vs_adaptive": {
      "delta_em": 0.015,
      "paired_95ci": [
        -0.005,
        0.04
      ],
      "wrong_to_right": 4,
      "right_to_wrong": 1
    },
    "adaptive_vs_fixed_two": {
      "delta_em": 0.02,
      "paired_95ci": [
        -0.005,
        0.05
      ],
      "wrong_to_right": 6,
      "right_to_wrong": 2
    }
  },
  "second_hop_new_evidence_questions": 19,
  "adaptive_new_evidence_questions": 7,
  "second_hop_repeated_query": 132
}