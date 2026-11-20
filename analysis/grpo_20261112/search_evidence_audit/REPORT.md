# Search-R1 搜索轨迹与证据覆盖审计

## 范围与口径

使用同一组 200 道 HotpotQA 验证题，检查 base、step375、step500 的实际执行轨迹。按问题映射到原始 HotpotQA validation parquet 的 supporting_facts，再查 BM25 corpus 是否含对应标题和句子，并在保存的检索 observation 上复现 generation.py 的 tokenizer 截断：每次 observation 最多 1024 token。tokenizer 为 right padding，因此逐条截取前 1024 token 与原批处理截断一致。

“返回”表示检索 observation 在截断前含完整支持句；“可见”表示截断后完整支持句仍在 observation。只统计完整句匹配。它衡量标注证据覆盖，不等同于所有可能的充分证据。

## 核心结果

|策略|EM|检索次数分布（0/1/2）|support facts 在 corpus|完整返回|截断后可见|题目全部 support facts 可见|
|---|---:|---|---:|---:|---:|---:|
|base|21/200 (10.5%)|64/129/7|469/469 (100%)|144/469 (30.7%)|144/469 (30.7%)|22/200|
|step375|46/200 (23.0%)|0/200/0|469/469 (100%)|261/469 (55.7%)|261/469 (55.7%)|46/200|
|step500|53/200 (26.5%)|0/200/0|469/469 (100%)|280/469 (59.7%)|280/469 (59.7%)|54/200|

## 关键发现

- 200 题的 469 条 supporting fact 全部能在 BM25 corpus 找到；当前错误不能归因于语料缺少这些标注证据。
- step500 的 469 条支持句中，280 条（59.7%）被完整检索返回且在 1024-token 截断后可见。返回数与可见数相同：这组轨迹中没有完整支持句因 1024-token 上限变得不可见。
- step500 有 54 题的全部标注支持句可见：24 题答对、30 题答错。另有 2 题答对但没有完整标注支持句可见。总体 147 题答错，30 题在全部标注支持句可见时仍答错；错误既有证据覆盖问题，也有推理或答案抽取问题。
- Baseline 64/200 未成功搜索、129/200 搜索一次、7/200 搜索两次；两次搜索的 7 题全错。step375 和 step500 均 200/200 只搜索一次，没有重复 query。不能据此断言第二次搜索无用：样本少、策略不同，且没有受控的同题第二次搜索对照。
- 这 200 题在原始 validation split 的 level 均为 hard，不能据此比较 easy/medium/hard。

## 对算法 A 的直接启发

证据奖励可先定义为本轮新增、去重后的 supporting fact 覆盖率，只对模型实际看见的 1024-token 截断后 observation 计分；再配合搜索调用成本和重复 query 惩罚。不要把文档标题命中算作支持证据，也不要对重复出现的支持句重复加分。训练后模型已稳定为一次搜索，单纯增加搜索惩罚可能压掉有效搜索。建议固定初始化、数据顺序和更新步数，对照 EM only、EM + evidence gain、EM + evidence gain + search/repeat cost。

## 输出文件

- summary.csv：三种策略汇总
- summary_detailed.json：检索次数和证据可见分层细项
- per_trajectory_evidence.csv：600 条逐轨迹统计，含 query 与 supporting facts
- step500_by_type_level.csv：step500 按题型/难度切分
- step500_wrong_cases.csv：step500 错题明细
