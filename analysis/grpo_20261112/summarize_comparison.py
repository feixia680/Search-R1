import json,csv,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path("/g22538014wej/11x22f/Search-R1")
OUT=ROOT/"analysis/grpo_20261112"
summ=json.loads((OUT/"training_summary.json").read_text())
audit=json.loads((OUT/"checkpoint_audit.json").read_text())
def wilson(k,n):
 z=1.95996398454;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [mid-d,mid+d]
records={};table=[]
for label in ["base","step375","step500"]:
 rows=[json.loads(s) for s in (OUT/(label+".jsonl")).read_text().splitlines()]
 assert len(rows)==200 and len({r["sample_id"] for r in rows})==200
 records[label]=rows
 s=json.loads((OUT/(label+"_summary.json")).read_text());s["wilson95"]=wilson(s["correct"],s["n"])
 s["legacy_action_em_mismatches"]=sum(r["legacy_em"]!=r["action_em"] for r in rows)
 table.append(s)
for label in records:
 assert [r["sample_id"] for r in records[label]]==[r["sample_id"] for r in records["base"]]
 assert [r["gold"] for r in records[label]]==[r["gold"] for r in records["base"]]
paired={};rng=np.random.default_rng(42)
for a,b in [("base","step375"),("base","step500"),("step375","step500")]:
 x=np.array([r["legacy_em"] for r in records[a]]);y=np.array([r["legacy_em"] for r in records[b]])
 dif=y-x;up=int(((x==0)&(y==1)).sum());down=int(((x==1)&(y==0)).sum())
 boots=dif[rng.integers(0,200,size=(20000,200))].mean(1)
 paired[a+"_vs_"+b]=dict(delta_pp=float(dif.mean()*100),relative_gain=float(y.mean()/x.mean()-1) if x.mean() else None,wrong_to_right=up,right_to_wrong=down,both_right=int(((x==1)&(y==1)).sum()),both_wrong=int(((x==0)&(y==0)).sum()),paired_bootstrap95_pp=(np.quantile(boots,[.025,.975])*100).tolist(),mcnemar_exact_p=binomtest(up,up+down,.5).pvalue if up+down else 1.)
result=dict(models=table,paired=paired,scope="200 same validation questions used during training; exploratory one-seed evaluation, not a new independent test set; one GPU for re-evaluation vs two GPUs in historical run")
(OUT/"comparison_summary.json").write_text(json.dumps(result,indent=2))
with (OUT/"paired_comparison.csv").open("w") as f:
 fields=["sample_id","gold"]+[f"{label}_{m}" for label in records for m in ["legacy_em","action_em","final_action_answer","retrieval_calls","invalid_actions"]]
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for i in range(200):
  r={"sample_id":records["base"][i]["sample_id"],"gold":json.dumps(records["base"][i]["gold"],ensure_ascii=False)}
  for label in records:
   for m in ["legacy_em","action_em","final_action_answer","retrieval_calls","invalid_actions"]:r[label+"_"+m]=records[label][i][m]
  w.writerow(r)
lines=["# Search-R1 GRPO 训练分析","", "## 运行与保存","", "模型为 Qwen2.5-3B Base，HotpotQA 训练文件5000题，batch=8，每题2条轨迹；不是新下载的7B。","", "日志完整记录更新1—624；计数加一后执行 >=625 提前返回，最后评估标签为625，实际没有第625次更新，因此仅使用4992道训练题。最终模型未保存；当前进程退出后不能从已有文件恢复最后624次更新后的精确权重。","", "## 训练曲线","", "| 指标 | 前100次更新均值 | 后100次更新均值 |","|---|---:|---:|"]
for k in summ["windows"]["first100"]:
 lines.append(f'| {k} | {summ["windows"]["first100"][k]:.4f} | {summ["windows"]["last100"][k]:.4f} |')
lines+=["", "训练奖励/EM、格式有效率和结束率均改善；KL升高、熵下降，说明策略逐渐偏离初始模型并更确定。单凭熵下降不能断言策略坍塌。训练奖励来自不同题目及随机采样，不能直接与固定题greedy验证EM当作相同分布对比。", "", f'624次更新中{summ["zero_advantage_batches"]}个batch的已记录优势最大值/最小值均为0；这里只能说明日志精度下没有明显的组内答案区分信号，不表示KL项也没有梯度。最后100次更新有{summ["zero_advantage_last100"]}个此类batch。', "", f'已记录指标未发现非有限值。梯度范数最大{summ["max_grad_norm"]}；该值为梯度裁剪前范数，不能单凭峰值判定训练发散。控制台将学习率保留3位小数，显示0.000不代表实际学习率为0。', "", "## 历史验证与检查点","", "| 日志step | EM | 权重保存 |","|---:|---:|---|"]
for v in summ["validation"]:lines.append(f'| {v["step"]} | {v["em"]:.1%} | '+("已保存" if v["step"] in (125,250,375,500) else "未保存；实际为624次更新后评估")+" |")
lines+=["", "四个检查点的safetensors索引、分片大小与头部数据偏移、基础模型对应张量形状均核对。保存权重为FP32，Base为BF16；额外lm_head来自共享输出权重的保存副本，不能把文件体积变化解释成参数规模增长。文件夹未保存优化器/调度器状态，不能据此精确恢复训练。", "", "## 同题比较","", "复用训练验证的sample(200, random_state=42)及顺序，batch=8，原始生成循环；greedy，最多两轮检索加最终生成，单次256token，检索topk=3，max_obs_length=1024，max_prompt_length=2816，max_start_length=512；同一BM25索引、语料、Base tokenizer与BF16推理。每个模型独立进程加载。", "", "| 模型 | EM | 95% Wilson区间 | 有效动作最终答案EM | 实际检索次数/题 | 缺少答案数 |","|---|---:|---|---:|---:|---:|"]
for s in table:lines.append(f'| {s["label"]} | {s["correct"]}/{s["n"]} = {s["em"]:.1%} | {s["wilson95"][0]:.1%}—{s["wilson95"][1]:.1%} | {s["action_correct"]}/200 | {s["mean_search"]:.3f} | {s["missing_final"]} |')
lines+=["", "主指标采用原仓库完整prompt+response的EM计分函数；另外保存实际answer动作的EM，检查二者是否因prompt/检索内容标签而不一致。", ""]
for key,s in paired.items():
 lines.append(f'- {key}: ΔEM={s["delta_pp"]:+.2f}个百分点；错→对{s["wrong_to_right"]}题，对→错{s["right_to_wrong"]}题；配对bootstrap95%区间{s["paired_bootstrap95_pp"][0]:+.2f}至{s["paired_bootstrap95_pp"][1]:+.2f}个百分点，McNemar精确p={s["mcnemar_exact_p"]:.5g}。')
lines+=["", "这些题已用于训练期间验证并影响检查点选择，因此本轮属于既有验证集的配对复核；验证过程本身不反传，不能仅因文件叫test.parquet就断言训练标签泄露。需要另留未参与选择的题目，才能报告独立测试提升。单卡重新评估相对历史双卡可能受浮点/批次执行差异影响，历史分数与新评估分开列出。", "", "step375按既有验证分数选为已保存最佳，step500为最新已保存模型；最后25.5%的模型未保存，不能把这个分数归给step500。", "", "## 文件","", "- training_metrics.csv：全部原始解析指标（控制台精度）","- training_summary.json、training_curves.png/pdf：曲线及统计","- checkpoint_audit.json：分片与张量结构核对","- evaluation_manifest.json：相同200题清单","- base/step375/step500.jsonl：逐题输出、检索文档及动作","- comparison_summary.json、paired_comparison.csv：配对统计与逐题比较",""]
(OUT/"REPORT.md").write_text("\n".join(lines))
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
vals=[s["em"]*100 for s in table]
errors=np.array([[s["em"]-s["wilson95"][0],s["wilson95"][1]-s["em"]] for s in table]).T*100
ax.bar([s["label"] for s in table],vals,yerr=errors,capsize=5,color=["#777777","#3182bd","#31a354"])
ax.set(ylabel="Exact match (%)",title="Same 200 validation questions; Wilson 95% CI")
for i,s in enumerate(table):ax.text(i,vals[i]+1,f'{s["correct"]}/200',ha="center")
fig.savefig(OUT/"baseline_comparison.png",dpi=180);fig.savefig(OUT/"baseline_comparison.pdf");plt.close(fig)
print(json.dumps(result,indent=2))

