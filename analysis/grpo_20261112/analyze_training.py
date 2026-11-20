import json,re,math,struct,csv,hashlib
from pathlib import Path
import numpy as np
ROOT=Path("/g22538014wej/11x22f/Search-R1");OUT=ROOT/"analysis/grpo_20261112"
raw=(ROOT/"hotpotqa-qwen25-3b-grpo-2gpu.log").read_text(errors="replace")
text=re.sub(r"\x1b\[[0-9;]*m","",raw)
rows=[]
for line in text.splitlines():
 m=re.search(r"step:(\d+) - (.*)",line)
 if m:
  r={"step":int(m[1])}
  for k,v in re.findall(r"(?:^| - )([^ :]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)",m[2]):r[k]=float(v)
  rows.append(r)
keys=["step"]+sorted(set().union(*(r.keys() for r in rows))-{"step"})
with (OUT/"training_metrics.csv").open("w") as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
train=[r for r in rows if "actor/grad_norm" in r]
val=[{"step":r["step"],"em":r["val/test_score/hotpotqa"]} for r in rows if "val/test_score/hotpotqa" in r]
fields=["critic/score/mean","actor/kl_loss","actor/entropy_loss","actor/grad_norm","env/finish_ratio","env/ratio_of_valid_action","env/number_of_valid_search","response_length/mean","timing_s/gen","timing_s/update_actor"]
windows={}
for name,items in [("first100",train[:100]),("last100",train[-100:])]:
 windows[name]={k:float(np.mean([r[k] for r in items if k in r])) for k in fields}
summary=dict(training_records=len(train),first_update=train[0]["step"],last_update=train[-1]["step"],validation=val,windows=windows,
 nonfinite_values=[{"step":r["step"],"key":k} for r in rows for k,v in r.items() if not math.isfinite(v)],
 zero_advantage_batches=sum(r.get("critic/advantages/max")==0 and r.get("critic/advantages/min")==0 for r in train),
 zero_advantage_last100=sum(r.get("critic/advantages/max")==0 and r.get("critic/advantages/min")==0 for r in train[-100:]),
 logged_training_wall_hours=sum(r.get("timing_s/step",0) for r in train)/3600,
 max_grad_norm=max(r["actor/grad_norm"] for r in train),
 lr_logging_precision_note="console rounds LR to three decimals; zero in CSV is not evidence of zero actual learning rate",
 final_checkpoint_note="fit increments global_steps before >= total_training_steps check, so updates 1..624 are logged, final evaluation labeled625; no final save")
(OUT/"training_summary.json").write_text(json.dumps(summary,indent=2))
def inventory(p):
 index=json.loads((p/"model.safetensors.index.json").read_text())
 tensors={};files=[]
 for name in sorted(set(index["weight_map"].values())):
  f=p/name
  with f.open("rb") as s:
   n=struct.unpack("<Q",s.read(8))[0];h=json.loads(s.read(n))
  h.pop("__metadata__",None);tensors.update(h)
  files.append(dict(name=name,size=f.stat().st_size,header_extent_valid=(8+n+max(x["data_offsets"][1] for x in h.values())==f.stat().st_size)))
 return dict(path=str(p),files=files,tensor_count=len(tensors),stored_parameters=sum(math.prod(x["shape"]) for x in tensors.values()),dtypes=sorted(set(x["dtype"] for x in tensors.values())),indexed_tensors_present=set(index["weight_map"]).issubset(tensors)),tensors
base,bt=inventory(ROOT/"models/Qwen2.5-3B")
checkpoints=[]
for step in (125,250,375,500):
 inv,t=inventory(ROOT/f"verl_checkpoints/hotpotqa-qwen25-3b-grpo-2gpu/actor/global_step_{step}")
 inv["step"]=step;inv["missing_base_tensors"]=sorted(set(bt)-set(t));inv["shape_mismatches"]=[k for k in bt if k in t and bt[k]["shape"]!=t[k]["shape"]];inv["extra_tensors"]=sorted(set(t)-set(bt));checkpoints.append(inv)
(OUT/"checkpoint_audit.json").write_text(json.dumps(dict(base=base,checkpoints=checkpoints),indent=2))
try:
 import matplotlib;matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
 panels=[("critic/score/mean","Training EM / reward"),("actor/kl_loss","KL loss vs frozen reference"),("actor/entropy_loss","Policy entropy"),("env/finish_ratio","Answer completion ratio"),("env/number_of_valid_search","Search actions / trajectory"),("response_length/mean","Response tokens (includes retrieval)")]
 steps=[r["step"] for r in train]
 for ax,(k,title) in zip(axes.flat,panels):
  y=np.array([r[k] for r in train]);ax.plot(steps,y,alpha=.22,lw=.6)
  smooth=np.convolve(y,np.ones(25)/25,mode="valid");ax.plot(steps[24:],smooth,label="25-step moving mean")
  if k=="critic/score/mean":ax.plot([v["step"] for v in val],[v["em"] for v in val],"o-",label="Validation EM (200 questions)");ax.legend(fontsize=8)
  ax.set(title=title,xlabel="Logged step");ax.grid(alpha=.2)
 fig.suptitle("Search-R1 Qwen2.5-3B GRPO: 624 logged updates")
 fig.savefig(OUT/"training_curves.png",dpi=180);fig.savefig(OUT/"training_curves.pdf");plt.close(fig)
except ImportError as e:print("PLOT_IMPORT_ERROR",e)
print(json.dumps(summary,indent=2))

