from pathlib import Path
import json, re, math
import torch
from safetensors import safe_open
root=Path('/g22538014wej/cityins1122/cityins')
analysis=root/'analysis/evidence_dapo_30538_20261120_095215'
run=Path((analysis/'smoke_run_dir').read_text())
assert (analysis/'smoke_exit_code').read_text().strip() == '0'
checkpoint=run/'checkpoints/actor/global_step_2'
base=root/'models/Qwen2.5-3B'
def weight_map(folder):
 index=folder/'model.safetensors.index.json'
 if index.exists():
  return json.loads(index.read_text())['weight_map']
 result={}
 for file in folder.glob('*.safetensors'):
  with safe_open(str(file),framework='pt',device='cpu') as f:
   result.update({k:file.name for k in f.keys()})
 return result
saved=weight_map(checkpoint); original=weight_map(base)
assert saved and set(original).issubset(saved)
extra_keys=set(saved)-set(original)
assert extra_keys.issubset({'lm_head.weight'}), extra_keys
if extra_keys:
 assert json.loads((base/'config.json').read_text())['tie_word_embeddings']
 with safe_open(str(checkpoint/saved['lm_head.weight']),framework='pt',device='cpu') as f:
  head_shape=f.get_slice('lm_head.weight').get_shape()
 with safe_open(str(base/original['model.embed_tokens.weight']),framework='pt',device='cpu') as f:
  assert head_shape==f.get_slice('model.embed_tokens.weight').get_shape()
for filename in set(saved.values()):
 with safe_open(str(checkpoint/filename),framework='pt',device='cpu') as f:
  assert all(k in f.keys() for k,v in saved.items() if v==filename)
changes={}
for key in ['model.layers.0.self_attn.q_proj.weight','model.layers.0.input_layernorm.weight','model.norm.weight']:
 with safe_open(str(checkpoint/saved[key]),framework='pt',device='cpu') as f:
  actual=f.get_tensor(key).float()
 with safe_open(str(base/original[key]),framework='pt',device='cpu') as f:
  before=f.get_tensor(key).float()
 assert torch.isfinite(actual).all()
 changes[key]=float((actual-before).abs().max())
assert max(changes.values())>0
records=[json.loads(x) for x in (run/'traces/train.jsonl').read_text().splitlines()]
assert len(records)==32
assert {x['global_step'] for x in records}=={1,2}
assert all(abs(x['total_reward']-x['em']-x['evidence_reward'])<1e-6 for x in records)
validation=[json.loads(x) for x in (run/'traces/validation.jsonl').read_text().splitlines()]
assert len(validation)==4
metrics=[]
for line in (run/'train.log').read_text(errors='replace').splitlines():
 if 'actor/token_mean_loss:' in line:
  row=dict(re.findall(r'(step|actor/token_mean_loss|actor/grad_norm|algorithm_a/total_reward|timing_s/step):([-+0-9.e]+)',line))
  assert float(row['actor/token_mean_loss'])==1
  assert math.isfinite(float(row['actor/grad_norm']))
  metrics.append(row)
assert len(metrics)==2
result=dict(smoke_passed=True,training_steps=2,training_trajectories=len(records),
 validation_trajectories=len(validation),checkpoint=str(checkpoint),
 checkpoint_tensor_count=len(saved),checkpoint_bytes=sum((checkpoint/f).stat().st_size for f in set(saved.values())),
 parameter_max_absolute_changes=changes,metrics=metrics,formal_training_started=False)
(analysis/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
