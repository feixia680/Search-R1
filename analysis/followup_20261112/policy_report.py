import json
from pathlib import Path
import numpy as np
p=Path('/g22538014wej/11x22f/Search-R1/analysis/followup_20261112')
rows=[json.loads(x) for x in (p/'policies/step500.jsonl').read_text().splitlines()]
scores={mode:np.array([r['em2'] if mode=='fixed_two' or (mode=='adaptive' and r['adaptive_second']) else r['em1'] for r in rows]) for mode in ['fixed_one','fixed_two','adaptive']}
rng=np.random.default_rng(42); pairs={}
for a,b in [('fixed_one','fixed_two'),('fixed_one','adaptive'),('adaptive','fixed_two')]:
 d=scores[b]-scores[a]; boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
 pairs[f'{a}_vs_{b}']=dict(delta_em=float(d.mean()),paired_95ci=np.quantile(boot,[.025,.975]).tolist(),wrong_to_right=int((d==1).sum()),right_to_wrong=int((d==-1).sum()))
result=dict(paired_comparisons=pairs,second_hop_new_evidence_questions=sum(r['gain2']>0 for r in rows),adaptive_new_evidence_questions=sum(r['gain2']>0 and r['adaptive_second'] for r in rows),second_hop_repeated_query=sum(r['query1'].strip().casefold()==r['query2'].strip().casefold() for r in rows))
(p/'policies/paired_analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
