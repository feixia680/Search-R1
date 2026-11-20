from pathlib import Path
import json,pickle,zipfile
meta=json.loads(Path('analysis/engineering_grid_pair_latest.json').read_text())
old=Path(meta['runs']['offload'])/'sampling/engineering_workload.pt'
with zipfile.ZipFile(old) as archive:
 names=[name for name in archive.namelist() if name.endswith('/data.pkl')]
 assert len(names)==1
 batch=pickle.loads(archive.read(names[0]))
new=old.with_suffix('.pkl');batch.save_to_disk(new)
from verl import DataProto
check=DataProto.load_from_disk(new)
assert len(check)==len(batch)
assert check.non_tensor_batch['uid'].tolist()==batch.non_tensor_batch['uid'].tolist()
print('Converted and verified trusted local workload:',str(new),len(check))
