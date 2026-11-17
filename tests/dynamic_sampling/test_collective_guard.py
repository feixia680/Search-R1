from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
import torch.distributed as dist
import torch.multiprocessing as mp
from verl.workers.dynamic_sampling_worker import collective_check


def guard_rank(rank,folder):
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),
                            rank=rank,world_size=2,timeout=timedelta(seconds=30))
    stopped=False
    try:
        collective_check(rank==0,'one_rank_nonfinite',folder,{'rank':rank})
    except FloatingPointError:
        stopped=True
    finally:
        (Path(folder)/f'result_{rank}.json').write_text(json.dumps({'stopped':stopped}))
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),'Gloo unavailable')
class CollectiveGuardTests(unittest.TestCase):
    def test_one_bad_rank_stops_both_ranks(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(guard_rank,args=(folder,),nprocs=2,join=True)
            for rank in (0,1):
                self.assertTrue(json.loads((Path(folder)/f'result_{rank}.json').read_text())['stopped'])


if __name__=='__main__':unittest.main()
