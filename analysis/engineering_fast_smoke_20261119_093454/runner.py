import subprocess,json
from pathlib import Path
result=subprocess.run(['bash', 'city_dynamic_2gpu_fast.sh', 'trainer.total_training_steps=3', 'trainer.test_freq=-1', 'trainer.save_freq=-1', '+trainer.skip_final_validation=true', 'dynamic_sampling.max_candidate_batches=2', 'behavior_monitor.window_size=1', 'behavior_monitor.warmup_steps=0', 'behavior_monitor.min_target_trajectories=2'],cwd='/g22538014wej/cityins1122/cityins')
Path('/g22538014wej/cityins1122/cityins/analysis/engineering_fast_smoke_20261119_093454/exit_code.json').write_text(json.dumps({"exit_code":result.returncode})+"\n")
raise SystemExit(result.returncode)
