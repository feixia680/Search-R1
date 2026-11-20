import subprocess,json
from pathlib import Path
result=subprocess.run(['bash', 'run_engineering_benchmark_2gpu.sh', 'engineering.profile_micro_batches=[2,4,8]', 'actor_rollout_ref.actor.engineering_loss_group_size=1', 'engineering.workload_path=/g22538014wej/cityins1122/cityins/algorithm_a_runs/engineering_offload_2gpu_20261119_092100/sampling/engineering_workload.pkl', 'actor_rollout_ref.actor.fsdp_config.param_offload=false', 'actor_rollout_ref.actor.fsdp_config.grad_offload=false', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=false', 'actor_rollout_ref.ref.fsdp_config.param_offload=false'],cwd='/g22538014wej/cityins1122/cityins')
Path('/g22538014wej/cityins1122/cityins/analysis/engineering_resident_replay_20261119_092829/exit_code.json').write_text(json.dumps({"exit_code":result.returncode})+"\n")
raise SystemExit(result.returncode)
