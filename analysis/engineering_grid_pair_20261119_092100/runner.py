from pathlib import Path
import subprocess,json,os,time
runs={'offload': '/g22538014wej/cityins1122/cityins/algorithm_a_runs/engineering_offload_2gpu_20261119_092100', 'resident': '/g22538014wej/cityins1122/cityins/algorithm_a_runs/engineering_resident_2gpu_20261119_092100'}
commands={'offload': ['bash', 'run_engineering_benchmark_2gpu.sh', 'engineering.profile_micro_batches=[2,4,8]', 'actor_rollout_ref.actor.engineering_loss_group_size=1'], 'resident': ['bash', 'run_engineering_benchmark_2gpu.sh', 'engineering.profile_micro_batches=[2,4,8]', 'actor_rollout_ref.actor.engineering_loss_group_size=1', 'engineering.workload_path=/g22538014wej/cityins1122/cityins/algorithm_a_runs/engineering_offload_2gpu_20261119_092100/sampling/engineering_workload.pt', 'actor_rollout_ref.actor.fsdp_config.param_offload=false', 'actor_rollout_ref.actor.fsdp_config.grad_offload=false', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=false', 'actor_rollout_ref.ref.fsdp_config.param_offload=false']}
audit=Path('/g22538014wej/cityins1122/cityins/analysis/engineering_grid_pair_20261119_092100')

for name in ('offload','resident'):
 environment=dict(os.environ,RUN_DIR=runs[name],CUDA_VISIBLE_DEVICES='0,1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='disabled')
 with (audit/(name+'.log')).open('w') as log:
  result=subprocess.run(commands[name],cwd='/g22538014wej/cityins1122/cityins',env=environment,stdout=log,stderr=subprocess.STDOUT)
 (audit/(name+'_exit.json')).write_text(json.dumps({'exit_code':result.returncode})+'\n')
 if result.returncode:raise SystemExit(result.returncode)
 for attempt in range(30):
  used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).split()
  if all(int(x)<500 for x in used):break
  time.sleep(1)
 else:raise RuntimeError('Prior GPU processes did not exit')
(audit/'exit_code.json').write_text('{"exit_code":0}\n')
