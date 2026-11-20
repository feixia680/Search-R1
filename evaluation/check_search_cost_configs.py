import os, subprocess, json
from pathlib import Path
import yaml
out=Path('analysis/cost_design_20261113')
summary=[]
for g in (1,2):
    for name in (f'train_grpo_cost_{g}gpu.sh',f'run_grpo_cost_{g}gpu_with_bm25.sh'):
        subprocess.run(['bash','-n',name],check=True)
    for arm,beta,lam in [('evidence',0,0),('uniform',.005,0),('state',.005,.01)]:
        env=dict(os.environ,COST_ARM=arm,CUDA_VISIBLE_DEVICES='')
        for key in ['SEARCH_BETA','AFTER_FULL_LAMBDA','RUN_DIR','EXPERIMENT_NAME','EVIDENCE_ALPHA']:
            env.pop(key,None)
        r=subprocess.run(['bash',f'run_grpo_cost_{g}gpu_with_bm25.sh','--print-config'],
                         env=env,text=True,capture_output=True,check=True)
        (out/f'config_{arm}_{g}gpu.yaml').write_text(r.stdout)
        config=yaml.safe_load(r.stdout)
        reward=config['search_reward']
        assert reward['alpha']==.1 and reward['beta']==beta and reward['after_full_lambda']==lam,reward
        assert reward['require_final_answer'] is True
        assert reward['gamma']==0 and reward['delta']==0
        assert config['trainer']['n_gpus_per_node']==g
        assert config['data']['train_batch_size']==8 and config['data']['val_data_num']==200
        assert config['trainer']['total_training_steps']==625 and config['max_turns']==2
        assert 'cost_'+arm+'_'+str(g)+'gpu' in config['trainer']['default_local_dir']
        assert config['actor_rollout_ref']['model']['path'].endswith('/models/Qwen2.5-3B')
        summary.append(dict(gpus=g,arm=arm,beta=beta,after_full_lambda=lam,
                            guard=reward['require_final_answer'],steps=625))
(out/'config_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: 4 shell syntax checks and 6 resolved Hydra configurations; no GPU training launched.')

