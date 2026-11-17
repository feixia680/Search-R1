"""Resolve both launchers/control modes without starting retrieval or GPU jobs."""
import json
import os
from pathlib import Path
import subprocess
import yaml
from omegaconf import OmegaConf
from verl.trainer.main_dynamic_sampling import validate_config


def main():
    root=Path.cwd()
    out=root/'analysis/dynamic_sampling_implementation_20261117'
    out.mkdir(parents=True,exist_ok=True)
    checks=[]
    for gpu in (1,2):
        for script in (f'train_grpo_dynamic_sampling_{gpu}gpu.sh',f'run_grpo_dynamic_sampling_{gpu}gpu_with_bm25.sh'):
            subprocess.run(['bash','-n',script],check=True)
        for enabled in (True,False):
            preview=out/f'preview_must_not_be_created_{gpu}_{enabled}'
            if preview.exists():raise RuntimeError('Preview output directory unexpectedly exists')
            env=dict(os.environ,SAMPLING_ENABLED=str(enabled).lower(),MAX_CANDIDATE_BATCHES='3',
                     EVIDENCE_ALPHA='0.1',RUN_DIR=str(preview))
            result=subprocess.run(['bash',f'run_grpo_dynamic_sampling_{gpu}gpu_with_bm25.sh','--print-config'],
                                  env=env,text=True,capture_output=True,check=True)
            config=OmegaConf.create(yaml.safe_load(result.stdout))
            OmegaConf.resolve(config)
            validate_config(config)
            assert config.dynamic_sampling.enabled is enabled
            assert config.dynamic_sampling.max_candidate_batches==3
            assert config.trainer.n_gpus_per_node==gpu
            assert config.data.train_batch_size==8 and config.data.val_data_num==200
            assert config.max_turns==2 and config.trainer.total_training_steps==625
            assert config.search_reward.alpha==.1 and config.search_reward.require_final_answer is False
            assert all(config.search_reward[k]==0 for k in ('beta','gamma','delta','after_full_lambda'))
            assert not preview.exists(),'Config preview unexpectedly created a run'
            (out/f'config_{gpu}gpu_{enabled}.yaml').write_text(OmegaConf.to_yaml(config,resolve=True))
            checks.append(dict(gpus=gpu,enabled=enabled,status='PASS',max_batches=3,alpha=.1))
    for change in ({'dynamic_sampling':{'max_candidate_batches':0}},
                   {'search_reward':{'beta':.1}},
                   {'actor_rollout_ref':{'actor':{'use_dynamic_bsz':True}}}):
        bad=OmegaConf.merge(config,change)
        try:validate_config(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid configuration accepted: '+str(change))
    (out/'config_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('PASS: four shell syntax checks, four resolved configurations, three invalid-config checks; no GPU/retrieval launched.')


if __name__=='__main__':main()
