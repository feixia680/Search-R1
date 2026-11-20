"""Separate FSDP GRPO entrypoint for bounded dynamic sampling or uniform control."""
import hydra
import ray
from omegaconf import OmegaConf


def validate_config(config):
    from search_r1.dynamic_sampling.selection import BoundedGroupSelector
    BoundedGroupSelector(int(config.data.train_batch_size),
        group_size=int(config.actor_rollout_ref.rollout.n_agent),
        max_batches=int(config.dynamic_sampling.max_candidate_batches),
        epsilon=float(config.dynamic_sampling.reward_range_epsilon))
    engineering = config.get('engineering', {})
    if engineering.get('profile_only', False):
        if float(config.actor_rollout_ref.actor.optim.lr) != 0:
            raise ValueError('Engineering benchmark requires actor learning rate zero')
        if int(engineering.get('profile_repeats', 3)) < 1:
            raise ValueError('Engineering profile_repeats must be positive')
    world = int(config.trainer.n_gpus_per_node)*int(config.trainer.nnodes)
    mini = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    micro = int(config.actor_rollout_ref.actor.ppo_micro_batch_size)
    group = int(config.actor_rollout_ref.actor.get('engineering_loss_group_size', 0))
    if micro < world or micro % world or mini % micro:
        raise ValueError('Fixed microbatch must divide world size and global minibatch')
    if group < 0 or (group and (micro//world) % group):
        raise ValueError('Legacy loss groups must divide each rank microbatch')
    grid = list(engineering.get('profile_micro_batches', []))
    if grid:
        if not engineering.get('profile_only', False):
            raise ValueError('Microbatch grid is only supported by the isolated benchmark')
        if any(m < world or m % world or mini % m for m in grid):
            raise ValueError('Profile microbatch must divide world size and global minibatch')
        if max(grid) > min(grid) and group != min(grid)//world:
            raise ValueError('Microbatch comparison must preserve the baseline loss groups')
    behavior = config.get('behavior_sampling', {})
    if behavior.get('enabled', False):
        from search_r1.dynamic_sampling.stratified import StratifiedQuestionSampler
        StratifiedQuestionSampler([], {}, int(config.data.train_batch_size),
            uniform_fraction=float(behavior.get('uniform_fraction', .25)),
            role_weights=dict(behavior.get('role_weights', {})))
    monitor = config.get('behavior_monitor', {})
    if monitor.get('enabled', False):
        from search_r1.dynamic_sampling.behavior import validate_monitor_options
        validate_monitor_options(**{key: value for key, value in monitor.items() if key != 'enabled'})
    if config.algorithm.adv_estimator != 'grpo' or not config.do_search:
        raise ValueError('This entrypoint supports multi-turn search GRPO only')
    arc = config.actor_rollout_ref
    if arc.actor.strategy != 'fsdp' or not arc.actor.use_kl_loss or not arc.actor.state_masking:
        raise ValueError('Requires FSDP, actor KL loss and state masking')
    if config.reward_model.enable or arc.rollout.n != 1 or arc.rollout.n_agent < 2:
        raise ValueError('Requires rule reward, rollout.n=1 and n_agent>=2')
    if arc.actor.use_dynamic_bsz or arc.rollout.log_prob_use_dynamic_bsz or arc.ref.log_prob_use_dynamic_bsz:
        raise ValueError('Numerical collectives require the fixed microbatch schedule')
    if arc.actor.ulysses_sequence_parallel_size != 1 or arc.rollout.tensor_model_parallel_size != 1:
        raise ValueError('This version supports data-parallel FSDP with SP=TP=1')
    if config.trainer.critic_warmup != 0 or config.trainer.nnodes != 1:
        raise ValueError('This version requires no critic warmup and one node')
    for key in ('beta', 'gamma', 'delta', 'after_full_lambda'):
        if config.search_reward.get(key, 0.) != 0:
            raise ValueError('Sampling experiment keeps all search penalties disabled: '+key)
    if config.search_reward.require_final_answer:
        raise ValueError('Keep final-answer guard identical to the historical evidence objective (false)')
    if int(config.trainer.total_training_steps) < 2:
        raise ValueError('Legacy stopping convention needs total_training_steps>=2')
    if config.trainer.total_epochs != 1:
        raise ValueError('Use the explicit update budget, not epoch count; total_epochs must remain 1')


@hydra.main(config_path='config', config_name='dynamic_sampling', version_base=None)
def main(config):
    validate_config(config)
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    try:
        ray.get(main_task.remote(config))
    finally:
        ray.shutdown()


@ray.remote
def main_task(config):
    from transformers import set_seed
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.trainer.main_ppo import RewardManager
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.trainer.ppo.dynamic_sampling_trainer import DynamicSamplingTrainer
    from verl.workers.dynamic_sampling_worker import GuardedActorRolloutRefWorker
    from verl.workers.fsdp_workers import CriticWorker
    from verl.single_controller.ray import RayWorkerGroup
    OmegaConf.resolve(config)
    validate_config(config)
    set_seed(config.trainer.get('seed', 42))
    tokenizer = hf_tokenizer(copy_local_path_from_hdfs(config.actor_rollout_ref.model.path))
    options = dict(alpha=config.search_reward.alpha,
        invalid_evidence_policy=config.search_reward.invalid_evidence_policy,
        trace_dir=config.search_reward.trace_dir, max_search_calls=config.max_turns)
    reward = RewardManager(tokenizer, 0, split='train', **options)
    candidate_reward = RewardManager(tokenizer, 0, split='candidates', **options)
    validation_reward = RewardManager(tokenizer, 1, trace_dir=config.search_reward.trace_dir, split='validation')
    roles = {Role.ActorRollout: ray.remote(GuardedActorRolloutRefWorker),
             Role.RefPolicy: ray.remote(GuardedActorRolloutRefWorker), Role.Critic: ray.remote(CriticWorker)}
    pool = ResourcePoolManager(
        resource_pool_spec={'global_pool': [config.trainer.n_gpus_per_node]},
        mapping={role: 'global_pool' for role in roles})
    trainer = DynamicSamplingTrainer(config=config, tokenizer=tokenizer,
        role_worker_mapping=roles, resource_pool_manager=pool, ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward, candidate_reward_fn=candidate_reward, val_reward_fn=validation_reward)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
