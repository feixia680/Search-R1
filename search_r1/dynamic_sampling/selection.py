"""Pure-Python selection kernel; no model, GPU, or training-state mutations."""
from dataclasses import dataclass
import math
import random
import struct


@dataclass(frozen=True)
class CandidateGroup:
    batch_id: int
    sample_id: str
    row_indices: tuple
    rewards: tuple

    @property
    def reward_range(self):
        return max(self.rewards) - min(self.rewards)


class BoundedGroupSelector:
    def __init__(self, target_groups, group_size=2, max_batches=3, epsilon=1e-8,
                 enabled=True, seed=42):
        if target_groups < 1 or group_size < 2 or max_batches < 1:
            raise ValueError('Positive target/budget and at least two rollouts per group required')
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError('epsilon must be finite and nonnegative')
        self.target, self.group_size = target_groups, group_size
        self.max_batches, self.epsilon = max_batches, epsilon
        self.enabled, self.rng = enabled, random.Random(seed)
        self.seen, self.informative, self.zero = set(), [], []
        self.offered = []
        self.rounds = self.duplicate_groups = 0
        self._finished = False

    @property
    def ready(self):
        return (len(self.informative) >= self.target if self.enabled
                else len(self.informative) + len(self.zero) >= self.target)

    def offer(self, batch_id, sample_ids, rewards):
        if self._finished or self.rounds >= self.max_batches or self.ready:
            raise ValueError('Cannot add candidates after selection is complete or budget exhausted')
        if batch_id != self.rounds or len(sample_ids) != len(rewards):
            raise ValueError('Candidate batches must be consecutive and aligned')
        by_id = {}
        for i, (sid, reward) in enumerate(zip(sample_ids, rewards)):
            reward = struct.unpack('f', struct.pack('f', float(reward)))[0]
            if not math.isfinite(reward):
                raise FloatingPointError('Candidate outcome reward is nonfinite')
            by_id.setdefault(str(sid), []).append((i, reward))
        if len(by_id) != self.target:
            raise ValueError('Each candidate batch must contain target_groups distinct questions')
        # Validate before mutating the pool; duplicate questions within one batch
        # would otherwise merge two groups into one GRPO uid.
        if any(len(rows) != self.group_size for rows in by_id.values()):
            raise ValueError('Incorrect group size or duplicate question inside candidate batch')
        for sid, rows in by_id.items():
            group = CandidateGroup(batch_id, sid, tuple(x[0] for x in rows),
                                   tuple(x[1] for x in rows))
            self.offered.append(group)
            if sid in self.seen:
                self.duplicate_groups += 1
                continue
            self.seen.add(sid)
            (self.informative if group.reward_range > self.epsilon else self.zero).append(group)
        self.rounds += 1

    def finish(self):
        if self._finished or not self.rounds:
            raise ValueError('Selection is empty or already finalized')
        if not self.ready and self.rounds < self.max_batches:
            raise ValueError('Refill budget remains; selection may not finish early')
        if self.enabled:
            selected = self.informative[:self.target]
            missing = self.target - len(selected)
            if missing > len(self.zero):
                raise ValueError('Not enough unique finite groups for fallback')
            selected += self.rng.sample(self.zero, missing)
        else:
            # Uniform control: keep the first batch, including zero-difference
            # groups. Never silently apply dynamic filtering to the control.
            selected = self.offered[:self.target]
            missing = 0
        if len({g.sample_id for g in selected}) != self.target:
            raise ValueError('Selected questions must be unique inside the update')
        self._finished = True
        informative = sum(g.reward_range > self.epsilon for g in selected)
        metrics = dict(candidate_batches=self.rounds, candidate_groups=len(self.offered),
            candidate_trajectories=len(self.offered) * self.group_size,
            selected_groups=self.target, selected_trajectories=self.target * self.group_size,
            candidate_informative_groups=sum(g.reward_range > self.epsilon for g in self.offered),
            selected_informative_groups=informative, selected_zero_groups=self.target-informative,
            zero_fallback_groups=missing, duplicate_groups=self.duplicate_groups,
            discarded_groups=len(self.offered)-self.target,
            selected_effective_group_fraction=informative/self.target,
            generation_multiplier=len(self.offered)/self.target,
            budget_exhausted=int(self.enabled and self.rounds == self.max_batches and informative < self.target))
        return selected, metrics
