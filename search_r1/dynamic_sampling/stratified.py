"""Offline prompt strata and deterministic sampling; no reward/prompt changes."""
import hashlib
import json
import math
from pathlib import Path
import random
from collections import Counter

ROLES = ('first_sufficient', 'first_insufficient', 'query_rewrite', 'unobserved')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path, dataset_path, sample_ids):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported behavior-strata schema')
    if manifest.get('dataset_sha256') != file_sha256(dataset_path):
        raise ValueError('Behavior strata were built for different training data; rebuild the manifest')
    roles = manifest.get('roles', {})
    missing = set(map(str, sample_ids)) - set(roles)
    if missing or any(role not in ROLES for role in roles.values()):
        raise ValueError('Behavior strata have missing IDs or unsupported roles')
    return manifest


class StratifiedQuestionSampler:
    """Sample prompt indices, reserving uniform exploration and whole GRPO groups.

    Quotas cover candidate prompts, not model actions or post-filtered groups.
    Pool exhaustion falls back to uniform sampling and is explicitly counted.
    """
    def __init__(self, sample_ids, roles, batch_size, seed=42, uniform_fraction=.25,
                 role_weights=None):
        self.sample_ids = list(map(str, sample_ids))
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError('Training sample IDs must be unique')
        if not 0 <= uniform_fraction <= 1 or batch_size < 1:
            raise ValueError('Invalid stratified batch size or uniform fraction')
        self.weights = dict(role_weights or {'first_sufficient': 1., 'first_insufficient': 2., 'query_rewrite': 1.})
        if any(k not in ROLES or not math.isfinite(float(v)) or v < 0 for k, v in self.weights.items()):
            raise ValueError('Invalid behavior role weight')
        if uniform_fraction < 1 and sum(self.weights.values()) <= 0:
            raise ValueError('At least one positive behavior role weight is required')
        self.roles = {sid: roles[sid] for sid in self.sample_ids}
        self.batch_size, self.uniform_fraction = int(batch_size), float(uniform_fraction)
        self.rng = random.Random(int(seed))
        self.pools = {'uniform': list(range(len(self.sample_ids)))}
        for role in ROLES:
            self.pools[role] = [i for i, sid in enumerate(self.sample_ids) if self.roles[sid] == role]
        for pool in self.pools.values(): self.rng.shuffle(pool)
        self.positions = {role: 0 for role in self.pools}
        self.cycles = Counter()
        self.total_drawn = 0

    def _quotas(self):
        uniform = min(self.batch_size, math.ceil(self.batch_size * self.uniform_fraction))
        remaining = self.batch_size - uniform
        result = {'uniform': uniform}
        if not remaining: return result
        total = sum(self.weights.values())
        raw = {role: remaining * float(weight) / total for role, weight in self.weights.items()}
        result.update({role: math.floor(value) for role, value in raw.items()})
        # Rotate equal fractional remainders deterministically rather than always
        # favoring the first role at small batch sizes.
        order = list(raw); self.rng.shuffle(order)
        order.sort(key=lambda role: raw[role] - math.floor(raw[role]), reverse=True)
        for role in order[:self.batch_size - sum(result.values())]: result[role] += 1
        return result

    def _draw(self, role, blocked):
        pool = self.pools[role]
        for _ in range(len(pool) * 2):
            if self.positions[role] == len(pool):
                self.rng.shuffle(pool); self.positions[role] = 0; self.cycles[role] += 1
            index = pool[self.positions[role]]; self.positions[role] += 1
            if self.sample_ids[index] not in blocked: return index
        return None

    def sample(self, excluded_ids=()):
        blocked = set(map(str, excluded_ids))
        if sum(sid not in blocked for sid in self.sample_ids) < self.batch_size:
            raise ValueError('Not enough distinct training questions for candidate refill')
        quotas = self._quotas()
        slots = [role for role, count in quotas.items() for _ in range(count)]
        # Allocate constrained pools before uniform slots so uniform sampling
        # cannot consume all members of a small role before its own allocation.
        slots.sort(key=lambda role: (role == 'uniform', len(self.pools[role])))
        selected, fallback = [], Counter()
        for role in slots:
            index = self._draw(role, blocked)
            if index is None:
                fallback[role] += 1; index = self._draw('uniform', blocked)
            if index is None: raise RuntimeError('Uniform fallback could not fill a unique prompt batch')
            selected.append(index); blocked.add(self.sample_ids[index])
        self.rng.shuffle(selected)
        self.total_drawn += len(selected)
        return selected, dict(requested=quotas, fallback=dict(fallback),
                              realized=dict(Counter(self.roles[self.sample_ids[i]] for i in selected)))
