"""Deterministic token-budget packing and greedy rank balancing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
@dataclass(frozen=True)
class TrajectoryCost:
    trajectory_id: str
    token_count: int
    group_id: str = ""
    priority: float = 0.0
    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
@dataclass(frozen=True)
class PackedBatch:
    items: Tuple[TrajectoryCost, ...]
    token_count: int
    group_ids: Tuple[str, ...]
@dataclass(frozen=True)
class RankAssignment:
    rank: int
    batches: Tuple[PackedBatch, ...]
    token_count: int
class TokenPacker:
    """Pack whole GRPO groups under a token budget."""
    def __init__(self, max_tokens: int, max_items: Optional[int] = None, allow_oversized_group: bool = False):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_tokens = max_tokens
        self.max_items = max_items
        self.allow_oversized_group = allow_oversized_group
    @staticmethod
    def _groups(items: Sequence[TrajectoryCost]):
        grouped: Dict[str, List[TrajectoryCost]] = {}
        order: List[str] = []
        for item in items:
            key = item.group_id or f"__single__:{item.trajectory_id}"
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(item)
        return [(index, key, grouped[key]) for index, key in enumerate(order)]
    def pack(self, items: Iterable[TrajectoryCost]) -> List[PackedBatch]:
        values = list(items)
        ids = [item.trajectory_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate trajectory_id in pack input")
        prepared = []
        for original_index, group_id, group_items in self._groups(values):
            token_count = sum(item.token_count for item in group_items)
            if token_count > self.max_tokens and not self.allow_oversized_group:
                raise ValueError(f"group {group_id!r} has {token_count} tokens, exceeds budget {self.max_tokens}")
            priority = max(item.priority for item in group_items)
            prepared.append((priority, -token_count, original_index, group_items, token_count))
        prepared.sort(key=lambda row: (-row[0], row[1], row[2]))
        batches: List[List[TrajectoryCost]] = []
        loads: List[int] = []
        for _, _, _, group_items, group_tokens in prepared:
            placed = False
            for index in range(len(batches)):
                if loads[index] + group_tokens > self.max_tokens:
                    continue
                if self.max_items is not None and len(batches[index]) + len(group_items) > self.max_items:
                    continue
                batches[index].extend(group_items)
                loads[index] += group_tokens
                placed = True
                break
            if not placed:
                batches.append(list(group_items))
                loads.append(group_tokens)
        return [PackedBatch(items=tuple(batch), token_count=load, group_ids=tuple(dict.fromkeys(item.group_id for item in batch if item.group_id))) for batch, load in zip(batches, loads)]
    @staticmethod
    def balance(batches: Sequence[PackedBatch], world_size: int) -> List[RankAssignment]:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        assignments: List[List[PackedBatch]] = [[] for _ in range(world_size)]
        loads = [0 for _ in range(world_size)]
        for _, batch in sorted(enumerate(batches), key=lambda pair: (-pair[1].token_count, pair[0])):
            rank = min(range(world_size), key=lambda index: (loads[index], index))
            assignments[rank].append(batch)
            loads[rank] += batch.token_count
        return [RankAssignment(rank=rank, batches=tuple(assignments[rank]), token_count=loads[rank]) for rank in range(world_size)]
