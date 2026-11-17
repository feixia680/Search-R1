import pytest
from search_r1.runtime.token_packer import TokenPacker, TrajectoryCost
def test_pack_respects_budget_and_keeps_groups_together():
    items = [TrajectoryCost("a1", 40, "g1"), TrajectoryCost("a2", 30, "g1"), TrajectoryCost("b1", 20, "g2"), TrajectoryCost("c1", 50, "g3")]
    batches = TokenPacker(max_tokens=70).pack(items)
    assert all(batch.token_count <= 70 for batch in batches)
    assert all(sum(item.group_id == "g1" for item in batch.items) in (0, 2) for batch in batches)
    assert {item.trajectory_id for batch in batches for item in batch.items} == {"a1", "a2", "b1", "c1"}
def test_pack_is_deterministic_and_rejects_duplicates():
    items = [TrajectoryCost("a", 10), TrajectoryCost("b", 20)]
    assert TokenPacker(30).pack(items) == TokenPacker(30).pack(items)
    with pytest.raises(ValueError, match="duplicate"):
        TokenPacker(30).pack([TrajectoryCost("a", 10), TrajectoryCost("a", 10)])
def test_balance_greedy_reduces_rank_load_skew():
    batches = TokenPacker(100).pack([TrajectoryCost(str(i), tokens) for i, tokens in enumerate([90, 80, 70, 60])])
    assignments = TokenPacker.balance(batches, world_size=2)
    loads = [assignment.token_count for assignment in assignments]
    assert sorted(loads) == [150, 150]
def test_oversized_group_requires_explicit_opt_in():
    item = TrajectoryCost("a", 101, "g")
    with pytest.raises(ValueError, match="exceeds budget"):
        TokenPacker(100).pack([item])
    assert TokenPacker(100, allow_oversized_group=True).pack([item])[0].token_count == 101
