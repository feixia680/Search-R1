import pytest
from search_r1.dynamic_sampling.engineering import partition_summary
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

def test_actual_balancer_reduces_gap():
    lengths=[1,2,3,4,100,101,102,103]
    before=partition_summary(lengths,2)
    parts=get_seqlen_balanced_partitions(lengths,k_partitions=2,equal_size=True)
    after=partition_summary([lengths[i] for part in parts for i in part],2)
    assert sorted(i for part in parts for i in part)==list(range(8))
    assert after['cv_percent']<before['cv_percent']
    assert after['mean_tokens']==before['mean_tokens']

def test_unequal_rank_counts_rejected():
    with pytest.raises(ValueError):partition_summary([1,2,3],2)
