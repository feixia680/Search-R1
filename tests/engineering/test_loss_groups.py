import torch
import pytest
from search_r1.dynamic_sampling.loss_groups import grouped_masked_mean,grouped_policy_loss
from verl.trainer.ppo.core_algos import compute_policy_loss
from verl.utils.torch_functional import masked_mean

@pytest.mark.parametrize('group',[1,2])
def test_preserves_legacy_loss_and_gradient(group):
    torch.manual_seed(6)
    old=torch.randn(8,11,dtype=torch.float64)
    log=(old+.1*torch.randn_like(old)).requires_grad_()
    advantage=torch.randn_like(old)
    entropy=torch.randn_like(old,requires_grad=True)
    mask=(torch.arange(11)[None,:]<torch.tensor([1,10,3,8,4,6,2,9])[:,None]).double()
    baseline=sum(compute_policy_loss(old[i:i+group],log[i:i+group],advantage[i:i+group],mask[i:i+group],.2)[0]
        -.01*masked_mean(entropy[i:i+group],mask[i:i+group]) for i in range(0,8,group))/(8/group)
    combined=grouped_policy_loss(old,log,advantage,mask,.2,group)[0]-.01*grouped_masked_mean(entropy,mask,group)
    assert torch.allclose(baseline,combined,atol=1e-12)
    grads1=torch.autograd.grad(baseline,(log,entropy),retain_graph=True)
    grads2=torch.autograd.grad(combined,(log,entropy))
    assert all(torch.allclose(a,b,atol=1e-12) for a,b in zip(grads1,grads2))
    # A plain token mean would change the objective on unequal lengths.
    plain=compute_policy_loss(old,log,advantage,mask,.2)[0]-.01*masked_mean(entropy,mask)
    assert not torch.allclose(baseline,plain,atol=1e-6)
