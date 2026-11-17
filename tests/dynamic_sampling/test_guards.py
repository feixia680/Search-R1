import tempfile
import unittest
from types import SimpleNamespace
import torch
from unittest.mock import patch
from verl import DataProto
from verl.workers.dynamic_sampling_worker import attach_numerical_guards
from verl.workers.actor.dynamic_sampling_actor import GuardedDataParallelPPOActor


class ToyActor:
    def __init__(self):
        self.actor_module=torch.nn.Linear(2,1)
        self.actor_optimizer=torch.optim.SGD(self.actor_module.parameters(),lr=.1)
        self.config=SimpleNamespace(grad_clip=1.)
    def _forward_micro_batch(self,micro_batch,temperature):
        x=micro_batch['input_ids'].float()
        y=self.actor_module(x)
        return y,y


class GuardTests(unittest.TestCase):
    def test_finite_update_changes_parameters(self):
        actor=ToyActor();attach_numerical_guards(actor)
        before=actor.actor_module.weight.detach().clone()
        entropy,lp=actor._forward_micro_batch(dict(input_ids=torch.ones(2,2)),1.)
        lp.sum().backward();actor._optimizer_step()
        self.assertFalse(torch.equal(before,actor.actor_module.weight))

    def test_nonfinite_gradient_blocks_optimizer(self):
        actor=ToyActor()
        with tempfile.TemporaryDirectory() as path:
            attach_numerical_guards(actor,path)
            before=actor.actor_module.weight.detach().clone()
            for p in actor.actor_module.parameters():p.grad=torch.full_like(p,float('nan'))
            with self.assertRaises(FloatingPointError):actor._optimizer_step()
            torch.testing.assert_close(before,actor.actor_module.weight)

    def test_empty_loss_mask_fails_before_forward(self):
        actor=ToyActor();attach_numerical_guards(actor)
        with self.assertRaises(FloatingPointError):
            actor._forward_micro_batch(dict(input_ids=torch.ones(2,2),loss_mask=torch.zeros(2,1)),1.)

    def test_nonfinite_model_output_fails(self):
        actor=ToyActor();attach_numerical_guards(actor)
        with torch.no_grad():actor.actor_module.weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            actor._forward_micro_batch(dict(input_ids=torch.ones(2,2)),1.)

    def test_nonfinite_policy_loss_is_stopped_before_backward(self):
        actor=ToyActor()
        actor.config=SimpleNamespace(grad_clip=1.,ppo_mini_batch_size=2,ppo_micro_batch_size=2,
            state_masking=True,use_kl_loss=False,clip_ratio=.2,entropy_coeff=.001)
        actor._forward_micro_batch=lambda micro_batch,temperature: (
            actor.actor_module.weight[:, :1].expand(2,1),
            actor.actor_module.weight[:, :1].expand(2,1))
        d=DataProto.from_dict(dict(input_ids=torch.ones(2,2,dtype=torch.long),
            responses=torch.ones(2,1,dtype=torch.long),attention_mask=torch.ones(2,2),
            position_ids=torch.ones(2,2,dtype=torch.long),old_log_probs=torch.full((2,1),-1000.),
            advantages=-torch.ones(2,1),loss_mask=torch.ones(2,1)),meta_info={'temperature':1.})
        with patch.object(type(d.batch),'cuda',lambda self:self):
            with self.assertRaises(FloatingPointError):
                GuardedDataParallelPPOActor.update_policy(actor,d)
        self.assertTrue(all(p.grad is None for p in actor.actor_module.parameters()))


if __name__ == '__main__': unittest.main()
