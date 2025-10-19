"""Contains functions to make reconstruction scripts compatible
with multi-GPU distributive approaches in PyTorch.

Multi-GPU computing here is based on distributed data parallelism, where
each GPU is given identical copies of a model and performs optimization
using different parts of the dataset. After the parameter gradients
are calculated (`loss.backwards()`) on each GPU, the gradients need to be
synchronized and averaged across all participating GPUs.

The functions in this module assist with gradient synchronization,
setting up conditions necessary to perform distributive computing, and
executing multi-GPU jobs.
"""
import torch as t
import torch.distributed as dist
import random
from cdtools.models import CDIModel

MIN_INT64 = t.iinfo(t.int64).min
MAX_INT64 = t.iinfo(t.int64).max

__all__ = ['sync_and_avg_grads',
           'sync_rng_seed']


def sync_and_avg_grads(model: CDIModel):
    """
    Synchronizes the average of the model parameter gradients across all
    participating GPUs using all_reduce.

    Parameters:
        model: CDIModel
            Model for CDI/ptychography reconstruction
    """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= model.world_size


def sync_rng_seed(seed: int = None):
    """
    Synchronizes the random number generator (RNG) seed used by all
    participating GPUs. Specifically, all subprocesses will use
    either Rank 0's RNG seed or the seed parameter value.

    Parameters:
        seed: int
            Optional. The random number generator seed.
    """
    if seed is None:
        seed_local = t.tensor(random.randint(MIN_INT64, MAX_INT64),
                              device='cuda',
                              dtype=t.int64)
        dist.broadcast(seed_local, 0)
        seed = seed_local.item()

    t.manual_seed(seed)
