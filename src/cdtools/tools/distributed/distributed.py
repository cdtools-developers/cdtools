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
from __future__ import annotations
from typing import TYPE_CHECKING

import torch as t
import torch.distributed as dist
import random
import datetime
import os

if TYPE_CHECKING:
    from cdtools.models import CDIModel

MIN_INT64 = t.iinfo(t.int64).min
MAX_INT64 = t.iinfo(t.int64).max

__all__ = ['get_rank',
           'get_world_size',
           'sync_and_avg_grads',
           'sync_rng_seed',
           'broadcast_lr',
           'setup',
           'cleanup']


def get_rank() -> int:
    """
    Returns the GPU rank assigned to the subprocess via the environment
    variable `RANK`. If this environment variable does not exist, a
    rank of 0 will be returned.

    Returns:
        rank: int
            The integer ID assigned to the participating GPU.
    """
    rank = os.environ.get('RANK')
    return int(rank) if rank is not None else 0


def get_world_size() -> int:
    """
    Returns the world_size of the reconstruction job via the environment
    variable `WORLD_SIZE`. If this environment variable does not exist, a
    world_size of 1 will be returned.

    Returns:
        world_size: int
            The number of participating GPUs
    """
    world_size = int(os.environ.get('WORLD_SIZE'))
    return int(world_size) if world_size is not None else 1


def sync_and_avg_grads(model: CDIModel,
                       world_size: int):
    """
    Synchronizes the average of the model parameter gradients across all
    participating GPUs using all_reduce.

    Parameters:
        model: CDIModel
            Model for CDI/ptychography reconstruction
        world_size: int
            Number of participating GPUs
    """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


def sync_rng_seed(rank: int,
                  seed: int = None):
    """
    Synchronizes the random number generator (RNG) seed used by all
    participating GPUs. Specifically, all subprocesses will use
    either Rank 0's RNG seed or the seed parameter value.

    Parameters:
        rank: int
            The integer ID assigned to the participating GPU.
        seed: int
            Optional. The random number generator seed.
    """
    if seed is None:
        seed_local = t.tensor(random.randint(MIN_INT64, MAX_INT64),
                              device=f'cuda:{rank}',
                              dtype=t.int64)
        dist.broadcast(seed_local, src=0)
        seed = seed_local.item()

    t.manual_seed(seed)


def broadcast_lr(rank: int,
                 optimizer: t.optim):
    """
    Broadcast the Rank 0 GPU's learning rate to all participating GPUs.

    Parameters:
        rank: int
            The integer ID assigned to the participating GPU.
        optimizer: t.optim
            Optimizer used for reconstructions.
    """
    for param_group in optimizer.param_groups:
        lr_tensor = t.tensor(param_group['lr'],
                             device=f'cuda:{rank}')
        dist.broadcast(lr_tensor, src=0)
        param_group['lr'] = lr_tensor.item()


def setup(rank: int,
          world_size: int,
          init_method: str = 'env://',
          backend: str = 'nccl',
          timeout: int = 60,
          seed: int = None):
    """
    Sets up the process group needed for communications between
    the participating GPUs. Also synchronizes the RNG seed used
    across all

    This function blocks until all processes have joined.

    For additional details on defining the parameters see
    https://docs.pytorch.org/docs/stable/distributed.html for
    additional information.

    Parameters:
        rank: int
            The integer ID assigned to the participating GPU.
        world_size: int
            Number of participating GPUs minus 1.
        init_method: str
            URL specifying how to initialize the process group. 
            Default is “env://”.
        backend: str
            Multi-gpu communication backend to use. Default is the 'nccl'
            backend, which is the only supported backend for CDTools.
        timeout: int
            Timeout for operations executed against the process group in
            seconds. Default is 30 seconds. 
        seed: int
            Optional. The random number generator seed.
    """
    dist.init_process_group(backend=backend,
                            init_method=init_method,
                            timeout=datetime.timedelta(timeout),
                            rank=rank,
                            world_size=world_size)

    sync_rng_seed(rank=rank, seed=seed)


def cleanup():
    """
    Destroys the process group.
    """
    dist.destroy_process_group()
