"""Contains functions to make reconstruction scripts compatible
with multi-GPU distributive approaches in PyTorch.

The functions in this module require parts of the user-written
reconstruction script to be first wrapped in a function (as shown in 
examples/fancy_ptycho_multi_gpu_ddp.py). The functions in this module
are designed to wrap around/call these user-defined functions, enabling
reconstructions to be performed across several GPUs.

As of 20250302, the methods here are based on 
torch.nn.parallel.DistributedDataParallel, which implements distributed
data parallelism. In this scheme, replicas of the CDI/ptychography model
are given to each device. These devices will synchronize gradients across
each model replica. These methods however do not define how the Dataset is
distributed across each device; this process can be handled by using
DistributedSampler with the DataLoader.
"""

import numpy as np
import torch as t
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from multiprocessing.connection import Connection
from cdtools.models import CDIModel
from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
import datetime
import os
from typing import Callable

__all__ = ['distributed_wrapper', 'spawn']


def distributed_wrapper(rank: int, 
                        func: Callable[[CDIModel, Ptycho2DDataset, int, int], None], 
                        model: CDIModel, 
                        dataset: Ptycho2DDataset, 
                        world_size: int,
                        backend: str = 'nccl', 
                        timeout: int = 600,
                        pipe: Connection = None):
    """Wraps functions containing reconstruction loops (i.e., `for loss in 
    model.Adam_optimize`) to enable multi-GPU operations to be set up. The 
    wrapped function needs to passed to `torch.multiprocessing.spawn` or 
    `cdtools.tools.distributed.distributed.spawn`

    Parameters:
        rank: int
            Rank of the GPU, with value ranging from [0, world_size-1]. This
            is defined by the spawning methods and not directly by the user.
        func: Callable[[CDIModel, Ptycho2DDataset, int, int]]
            Function wrapping user-defined reconstruction loops. The function must
            have the following format: func(model, dataset, rank, world_size).
        model: CDIModel
            Model for CDI/ptychography reconstruction
        dataset: Ptycho2DDataset
            The dataset to reconstruct against
        world_size: int
            Number of GPUs to use
        backend: str
            Multi-gpu communication backend to use. Default is the 'nccl' backend,
            which is the only supported backend for CDTools.
            See https://pytorch.org/docs/stable/distributed.html for additional info
            about PyTorch-supported backends.
        timeout: int
            Timeout for operations executed against the process group in seconds. 
            Default is 10 minutes. After timeout has been reached, all subprocesses
            will be aborted and the process calling this method will crash. 
        pipe: Connection
            A Connection object representing one end of a communication pipe. This
            parameter is needed if you're trying to get some values back from the
            wrapped function.
            BUG: Passing a CDIModel through connection generated with mp.Pipe or
                 query will cause the connection to hang.
    """
    # Convert timeout from int to datetime
    timeout = datetime.timedelta(seconds=timeout)

    # Update the rank in the model and indicate we're using multiple GPUs
    model.rank = rank
    model.world_size = world_size
    if world_size > 1: # In case we need to use 1 GPU for testing
        model.multi_gpu_used = True

    # Initialize the process group
    init_process_group(backend=backend, rank=rank, 
                       world_size=world_size, timeout=timeout)
    
    # Load the model to the appropriate GPU rank the process is using
    device = f'cuda:{rank}'
    model.to(device=device)
    dataset.get_as(device=device) 

    # Wrap the model with DistributedDataParallel
    model_DDP = DDP(model,
                    device_ids=[rank],  # Tells DDP which GPU the model lives in
                    output_device=rank, # Tells DDP which GPU to output to
                    find_unused_parameters=True) # TODO: Understand what this is really doing...
    
    # Don't start reconstructing until all GPUs have synced.
    barrier()   
    # Start the reconstruction loop, but feed in model_DDP.module so we don't
    # have to change `model._` to `model.module._` in the CDTools script
    # We also need to check if we want to pass a pipe to the function
    if pipe is None:
        func(model_DDP.module, dataset, rank, world_size)    
    else:
        func(model_DDP.module, dataset, rank, world_size, pipe)   

    # Wait for all GPUs to finish reconstructing
    barrier()                               
    # Destroy process group
    destroy_process_group()        


def spawn(func: Callable[[CDIModel, Ptycho2DDataset, int, int], None],
          model: CDIModel,
          dataset: Ptycho2DDataset,
          world_size: int,
          master_addr: str,
          master_port: str,
          backend: str = 'nccl',
          timeout: int = 600,
          nccl_p2p_disable: bool = True,
          pipe: Connection = None):
    """Spawns subprocesses on `world_size` GPUs that runs reconstruction
    loops wrapped around a function `func`.
    
    This is a wrapper around `torch.multiprocessing.spawn` which includes 
    the setup of OS environmental variables needed for initializing the 
    distributed backend.

    Parameters:
        func: Callable[[CDIModel, Ptycho2DDataset, int, int]]
            Function wrapping user-defined reconstruction loops. The function must
            have the following format: func(model, dataset, rank, world_size).
        model: CDIModel
            Model for CDI/ptychography reconstruction
        dataset: Ptycho2DDataset
            The dataset to reconstruct against
        world_size: int
            Number of GPUs to use
        master_addr: str
            IP address of the machine that will host the process with rank 0
        master_port: str
            A free port on the machine that will host the process with rank 0
        backend: str
            Multi-gpu communication backend to use. Default is the 'nccl' backend,
            which is the only supported backend for CDTools.
            See https://pytorch.org/docs/stable/distributed.html for additional info
            about PyTorch-supported backends.
        timeout: int
            Timeout for operations executed against the process group in seconds. 
            Default is 10 minutes. After timeout has been reached, all subprocesses
            will be aborted and the process calling this method will crash.   
        nccl_p2p_disable: bool
            Disable NCCL peer-2-peer communication
        pipe: Connection
            A Connection object representing one end of a communication pipe. This
            parameter is needed if you're trying to get some values back from the
            wrapped function.
            BUG: Passing a CDIModel through connection generated with mp.Pipe or
                 query will cause the connection to hang.
    """
    # Set up environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['NCCL_P2P_DISABLE'] = str(int(nccl_p2p_disable))

    # Ensure a "graceful" termination of subprocesses if something goes wrong.
    print('\nStarting up multi-GPU reconstructions...')
    mp.spawn(distributed_wrapper,
                args=(func, model, dataset, world_size, backend, timeout, pipe),
                nprocs=world_size,
                join=True)
    print('Reconstructions complete...')
