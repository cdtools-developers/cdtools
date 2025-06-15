"""Contains functions to make reconstruction scripts compatible
with multi-GPU distributive approaches in PyTorch.

Multi-GPU computing here is based on distributed data parallelism, where
each GPU is given identical copies of a model and performs optimization
using different parts of the dataset. After the parameter gradients
are calculated (`loss.backwards()`) on each GPU, the gradients need to be
synchronized and averaged across all participating GPUs. 

The functions in this module assist with both gradient synchronization and
setting up conditions necessary to perform distributive computing. Some
functions in this module require parts of the user-written
reconstruction script to be first wrapped in a function (as shown in 
examples/fancy_ptycho_multi_gpu_ddp.py). The functions in this module
are designed to wrap around/call these user-defined functions, enabling
reconstructions to be performed across several GPUs.

NOTE: These methods however do not define how the Dataset is
distributed across each device; this process can be handled by using
DistributedSampler with the DataLoader.
"""

import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
import os
from multiprocessing.connection import Connection
from typing import Callable, List

__all__ = ['sync_and_avg_gradients', 'distributed_wrapper', 'spawn']

def sync_and_avg_gradients(model):
    """
    Synchronizes the average of the model parameter gradients across all
    participating GPUs.

    Parameters:
        model: CDIModel
            Model for CDI/ptychography reconstruction  
    """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) 
            param.grad.data /= model.world_size


def distributed_wrapper(rank: int, 
                        func: Callable[[int, int], None], 
                        device_ids: List[int],
                        backend: str = 'nccl', 
                        timeout: int = 600,
                        pipe: Connection = None):
    """
    Wraps functions containing reconstruction loops (i.e., `for loss in 
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
        device_ids: list[int]
            List of GPU IDs to use
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

    # Define the world_size
    world_size = len(device_ids)

    # Update the rank in the model and indicate we're using multiple GPUs
    #model.rank = rank
    #model.device_id = device_ids[model.rank]
    #model.world_size = len(device_ids)

    # Allow the process to only see the GPU is has been assigned
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank) 

    # Within the called reconstruction function/script, we need to somehow
    # set up the multi-GPU model flags (model.rank, model.world_size,
    # and model.multi_gpu_used).
    #
    # One way to do this (without having to modify CDIModel here or explicitly
    # setting up the CDIModel attributes in the reconstruction script) is to
    # create environment variables for each subprocess. Then, when a model
    # is created within each subprocess, it can loop up its own local environment
    # variable and set the rank/world_size/multi_gpu_used flags accordingly.
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['NCCL_P2P_DISABLE'] = str(int(True))
    
    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, 
                            world_size=world_size, timeout=timeout)
    
    # Run the reconstruction script
    # We also need to check if we want to pass a pipe to the function
    if pipe is None:
        func()    
    else:
        func(pipe)   
                         
    # Destroy process group
    dist.destroy_process_group()        


def spawn(func: Callable[[int, int], None],
          device_ids: List[int],
          master_addr: str,
          master_port: str,
          backend: str = 'nccl',
          timeout: int = 600,
          nccl_p2p_disable: bool = True,
          pipe: Connection = None):
    """
    Spawns subprocesses on `world_size` GPUs that runs reconstruction
    loops wrapped around a function `func`.
    
    This is a wrapper around `torch.multiprocessing.spawn` which includes 
    the setup of OS environmental variables needed for initializing the 
    distributed backend.

    Parameters:
        func: Callable[[CDIModel, Ptycho2DDataset, int, int]]
            Function wrapping user-defined reconstruction loops. The function must
            have the following format: func(model, dataset, rank, world_size).
        device_ids: list[int]
            List of GPU IDs to use
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
    print('\nStarting up multi-GPU reconstructions...\n')
    mp.spawn(distributed_wrapper,
                args=(func, device_ids, backend, timeout, pipe),
                nprocs=len(device_ids),
                join=True)
    print('Reconstructions complete...')
    