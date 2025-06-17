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
import sys
import importlib
import subprocess
import inspect
import argparse
from multiprocessing.connection import Connection
from typing import Callable, List
from pathlib import Path

DISTRIBUTED_PATH = os.path.dirname(os.path.abspath(__file__))

__all__ = ['sync_and_avg_gradients', 
           'torchrunner',
           'run_single_to_multi_gpu',
           'wrap_single_gpu_script',
           '_spawn_wrapper',
           'spawn']


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


def torchrunner(script_name: str,
                n_gpus: int = 4):
    """
    Executes a torchrun command in a python script or jupyter notebook.

    Parameters:
        script_name: str
            The file name of the target script
        n_gpus: int
            Number of GPUs to distribute the job over
    """

    # Perform the torchrun call of the wrapped function
    subprocess.run(['torchrun', 
                    '--nnodes=1', 
                    f'--nproc_per_node={n_gpus}', 
                    f'{script_name}'])


def run_single_to_multi_gpu():
    """
    Runs a single-GPU reconstruction script as a multi-GPU job via torchrun.
    
    This function can be executed as `cdt-torchrun` in the command line.

    This function is a wrapper over both the single-GPU wrapping sc
    
    For example, if we have the reconstruction script `reconstruct.py` and want to use
    4 GPUs, we can write the following:

    ```
    cdt-torchrun --nproc_per_node=4 s script_path=reconstruct.py
    ```

    Arguments:
        script_path: str
            Path of the single-GPU script (either full or partial path).
        --ngpus: int
            Number of GPUs to use.
        --nnodes: int
            Optional, number of nodes. Default 1; more than 1 nodes has not been tested.
        --backend: str
            Optional, communication backend for distributed computing (either `nccl` or `gloo`).
            Default is `nccl`
        --timeout: int
            Optional, time in seconds before the distributed process is killed. 
            Default is 30 seconds.
        --nccl_p2p_disable: int
            Optional, disable (1) or enable (0) NCCL peer-to-peer communication. Default
            is 1.
        
    """
    # Define the arguments we need to pass to dist.script_wrapper
    parser = argparse.ArgumentParser()

    parser.add_argument('--ngpus',
                        type=int,
                        help='Number of GPUs to use (called --nproc_per_node in torchrun)')
    parser.add_argument('--nnodes', 
                        type=str, 
                        default=1,
                        help='Number of nodes participating in distributive computing.')
    parser.add_argument('--backend', 
                        type=str, 
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Communication backend (nccl or gloo)')
    parser.add_argument('--timeout', 
                        type=int, 
                        default=30,
                        help='Time before process is killed in seconds')
    parser.add_argument('--nccl_p2p_disable', 
                        type=int, 
                        default=1,
                        choices=[0,1],
                        help='Disable (1) or enable (0) NCCL peer-to-peer communication')
    parser.add_argument('script_path', 
                        type=str, 
                        help='Single GPU script file name (with or without .py extension)')
    
    # Get the arguments
    args = parser.parse_args()

    # Don't let the user die in anticipation
    print(f'\n[CDTools]: Starting up multi-GPU reconstructions with {args.ngpus} GPUs.\n')
    
    # Perform the torchrun call of the wrapped function
    subprocess.run(['torchrun', # We set up the torchrun arguments first
                    '--nnodes=1', 
                    f'--nproc_per_node={args.ngpus}', 
                    os.path.join(DISTRIBUTED_PATH,'single_to_multi_gpu.py'), # Make the call to the single-to-multi-gpu wrapper script
                    f'--backend={args.backend}',
                    f'--timeout={args.timeout}',
                    f'--nccl_p2p_disable={args.nccl_p2p_disable}',
                    f'--script_path={args.script_path}'])
    
    # Let the user know the job is done
    print(f'\n[CDTools]: Reconstructions complete.\n')


def wrap_single_gpu_script(script_path: str,
                           backend: str = 'nccl',
                           timeout: int = 30,
                           nccl_p2p_disable: bool = True):
    """
    Wraps single-GPU reconstruction scripts to be ran as a multi-GPU job via
    torchrun calls.

    This function is intended to be called in a script (say, single_to_multi_gpu.py) 
    with the following form:

    ```
    import cdtools.tools.distributed as dist
    if __name__ == '__main__':
        dist.torchrun_single_to_multi_gpu(**kwargs)
    ```
    
    torchrun should then be used to run this script as a distributive job using,
    for instance:
    
    ```
    torchrun --nnodes=1 --nproc_per_node=4 single_to_multi_gpu.py
    ```

    Parameters:
        script_name: str
            The file path of the single-GPU script (either full or relative).
            If you're using a relative path, make sure the string doesn't start
            with a backslash.
        backend: str
            Multi-gpu communication backend to use. Default is the 'nccl' backend,
            which is the only supported backend for CDTools.
            See https://pytorch.org/docs/stable/distributed.html for additional info
            about PyTorch-supported backends.
        timeout: int
            Timeout for operations executed against the process group in seconds. 
            Default is 30 seconds. After timeout has been reached, all subprocesses
            will be aborted and the process calling this method will crash. 
        nccl_p2p_disable: bool
            Disable NCCL peer-2-peer communication
    """
    # Check if the script path actually exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f'Cannot open file: {os.path.join(os.getcwd(), script_path)}')

    # Make sure that the script_name doesn't contain `.py` and share
    # the same name as any of the imported modules
    script_name = Path(script_path).stem
    if script_name in sys.modules:
        raise NameError(
            f'The file name {script_name} cannot share the same name as modules'
             ' imported in CDTools. Please change the script file name.')

    # Kill the process if it hangs/pauses for a certain amount of time.
    timeout = datetime.timedelta(seconds=timeout)

    # Enable/disable NVidia Collective Communications Library (NCCL)
    # peer-to-peer communication. If you find that all your GPUs are at 100% use 
    # but don't seem to be doing anything, try enabling this variable.
    os.environ['NCCL_P2P_DISABLE'] = str(int(nccl_p2p_disable))

    # If this script is called by torchrun, the GPU rank is visible as an 
    # environment variable.
    rank = int(os.environ.get('RANK'))

    # We need to prevent each subprocess from seeing GPUs other than the one it has 
    # been assigned by torchrun.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

    # Start up the process group (needed so the different subprocesses can talk with 
    # each other)
    dist.init_process_group(backend=backend,
                            timeout=timeout)
      
    try:     
        # Run the single-GPU reconstruction script by importing it using either full 
        # or partial paths to the script.

        # We need to create a specification for a module's import-system-related state
        spec = importlib.util.spec_from_file_location(script_name, script_path)

        # Next, we need to import the module from spec
        module = importlib.util.module_from_spec(spec)
        sys.modules[script_name] = module

        # As a safeguard against opening something other than a reconstruction
        # script, check if the script imports CDTools.
        source_code = inspect.getsource(module)
        if not ('import cdtools' in source_code or 'from cdtools' in source_code):
            raise ValueError('Only CDTools reconstruction scripts can be used with this method.')

        # Execute the script
        spec.loader.exec_module(module)
        #importlib.import_module(script_path)

    finally:
        # Kill the process group
        dist.destroy_process_group()   


def _spawn_wrapper(rank: int, 
                  func: Callable[[int, int], None], 
                  device_ids: List[int],
                  backend: str = 'nccl', 
                  timeout: int = 30,
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
            Default is 30 seconds. After timeout has been reached, all subprocesses
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
    
    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, 
                            world_size=world_size, timeout=timeout)
    
    try:
        # Run the reconstruction script
        # We also need to check if we want to pass a pipe to the function
        if pipe is None:
            func()    
        else:
            func(pipe)   
    finally:                 
        # Destroy process group
        dist.destroy_process_group()        


def spawn(func: Callable[[int, int], None],
          device_ids: List[int],
          master_addr: str,
          master_port: str,
          backend: str = 'nccl',
          timeout: int = 30,
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
            Default is 30 seconds. After timeout has been reached, all subprocesses
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
    mp.spawn(_spawn_wrapper,
             args=(func, device_ids, backend, timeout, pipe),
             nprocs=len(device_ids),
             join=True)
    print('Reconstructions complete...')
    