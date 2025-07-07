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

import torch as t
import torch.distributed as dist
import datetime
import os
import subprocess
import argparse
import runpy
from ast import literal_eval
from matplotlib import pyplot as plt
import pickle
import numpy as np

DISTRIBUTED_PATH = os.path.dirname(os.path.abspath(__file__))

__all__ = ['sync_and_avg_gradients', 
           'run_single_to_multi_gpu',
           'wrap_single_gpu_script',
           'run_speed_test']


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

    If you want to use specific GPU IDs for reconstructions, you need to set up
    the environment variable `CDTOOLS_GPU_IDS` rather than `CUDA_VISIBLE_DEVICES`. 
    If you wanted to use GPU IDs `1, 3, 4` for example, write:
    
    ```
    CDTOOLS_GPU_IDS=1,3,4 cdt-torchrun --nnodes=1 --nproc_per_node=3 reconstruct.py
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

    # Set an environment variable for OMP_NUM_THREADS (sets number of threads)

    # Don't let the user die in anticipation
    print(f'\n[INFO]: Starting up multi-GPU reconstructions with {args.ngpus} GPUs.\n')

    # Set a random seed that all participaring workers will use
    seed = t.initial_seed()
    
    # Perform the torchrun call of the wrapped function
    subprocess.run(['torchrun', # We set up the torchrun arguments first
                    '--standalone', # Indicates that we're running a single machine, multiple GPU job.
                    f'--nnodes={args.nnodes}', 
                    f'--nproc_per_node={args.ngpus}', 
                    os.path.join(DISTRIBUTED_PATH,'single_to_multi_gpu.py'), # Make the call to the single-to-multi-gpu wrapper script
                    f'--backend={args.backend}',
                    f'--timeout={args.timeout}',
                    f'--nccl_p2p_disable={args.nccl_p2p_disable}',
                    f'--script_path={args.script_path}',
                    f'--seed={seed}'])
    
    # Let the user know the job is done
    print(f'\n[INFO]: Reconstructions complete.\n')


def wrap_single_gpu_script(script_path: str,
                           backend: str = 'nccl',
                           timeout: int = 30,
                           nccl_p2p_disable: bool = True,
                           seed: int = 0):
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

    If you want to use specific GPU IDs for reconstructions, you need to set up
    the environment variable `CDTOOLS_GPU_IDS` rather than `CUDA_VISIBLE_DEVICES`. 
    If you wanted to use GPU IDs `1, 3, 4` for example, write:
    
    ```
    CDTOOLS_GPU_IDS=1,3,4 torchrun --nnodes=1 --nproc_per_node=3 single_to_multi_gpu.py
    ```
    
    NOTE: For each subprocess `cdt-torchrun` creates, the environment variable
          `CUDA_VISIBLE_DEVICES` will be (re)defined as the GPU rank.

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
            Disable NCCL peer-2-peer communication. If you find that all your GPUs
            are at 100% useage but the program isn't doing anything, try enabling
            this variable.
        seed: int
            Seed for generating random numbers. This value must be identical across all
            participating devices.
            
    """
    ######################   Check if the script is safe to run  ######################
    ###################################################################################

    # 1) Check if the file path actually exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f'Cannot open file: {os.path.join(os.getcwd(), script_path)}')

    # 2) Check if the file is a CDTools reconstruction script.
    with open(script_path, 'r') as f:
        source_code = f.read()
    if not ('import cdtools' in source_code or 'from cdtools' in source_code):
        raise ValueError('File is not a CDTools reconstruction script (the script must import cdtools modules).')

    ##########   Force each subprocess to see only the GPU ID we assign it  ###########
    ###################################################################################

    # The GPU rank is visible as an environment variable through torchrun calls.
    rank = int(os.environ.get('RANK'))

    # If the CDTOOLS_GPU_IDS environment variable is defined, then assign based
    # on the GPU IDS provided in that list. Otherwise, use the rank for the GPU ID.
    gpu_ids = os.environ.get('CDTOOLS_GPU_IDS')
    
    if gpu_ids is None:
        gpu_id = rank
    else:
        gpu_id = literal_eval(gpu_ids)[rank]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    ################################  Run the script  #################################
    ###################################################################################
    
    # Enable/disable NCCL peer-to-peer communication. The boolean needs to be converted into
    # a string for the environment variable.
    os.environ['NCCL_P2P_DISABLE'] = str(int(nccl_p2p_disable))

    # Start up the process group (lets the different subprocesses can talk with each other)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=timeout))

    # Force this subprocess to use a given RNG seed (should be identical across all subprocesses)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
      
    try:     
        # Run the single-GPU reconstruction script 
        runpy.run_path(script_path, run_name='__main__')

    finally:
        # Kill the process group
        dist.destroy_process_group()   


def run_speed_test(world_sizes: int, 
                   runs: int,
                   script_path: str,
                   output_dir: str):
    """
    Executes a reconstruction script `n` x `m` times using `n` GPUs and `m` trials 
    per GPU count using cdt-torchrun.

    This function assumes that

    Parameters:
        world_sizes: list[int]
            Number of GPUs to use. User can specify several GPU counts in a list.
        runs: int
            How many repeat reconstructions to perform
        script_path: str
            Path of the single-gpu reconstruction script.
        output_dir: str
            Directory of the loss-vs-time/epoch data generated for the speed test.
    """
    # Set stuff up for plots
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)

    # Store the value of the single GPU time
    time_1gpu = 0
    std_1gpu = 0

    for world_size in world_sizes:
        # Get the GPU IDs to use
        #dev_id = device_ids[0:world_size] 
        #print(f'\nNumber of GPU(s): {world_size} | Using GPU IDs {*dev_id,}')

        # Make a list to store the values
        time_list = []
        loss_hist_list = []

        for i in range(runs):
            print(f'Resetting the model...')
            print(f'Starting run {i+1}/{runs} on {world_size} GPU(s)')

            # The scripts running speed tests need to read the trial number
            # they are on using an environment variable
            os.environ['CDTOOLS_TRIAL_NUMBER'] = str(i)
            
            # Run cdt-torchrun
            subprocess.run(['cdt-torchrun',
                            f'--ngpus={world_size}',
                            f'{script_path}'])

            print(f'[INFO]: Reconstruction complete. Loading loss results...')
            with open(os.path.join(output_dir, f'speed_test_nGPUs_{world_size}_TRIAL_{i}.pkl'), 'rb') as f:
                results = pickle.load(f)
            time_list.append(results['time history'])
            loss_hist_list.append(results['loss history'])

            
        # Calculate the statistics
        time_mean = np.array(time_list).mean(axis=0)/60
        time_std = np.array(time_list).std(axis=0)/60
        loss_mean = np.array(loss_hist_list).mean(axis=0)
        loss_std = np.array(loss_hist_list).std(axis=0)

        # If a single GPU is used, store the time
        if world_size == 1:
            time_1gpu = time_mean[-1]
            std_1gpu = time_std[-1]

        # Calculate the speed-up relative to using a single GPU
        speed_up_mean = time_1gpu / time_mean[-1] 
        speed_up_std = speed_up_mean * \
            np.sqrt((std_1gpu/time_1gpu)**2 + (time_std[-1]/time_mean[-1])**2)

        # Add another plot
        ax1.errorbar(time_mean, loss_mean, yerr=loss_std, xerr=time_std,
                    label=f'{world_size} GPUs')
        ax2.errorbar(np.arange(0,loss_mean.shape[0]), loss_mean, yerr=loss_std,
                    label=f'{world_size} GPUs')
        ax3.errorbar(world_size, speed_up_mean, yerr=speed_up_std, fmt='o')
        
    # Plot
    fig.suptitle(f'Multi-GPU performance test | {runs} runs performed')
    ax1.set_yscale('log')
    ax1.set_xscale('linear')
    ax2.set_yscale('log')
    ax2.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.set_xscale('linear')
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax3.set_xlabel('Number of GPUs')
    ax3.set_ylabel('Speed-up relative to single GPU')
    plt.show()
