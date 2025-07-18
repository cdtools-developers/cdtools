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
import datetime
import os
import subprocess
import argparse
import runpy
from ast import literal_eval
from matplotlib import pyplot as plt
import pickle
import random
from typing import Callable, Tuple, List
from pathlib import Path
from cdtools.models import CDIModel

DISTRIBUTED_PATH = os.path.dirname(os.path.abspath(__file__))
MIN_INT64 = t.iinfo(t.int64).min
MAX_INT64 = t.iinfo(t.int64).max

__all__ = ['sync_and_avg_gradients',
           'run_single_to_multi_gpu',
           'run_single_gpu_script',
           'report_speed_test',
           'run_speed_test']


def sync_and_avg_gradients(model: CDIModel):
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


def run_single_gpu_script(script_path: str,
                          backend: str = 'nccl',
                          timeout: int = 30,
                          nccl_p2p_disable: bool = True,
                          seed: int = None):
    """
    Wraps single-GPU reconstruction scripts to be ran as a multi-GPU job via
    torchrun calls.

    `run_single_gpu_script` is intended to be called within a script
    (e.g., cdtools.tools.distributed.single_to_multi_gpu) with the following
    form:

    ```
    # multi_gpu_job.py
    import cdtools.tools.distributed as dist
    if __name__ == '__main__':
        dist.run_single_to_multi_gpu(script_path='YOUR_RECONSTRUCTION_SCRIPT.py',
                                     backend='nccl',
                                     timeout=30,
                                     nccl_p2p_disable=True)
    ```

    `torchrun` should then be used to run this script as a single-node,
    multi-gpu job through the command line interface using, for instance:

    ```
    torchrun
        --standalone
        --nnodes=1
        --nproc_per_node=$nGPUs
        multi_gpu_job.py
    ```

    If you want to use specific GPU IDs for reconstructions, you need to set up
    the environment variable `CDTOOLS_GPU_IDS` rather than
    `CUDA_VISIBLE_DEVICES`. If you wanted to use GPU IDs `1, 3, 4` for example,
    write:

    ```
    CDTOOLS_GPU_IDS=1,3,4 torchrun
                            --standalone
                            --nnodes=1
                            --nproc_per_node=$nGPUs
                            multi_gpu_job.py
    ```

    `torchrun` will spawn a number of subprocesses equal to the number of GPUs
    specified (--nproc_per_node). Each subprocess will run the specified
    script (e.g., `multi_gpu_job` in the above example) which make a call to
    `run_single_gpu_script`.

    `run_single_gpu_script` will first set up process group (lets the
    different subprocesses and their respective GPUs communicate with each
    other) and environment variables necessary for multi-GPU jobs. Afterwards,
    each subprocess runs the single-GPU reconstruction script (e.g.,
    `YOUR_RECONSTRUCTION_SCRIPT.py` in the above example). Methods within
    the `cdtools.Reconstructor` class/subclasses handle gradient
    synchronization after backpropagation (loss.backward()) as well as
    distributive data shuffling/loading.


    NOTE:
        1) This method is indended to be called within a subprocess spawned
           by `torchrun`.
        2) For each subprocess `torchrun` creates, the environment variable
           `CUDA_VISIBLE_DEVICES` will be (re)defined based on the GPU rank
           or the GPU ID list if `CDTOOLS_GPU_IDS` is defined. The
           environment variable `NCCL_P2P_DISABLE` will also be (re)defined
           based on `nccl_p2p_disable`.
        3) This method has only been tested using the `nccl` backend on a
           single node, with `nccl_p2p_disable` set to `True`.

    Parameters:
        script_name: str
            The file path of the single-GPU script (either full or relative).
            If you're using a relative path, make sure the string doesn't start
            with a backslash.
        backend: str
            Multi-gpu communication backend to use. Default is the 'nccl'
            backend, which is the only supported backend for CDTools.
            See https://pytorch.org/docs/stable/distributed.html for
            additional info about PyTorch-supported backends.
        timeout: int
            Timeout for operations executed against the process group in
            seconds. Default is 30 seconds. After timeout has been reached,
            all subprocesses will be aborted and the process calling this
            method will crash.
        nccl_p2p_disable: bool
            Disable NCCL peer-2-peer communication. If you find that all your
            GPUs are at 100% usage but the program isn't doing anything, try
            enabling this variable.
        seed: int
            Seed for generating random numbers.

    Environment variables created/redefined:
        `NCCL_P2P_DISABLE`: Enables or disables NCCL peer-to-peer communication
            defined by `nccl_p2p_disable`.
        `CUDA_VISIBLE_DEVICES`: The GPU IDs visible to each subprocess. For
            each subprocess, this variable is set to the GPU ID the subprocess
            has been assigned.
    """

    # Check if the file path actually exists before starting the process group
    if not os.path.exists(script_path):
        raise FileNotFoundError('Cannot open file: ' +
                                f'{os.path.join(os.getcwd(), script_path)}')

    # Enable/disable NCCL peer-to-peer communication. The boolean needs to be
    # converted into a string for the environment variable.
    os.environ['NCCL_P2P_DISABLE'] = str(int(nccl_p2p_disable))

    """Force each subprocess to see only the GPU ID we assign it
    Why do this? If this constraint is not imposed, then calling all_reduce
    will cause all subprocess Ranks to occupy memory on both their own
    respective GPUs (normal) as well as Rank 0's GPU (not intended behavior).
    The root cause is not entirely clear but there are two ways to avoid
    this behavior empirically:
        1) Force each subprocesses' CUDA_VISIBLE_DEVICE to be their assigned
           GPU ids.
        2) Within the reconstruction script, change `device='cuda'` to
           `device=f'cuda{model.rank}'`

    Option 1 is chosen here to use single-GPU reconstruction scripts AS-IS
    for multi-GPU jobs.
    """
    # The GPU rank and world_size is visible as an environment variable
    # through torchrun calls.
    rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))

    # If the CDTOOLS_GPU_IDS environment variable is defined, then assign based
    # on the GPU IDS provided in that list. Otherwise, use the rank for the
    # GPU ID.
    gpu_ids = os.environ.get('CDTOOLS_GPU_IDS')

    if gpu_ids is None:
        gpu_id = rank
    else:
        gpu_id = literal_eval(gpu_ids)[rank]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if rank == 0:
        print('[INFO]: Starting up multi-GPU reconstructions with ' +
              f'{world_size} GPUs.')

    # Start up the process group (lets the different subprocesses talk with
    # each other)
    dist.init_process_group(backend=backend,
                            timeout=datetime.timedelta(seconds=timeout))

    # Run the script
    try:
        # Force all subprocesses to either use the pre-specified or Rank 0's
        # RNG seed
        if seed is None:
            seed_local = t.tensor(random.randint(MIN_INT64, MAX_INT64),
                                  device='cuda',
                                  dtype=t.int64)
            dist.broadcast(seed_local, 0)
            seed = seed_local.item()

        t.manual_seed(seed)

        runpy.run_path(script_path, run_name='__main__')

        if rank == 0:
            print('[INFO]: Reconstructions complete.' +
                  ' Terminating process group.')

    finally:
        dist.destroy_process_group()
        if rank == 0:
            print('[INFO]: Process group terminated. Multi-GPU job complete.')


def run_single_to_multi_gpu():
    """
    Runs a single-GPU reconstruction script as a single-node multi-GPU job via
    torchrun.

    This convienience function can be executed as a python console script as
    `cdt-torchrun` and serves as a wrapper over a `torchrun` call to
    `cdtools.tools.distributed.single_to_multi_gpu`.

    In the simplest case, a reconstruction script can be ran as a multi-GPU job
    (with `nGPU` number of GPUs) using the following `cdt-torchrun` call in
    the command line interface:
    ```
    cdt-torchrun
        --ngpus=<nGPUs>
        YOUR_RECONSTRUCTION_SCRIPT.py
    ```

    which is equivalent to the following `torchrun` call
    ```
    torchrun
        --standalone
        --nnodes=1
        --nproc_per_node=$nGPUs
        -m cdtools.tools.distributed.single_to_multi_gpu
        --backend=nccl
        --timeout=30
        --nccl_p2p_disable=1
        YOUR_RECONSTRUCTION_SCRIPT.py
    ```

    Within a single node, `cdt-torchrun` will launch a given number of
    subprocesses equivalent to the number of GPUs specified. This number must
    be less than or equal to the actual number of GPUs available on your node.

    If you want to use specific GPU IDs for reconstructions, you need to set up
    the environment variable `CDTOOLS_GPU_IDS` rather than
    `CUDA_VISIBLE_DEVICES`. If you wanted to use GPU IDs `1, 3, 4` for example,
    write:

    ```
    CDTOOLS_GPU_IDS=1,3,4 cdt-torchrun
                              --ngpus=3
                              YOUR_RECONSTRUCTION_SCRIPT.py
    ```

    If additional `torchrun` arguments need to be passed, consider making
    a direct `torchrun` call rather than use `cdt-torchrun`.

    NOTE:
        1) This method has only been tested using the `nccl` backend on a
           single node, with `nccl_p2p_disable` set to `True`.

    Arguments:
        script_path: str
            Path of the single-GPU script (either full or partial path).
        --ngpus: int
            Number of GPUs to use.
        --nnodes: int
            Optional, number of nodes. Default 1; more than 1 nodes has not
            been tested.
        --backend: str
            Optional, communication backend for distributed computing (either
            `nccl` or `gloo`).
            Default is `nccl`
        --timeout: int
            Optional, time in seconds before the distributed process is killed.
            Default is 30 seconds.
        --nccl_p2p_disable: int
            Optional, disable (1) or enable (0) NCCL peer-to-peer
            communication. Default is 1.

    """
    # Define the arguments we need to pass to dist.script_wrapper
    parser = argparse.ArgumentParser()

    parser.add_argument('--ngpus',
                        type=int,
                        help='Number of GPUs to use.')
    parser.add_argument('--nnodes',
                        type=str,
                        default=1,
                        help='Number of participating nodes.')
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
                        choices=[0, 1],
                        help='Disable (1) or enable (0) NCCL peer-to-peer' +
                             'communication')
    parser.add_argument('script_path',
                        type=str,
                        help='Single GPU script file name (with or without ' +
                        '.py extension)')

    # Get the arguments
    args = parser.parse_args()

    # Perform the torchrun call of the wrapped function
    subprocess.run(['torchrun',
                    '--standalone',
                    f'--nnodes={args.nnodes}',
                    f'--nproc_per_node={args.ngpus}',
                    '-m',
                    'cdtools.tools.distributed.single_to_multi_gpu',
                    f'--backend={args.backend}',
                    f'--timeout={args.timeout}',
                    f'--nccl_p2p_disable={args.nccl_p2p_disable}',
                    f'{args.script_path}'])


def report_speed_test(func: Callable):
    """
    Decorator function which saves the loss-versus-time/epoch history of a
    reconstruction script as a pickle dump file in a specified directory.

    The entire reconstruction script (excluding import statements) must
    be wrapped by a function which returns the model. The script must also
    have an if-name-main block to call the wrapped script.

    This decorator is intended to only be used by reconstruction scripts
    that are called by `run_speed_test` to conduct multi-GPU performance
    studies (loss-versus-time/epoch and runtime speed-ups) using `N` GPUs
    and `M` trials per GPU count. `run_speed_test` sets several environment
    variables specifing the name and directory of the result files to be saved.

    Parameters:
        func: Callable
            The entire reconstruction script wrapped in a function. Within the
            script, the function must be called with an if-name-main statement.
            Additionally, the function must return the reconstructed model.

    Expected environment variables:
        `CDTOOLS_TRIAL_NUMBER`: The test trial number
        `CDTOOLS_SPEED_TEST_RESULTS_DIR`: Directory to save the pickle dump
            file.
        `CDTOOLS_SPEED_TEST_PREFIX`: Prefix of the pickle dump file name.

    Outputs in the pickle dump file:
        study_dict: dict
            Results of the `N` GPU `M`-th trial run. Contains the following
            key-value pairs:
            `study_dict['loss history']`: List[np.float32]
                Loss values as a function of epoch
            `study_dict['time history']`: List[float]
                Time recorded at each epoch in seconds
            `study_dict['nGPUs']`: int
                Number of GPUs used
            `study_dict['trial']`: int
                Trial number
    """
    def wrapper():
        # Figure out how to name the save file and where to save it to
        # These environment variables are provided by run_speed_test
        trial_number = int(os.environ.get('CDTOOLS_TRIAL_NUMBER'))
        output_dir = os.environ.get('CDTOOLS_SPEED_TEST_RESULTS_DIR')
        file_prefix = os.environ.get('CDTOOLS_SPEED_TEST_PREFIX')

        # Run the script
        model = func()

        # Save the model and loss history, but only using the rank 0 process
        if model.rank == 0:
            # Set up the file name:
            file_name = f'{file_prefix}_nGPUs_{model.world_size}_' +\
                        f'TRIAL_{trial_number}.pkl'
            # Grab the loss and time history
            loss_history = model.loss_history
            time_history = model.loss_times

            # Store quantities in a dictionary
            study_dict = {'loss history': loss_history,
                          'time history': time_history,
                          'nGPUs': model.world_size,
                          'trial': trial_number}
            # Save the quantities
            with open(os.path.join(output_dir, file_name), 'wb') as save_file:
                pickle.dump(study_dict, save_file)

            print(f'[INFO]: Saved results to: {file_name}')
    return wrapper


def run_speed_test(world_sizes: List[int],
                   runs: int,
                   script_path: str,
                   output_dir: str,
                   file_prefix: str = 'speed_test',
                   show_plot: bool = True,
                   delete_output_files: bool = False,
                   nnodes: int = 1,
                   backend: str = 'nccl',
                   timeout: int = 30,
                   nccl_p2p_disable: bool = True,
                   seed: int = None
                   ) -> Tuple[List[float],
                              List[float],
                              List[float],
                              List[float]]:
    """
    Executes a reconstruction script using `world_sizes` GPUs and `runs`
    trials per GPU count using `torchrun` and
    `cdtools.tools.distributed.single_to_multi_gpu`.

    `run_speed_test` requires the tested reconstruction script to be wrapped
    in a function, which returns the reconstructed model, along with a
    if-name-main block which calls the function. The function needs to be
    decorated with `report_speed_test`.

    The speed test (specifically, `report_speed_test`) will generate pickle
    dump files named `<file_prefix>_nGPUs_<world_size>_TRIAL_<run number>.pkl`
    at the directory `output_dir` (see documentation for `report_speed_test`
    for the file content). If `output_dir` does not exist, one will be created
    in the current directory.

    After each trial, the contents of the dump file are read and stored by
    `run_speed_test` to calculate the mean and standard deviation of the
    loss-versus-epoch/time and runtime speedup data over the `runs` trials
    executed. If `delete_output_files` is enabled, then the pickle dump files
    will be deleted after they have been read.

    `run_speed_test` executes the following in a subprocess to run
    single/multi-GPU jobs
    ```
    torchrun
        --standalone
        --nnodes=$NNODES
        --nproc_per_node=$WORLD_SIZE
        -m
        cdtools.tools.distributed.single_to_multi_gpu
        --backend=$BACKEND
        --timeout=$TIMEOUT
        --nccl_p2p_disable=$NCCL_P2P_DISABLE
        YOUR_RECONSTRUCTION_SCRIPT.py
    ```
    and provides the following environment variables to the child environment
    that are necessary for the pickle dump files to be generated by
    `report_speed_test`:
        `CDTOOLS_TRIAL_NUMBER`: The test trial number
        `CDTOOLS_SPEED_TEST_RESULTS_DIR`: Directory to save the pickle dump
            file.
        `CDTOOLS_SPEED_TEST_PREFIX`: Prefix of the pickle dump file name.

    Parameters:
        world_sizes: List[int]
            Number of GPUs to use. User can specify several GPU counts in a
            list. But the first entry must be 1 (single-GPU).
        runs: int
            How many repeat reconstructions to perform
        script_path: str
            Path of the single-gpu reconstruction script.
        output_dir: str
            Directory of the loss-vs-time/epoch data generated for the speed
            test.
        file_prefix: str
            Prefix of the speed test result file names
        show_plot: bool
            Show loss-versus-epoch/time and speed-up-versus-GPU count curves
        delete_output_files: bool
            Removes the results files produced by `report_speed_test` from
            the output_dir after each trail run.
        nnodes: int
            Number of nodes to use. This module has only been tested with 1
            node.
        backend: str
            Communication backend for distributive computing. NVidia Collective
            Communications Library ('nccl') is the default and only tested
            option. See https://docs.pytorch.org/docs/stable/distributed.html
            for other backends supported by pytorch (but have not been tested
            in this package).
        timeout: int
            Timeout for operations to be executed in seconds. All processes
            will be aborted after the timeout has been exceeded.
        nccl_p2p_disable: bool
            Sets the `NCCL_P2P_DISABLE` environment variable to enable/disable
            nccl peer-to-peer communication. If you find that all your GPUs
            are at 100% usage but the program isn't doing anything, try
            enabling this variable.
        seed: int
            Seed for generating random numbers. Default is None (seed is
            randomly generated).

    Returns:
        final_loss_mean_list: List[float]
            Mean final loss value over `runs` iterations for each `world_size`
            value specified.
        final_loss_std_list: List[float]
            Standard deviation of the final loss value over `runs` iterations
            for each `world_size`.
        speed_up_mean_list: List[float]
            Mean runtime speed-up over `runs` iterations for each `world_size`
            value specified. Speed-up is defined as the
            `runtime_nGPUs / runtime_1_GPU`.
        speed_up_std_list: List[float]
            Standard deviation of the runtime speed-up over `runs` iterations
            for each `world_size`.
    """

    # Make sure the directory exists; or else create it
    Path(output_dir).mkdir(parents=False, exist_ok=True)

    # Set stuff up for plots
    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Store the value of the single GPU time
    time_1gpu = 0
    std_1gpu = 0

    # Store values of the different speed-up factors and final losses
    # as a function of GPU count
    speed_up_mean_list = []
    speed_up_std_list = []
    final_loss_mean_list = []
    final_loss_std_list = []

    for world_size in world_sizes:
        # Make a list to store the values
        time_list = []
        loss_hist_list = []

        for i in range(runs):
            print('[INFO]: Resetting the model...')
            print(f'[INFO]: Starting run {i+1}/{runs} on {world_size} GPU(s)')

            # The scripts running speed tests need to read the trial number
            # they are on using. We send this information using environment
            # variables sent to the child processes spawned by subprocess.run
            child_env = os.environ.copy()
            child_env['CDTOOLS_TRIAL_NUMBER'] = str(i)
            child_env['CDTOOLS_SPEED_TEST_RESULTS_DIR'] = output_dir
            child_env['CDTOOLS_SPEED_TEST_PREFIX'] = file_prefix

            # Set up the terminal commands for a single-node, multi-GPU job
            cmd = ['torchrun',
                   '--standalone',
                   f'--nnodes={nnodes}',
                   f'--nproc_per_node={world_size}',
                   '-m',
                   'cdtools.tools.distributed.single_to_multi_gpu',
                   f'--backend={backend}',
                   f'--timeout={timeout}',
                   f'--nccl_p2p_disable={int(nccl_p2p_disable)}']

            if seed is not None:
                cmd.append(f'--seed={seed}')

            cmd.append(f'{script_path}')

            # Run the single/multi-GPU job
            try:
                subprocess.run(cmd, check=True, env=child_env)

            except subprocess.CalledProcessError as e:
                raise e

            # Load the loss results
            print('[INFO]: Reconstruction complete. Loading loss results...')

            save_path = os.path.join(output_dir,
                                     f'{file_prefix}_nGPUs_{world_size}_' +
                                     f'TRIAL_{i}.pkl')

            with open(save_path, 'rb') as f:
                results = pickle.load(f)
            time_list.append(results['time history'])
            loss_hist_list.append(results['loss history'])

            print('[INFO]: Loss results loaded.')

            if delete_output_files:
                print(f'[INFO]: Removing {save_path}')
                os.remove(save_path)

        # Calculate the statistics
        time_mean = t.tensor(time_list).mean(dim=0)/60
        time_std = t.tensor(time_list).std(dim=0)/60
        loss_mean = t.tensor(loss_hist_list).mean(dim=0)
        loss_std = t.tensor(loss_hist_list).std(dim=0)

        # If a single GPU is used, store the time
        if world_size == 1:
            time_1gpu = time_mean[-1]
            std_1gpu = time_std[-1]

        # Calculate the speed-up relative to using a single GPU
        speed_up_mean = time_1gpu / time_mean[-1]
        speed_up_std = speed_up_mean * \
            t.sqrt((std_1gpu/time_1gpu)**2 + (time_std[-1]/time_mean[-1])**2)

        # Store the final lossess and speed-ups
        final_loss_mean_list.append(loss_mean[-1].item())
        final_loss_std_list.append(loss_std[-1].item())
        speed_up_mean_list.append(speed_up_mean.item())
        speed_up_std_list.append(speed_up_std.item())

        # Add another plot
        if show_plot:
            ax1.errorbar(time_mean, loss_mean, yerr=loss_std, xerr=time_std,
                         label=f'{world_size} GPUs')
            ax2.errorbar(t.arange(0, loss_mean.shape[0]), loss_mean,
                         yerr=loss_std, label=f'{world_size} GPUs')
            ax3.errorbar(world_size, speed_up_mean, yerr=speed_up_std, fmt='o')

    # Plot
    if show_plot:
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

    print('[INFO]: Multi-GPU speed test completed.')

    return (final_loss_mean_list, final_loss_std_list,
            speed_up_mean_list, speed_up_std_list)
