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
from typing import TYPE_CHECKING, Tuple, Callable

import torch as t
import torch.distributed as dist
import random
import datetime
import os
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from cdtools.models import CDIModel

MIN_INT64 = t.iinfo(t.int64).min
MAX_INT64 = t.iinfo(t.int64).max


__all__ = ['get_launch_method',
           'get_rank',
           'get_world_size',
           'sync_and_avg_grads',
           'sync_rng_seed',
           'sync_loss',
           'sync_lr',
           'setup',
           'cleanup',
           'run_speed_test']


def get_launch_method() -> str:
    """
    Returns the method used to spawn the multi-GPU job.

    It is assumed that multi-GPU jobs will be created through
    one of two means: `torchrun` or `torch.multiprocessing.spawn`

    Returns:
        launch_method: str
            The method used to launch multi-GPU jobs. This parameter
            is either 'torchrun' or 'spawn'.
    """
    return 'torchrun' if 'TORCHELASTIC_RUN_ID' in os.environ else 'spawn'


def get_rank() -> int:
    """
    Returns the rank assigned to the current subprocess via the environment
    variable `RANK`. If this environment variable does not exist, a rank of 0
    will be returned.

    This value should range from 0 to `world_size`-1 (`world_size` being the
    total number of participating subprocesses/GPUs)

    Returns:
        rank: int
            Rank of the current subprocess.
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
    world_size = os.environ.get('WORLD_SIZE')
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


def sync_rng_seed(seed: int = None,
                  rank: int = None,):
    """
    Synchronizes the random number generator (RNG) seed used by all
    participating GPUs. Specifically, all subprocesses will use
    either Rank 0's RNG seed or the seed parameter value.

    Parameters:
        seed: int
            Optional. The random number generator seed.
        rank: int
            Optional, the rank of the current subprocess. If the multi-GPU
            job is created with torch.multiprocessing.spawn, this parameter
            should be explicitly defined.
    """
    if rank is None:
        rank = get_rank()

    if seed is None:
        seed_local = t.tensor(random.randint(MIN_INT64, MAX_INT64),
                              device=f'cuda:{rank}',
                              dtype=t.int64)
        dist.broadcast(seed_local, src=0)
        seed = seed_local.item()

    t.manual_seed(seed)


def sync_lr(optimizer: t.optim,
            rank: int = None):
    """
    Synchronizes the learning rate of all participating GPUs to that of
    Rank 0's GPU.

    Parameters:
        optimizer: t.optim
            Optimizer used for reconstructions.
        rank: int
            Optional, the rank of the current subprocess. If the multi-GPU
            job is created with torch.multiprocessing.spawn, this parameter
            should be explicitly defined.
    """
    if rank is None:
        rank = get_rank()

    for param_group in optimizer.param_groups:
        lr_tensor = t.tensor(param_group['lr'],
                             device=f'cuda:{rank}')
        dist.broadcast(lr_tensor, src=0)
        param_group['lr'] = lr_tensor.item()


def sync_loss(loss):
    """
    Synchronizes Rank 0 GPU's learning rate to all participating GPUs.

    Parameters:
        optimizer: t.optim
            Optimizer used for reconstructions.
    """
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)


def setup(rank: int = None,
          world_size: int = None,
          init_method: str = 'env://',
          master_addr: str = None,
          master_port: str = None,
          backend: str = 'nccl',
          timeout: int = 60,
          seed: int = None,
          verbose: bool = False):
    """
    Sets up the process group and envrionment variables needed for
    communications between the participating subprocesses. Also synchronizes
    the RNG seed used across all subprocesses. Currently, `torchrun` and
    `torch.multiprocessing.spawn` are supported for starting up multi-GPU jobs.

    This function blocks until all processes have joined.

    The following parameters need to be explicitly defined if the multi-GPU
    job is started up by `torch.multiprocessing.spawn`: `rank`, `world_size`,
    `master_addr`, `master_port`. If the multi-GPU job is started using
    `torchrun`, this function will use the environment variables `torchrun`
    provides to define the parameters discussed above.

    For additional details on defining the parameters see
    https://docs.pytorch.org/docs/stable/distributed.html for
    additional information.

    Parameters:
        rank: int
            Optional, the rank of the current subprocess.
        world_size: int
            Optional, the number of participating GPUs.
        master_addr: str
            Optional, address of the rank 0 node. If
        master_port: str
            Optional, free port of on the machine hosting rank 0.
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
        verbose: bool
            Optional. Shows messages indicating status of the setup procedure
    """
    # Make sure that the user explicitly defines parameters if spawn is used
    if get_launch_method() == 'spawn':
        if None in (rank, world_size, master_addr, master_port):
            raise RuntimeError(
                'torch.multiprocessing.spawn was detected as the launching ',
                'method, but either rank, world_size, master_addr, or ',
                'master_port has not been explicitly defined. Please ensure ',
                'these parameters have been explicitly defined, or ',
                'alternatively launch the multi-GPU job with torchrun.'
            )
        elif init_method == 'env://':
            # Set up the environment variables
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port

    if rank is None:
        rank = get_rank()
    if world_size is None:
        world_size = get_world_size()

    t.cuda.set_device(rank)
    if rank == 0:
        print('[INFO]: Initializing process group.')

    dist.init_process_group(rank=rank,
                            world_size=world_size,
                            backend=backend,
                            init_method=init_method,
                            timeout=datetime.timedelta(timeout))

    if rank == 0:
        print('[INFO]: Process group initialized.')

    sync_rng_seed(rank=rank, seed=seed)

    if rank == 0:
        print('[INFO]: RNG seed synchronized across all subprocesses.')


def cleanup():
    """
    Destroys the process group.
    """
    rank = get_rank()
    dist.destroy_process_group()
    if rank == 0:
        print('[INFO]: Process group destroyed.')


def run_speed_test(fn: Callable,
                   gpu_counts: Tuple[int],
                   runs: int = 3,
                   show_plot: bool = True):
    """
    Perform a speed test comparing the performance of multi-GPU reconstructions
    against single-GPU reconstructions. Multi-GPU jobs are created using
    `torch.multiprocessing.spawn`.

    The speed test is performed by calling a function-wrapped reconstruction
    script `runs` times using the number of GPUs specified in `gpu_counts`
    tuple. For each GPU count specified in `gpu_counts`, the average
    `model.loss_times` and `model.loss_history` values are accumulated
    across all the runs to calculate the mean and standard deviation of
    the loss-versus-time/epoch and speed-up-versus-GPUs curves.

    The function-wrapped reconstruction should use the following syntax:

    ```
    def reconstruct(rank, world_size, conn):
        cdtools.tools.distributed.setup(rank=rank, world_size=world_size)

        # Reconstruction script content goes here

        conn.send((model.loss_times, model.loss_history))
        cdtools.tools.distributed.cleanup()
    ```
    The parameters in this example function are defined internally by
    `run_speed_test`; the user does not need to define these within the script.
    `world_size` is the number of participating GPUs used. `rank` is an ID
    number given to a process that will run one of the `world_size` GPUs and
    varies in value from [0, `world_size`-1]. `conn.send` is a
    `multiprocessing.Connection.Pipe` that allows `run_speed_test` to
    retrieve data from the function-wrapped reconstruction.

    Parameters:
        fn: Callable
            The reconstruction script wrapped in a function. It is recommended
            to comment out all plotting and saving-related functions in the
            script to properly assess the multi-GPU performance.
        gpu_counts: Tuple[int]
            Number of GPUs to use for each test run. The first element must be
            1 (performance is compared against that of a single GPU).
        runs: int
            Number of repeat reconstructions to perform for each `gpu_counts`.
        show_plot: bool
            Optional, shows a plot of time/epoch-versus-loss and relative
            speedups that summarize the test results.

    Returns:
        loss_mean_list and loss_std_list: List[t.Tensor]
            The mean and standard deviation of the epoch/time-dependent losses.
            Each element corresponds with the GPU count used for that test.
        time_mean_list and time_std_list: List[t.Tensor]
            The mean and standard deviation of the epoch-dependent times.
            Each element corresponds with the GPU count used for that test.
        speed_up_mean_list and speed_up_std_list: List[float]
            The mean and standard deviation of the GPU-count-dependent
            speed ups. Each element corresponds with the GPU count used for
            that test.

    """
    # Make sure that the first element of gpu_counts is 1
    if gpu_counts[0] != 1:
        raise RuntimeError('The first element of gpu_counts needs to be 1.')

    # Set stuff up for plots
    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Store values of the different speed-up factors, losses, and times
    # as a function of GPU count
    loss_mean_list = []
    loss_std_list = []
    time_mean_list = []
    time_std_list = []
    speed_up_mean_list = []
    speed_up_std_list = []

    # Set up a parent/child connection to get the loss-versus-time
    # data from each GPU test run created by t.multiprocessing.spawn().
    parent_conn, child_conn = t.multiprocessing.Pipe()

    for gpus in gpu_counts:
        # Make a list to store the loss-versus-time values from each run
        time_list = []
        loss_hist_list = []

        for run in range(runs):
            # Spawn the multi-GPU or single-GPU job
            print('[INFO]: Starting run ',
                  f'{run+1}/{runs} on {gpus} GPU(s).')
            t.multiprocessing.spawn(fn,
                                    args=(gpus, child_conn),
                                    nprocs=gpus)

            # Get the loss-versus-time data
            while parent_conn.poll():
                loss_times, loss_history = parent_conn.recv()

            # Update the loss-versus-time data
            time_list.append(loss_times)
            loss_hist_list.append(loss_history)

        # Calculate the statistics over the runs performed
        loss_mean = t.tensor(loss_hist_list).mean(dim=0)
        loss_std = t.tensor(loss_hist_list).std(dim=0)
        time_mean = t.tensor(time_list).mean(dim=0)/60
        time_std = t.tensor(time_list).std(dim=0)/60

        if gpus == 1: # Assumes 1 GPU is used first in the test
            time_1gpu = time_mean[-1]
            std_1gpu = time_std[-1]

        # Calculate the speed-up relative to using a single GPU
        speed_up_mean = time_1gpu / time_mean[-1]
        speed_up_std = speed_up_mean * \
            t.sqrt((std_1gpu/time_1gpu)**2 + (time_std[-1]/time_mean[-1])**2)

        # Store the final loss-vs-time and speed-ups
        loss_mean_list.append(loss_mean)
        loss_std_list.append(loss_std)
        time_mean_list.append(time_mean)
        time_std_list.append(time_std)
        speed_up_mean_list.append(speed_up_mean.item())
        speed_up_std_list.append(speed_up_std.item())

        # Add another loss-versus-epoch/time curve
        if show_plot:
            ax1.errorbar(time_mean, loss_mean, yerr=loss_std, xerr=time_std,
                         label=f'{gpus} GPUs')
            ax2.errorbar(t.arange(0, loss_mean.shape[0]), loss_mean,
                         yerr=loss_std, label=f'{gpus} GPUs')
            ax3.errorbar(gpus, speed_up_mean, yerr=speed_up_std, fmt='o')

    print('[INFO]: Speed test completed.')

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
        plt.show()

    return loss_mean_list, loss_std_list, \
        time_mean_list, time_std_list, \
        speed_up_mean_list, speed_up_std_list
