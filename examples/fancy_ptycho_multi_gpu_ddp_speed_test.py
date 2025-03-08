'''This is a testing script to study how the reconstruction speed
and convergence rate scales with the number of GPUs utilized.

The test is set up so that you can run n-trials for each number of GPUs
you want to study and plot statistics of loss-versus-time as a function
of GPU counts. 

This test is based on fancy_ptycho_multi_gpu_ddp.py and fancy_ptycho.py.

'''

import cdtools
from cdtools.models import CDIModel
from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
from cdtools.tools.distributed import distributed
from multiprocessing.connection import Connection
from typing import Tuple
from matplotlib import pyplot as plt
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
import datetime 
import time
import numpy as np
from copy import deepcopy

TIMEOUT = datetime.timedelta(seconds=10)   # Auto-terminate if things hang
BACKEND = 'nccl'

# Load the dataset
filename = r'examples/example_data/lab_ptycho_data.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# Create the model
model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3,
    oversampling=2, 
    probe_support_radius=120, 
    propagation_distance=5e-3,
    units='mm', 
    obj_view_crop=-50,
)

# Multi-GPU supported reconstruction
def reconstruct(model: CDIModel,
                dataset: Ptycho2DDataset,
                rank: int, 
                world_size: int,
                conn: Connection = None,
                schedule: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Perform the reconstruction using several GPUs
    If only one GPU is used, we don't bother loading the the process group
    or doing any of the fancy stuff associated with multi-GPU operation.

    Parameters:
        model: CDIModel
                Model for CDI/ptychography reconstruction
        dataset: Ptycho2DDataset
            The dataset to reconstruct against
        rank: int
            The rank of the GPU to be used. Value should be within
            [0, world_size-1]
        world_size: int
            The total number of GPUs to use
        conn: Connection
            A Connection object representing one end of a communication pipe. This
            parameter is needed if you're trying to get some values back from the
            wrapped function.
        schedule: bool
            Toggles the use of the scheduler
    
    Returns:
        time_history: np.array
            Array of when each loss was measured
        loss_history: np.array
            The total history of the model
    """
    # Create a list to keep track of when each module report was printed
    t_list = []
    # Start counting time
    t_start = time.time()

    if world_size == 1:
        device = 'cuda'
        model.to(device=device)
        dataset.get_as(device=device)


    # Perform reconstructions on either single or multi-GPU workflows.
    for loss in model.Adam_optimize(25, dataset, lr=0.02, batch_size=10, schedule=schedule):
        if rank == 0:
            print(model.report())
            t_list.append(time.time() - t_start)

    for loss in model.Adam_optimize(25, dataset, lr=0.005, batch_size=50, schedule=schedule):
        if rank == 0:
            print(model.report())
            t_list.append(time.time() - t_start)

    # We need to send the time_history and loss_history through
    # the child connection to the parent (sitting in the name-main block)
    if rank == 0:
        loss_history = np.array(model.loss_history)
        time_history = np.array(t_list)

        if conn is not None: 
            conn.send((time_history, loss_history))

    # Return the measured time and loss history if we're on a single GPU
    if world_size == 1: 
        return time_history, loss_history


# This will execute the multi_gpu_reconstruct upon running this file
if __name__ == '__main__':
    # Define the number of GPUs to use.
    world_sizes = [8, 4, 2, 1] 

    # Define if we want to use the scheduler or not
    schedule=True

    # Define how many iterations we want to perform of the reconstructions
    # for statistics
    runs = 5

    # Set up a parent/child connection to get some info from the GPU-accelerated function
    parent_conn, child_conn = mp.Pipe()

    # Write a try/except statement to help the subprocesses (and GPUs)
    # terminate gracefully. Otherwise, you may have stuff loaded on
    # several GPU even after terminating.
    for world_size in world_sizes:
        print(f'Number of GPU(s): {world_size}')
        # Make a list to store the values
        time_list = []
        loss_hist_list = []

        for i in range(runs):
            print(f'Resetting the model...')
            print(f'Starting run {i+1}/{runs} on {world_size} GPU(s)')
            model_copy = deepcopy(model)
            if world_size == 1:
                final_time, loss_history = reconstruct(model=model_copy, 
                                                        dataset=dataset,
                                                        rank=0,
                                                        world_size=1)
                time_list.append(final_time)
                loss_hist_list.append(loss_history)
            else:
                # Spawn the processes
                distributed.spawn(reconstruct,
                                    model=model_copy,
                                    dataset=dataset,
                                    world_size=world_size,
                                    master_addr = 'localhost',
                                    master_port = '8888',
                                    timeout=300,
                                    pipe=child_conn)
                while parent_conn.poll():
                    final_time, loss_history = parent_conn.recv()
                    time_list.append(final_time)
                    loss_hist_list.append(loss_history)
            
        
        # Calculate the statistics
        time_mean = np.array(time_list).mean(axis=0)/60
        time_std = np.array(time_list).std(axis=0)/60
        loss_mean = np.array(loss_hist_list).mean(axis=0)
        loss_std = np.array(loss_hist_list).std(axis=0)

        # Plot
        plt.errorbar(time_mean, loss_mean, yerr=loss_std, xerr=time_std,
                    label=f'{world_size} GPUs')
        plt.yscale('log')
        plt.xscale('linear')
        
        plt.legend()
        plt.xlabel('Time (min)')
        plt.ylabel('Loss')
        plt.show()
    
