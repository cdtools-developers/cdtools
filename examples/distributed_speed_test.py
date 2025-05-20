'''This is a testing script to study how the reconstruction speed
and convergence rate scales with the number of GPUs utilized.

The test is set up so that you can run n-trials for each number of GPUs
you want to study and plot statistics of loss-versus-time as a function
of GPU counts. 

This test is based on fancy_ptycho_multi_gpu_ddp.py and fancy_ptycho.py.

'''

import cdtools
from cdtools.models import CDIModel
import cdtools.optimizer
from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
from cdtools.tools.distributed import distributed
import torch.multiprocessing as mp
import torch as t
from multiprocessing.connection import Connection
from typing import Tuple, List
from matplotlib import pyplot as plt
import time
import numpy as np
from copy import deepcopy

#The test to run:
#   'fancy_ptycho' - Runs reconstruction parameters from examples/fancy_ptycho.py
#   'gold_balls' - Runs reconstruction parameters from examples/gold_ball_ptycho.py
TEST = 'fancy_ptycho'

# Multi-GPU supported reconstruction
def reconstruct(model: CDIModel,
                dataset: Ptycho2DDataset,
                rank: int, 
                device_ids: List[int],
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
            [0, #_of_GPUs_used-1]
        device_ids: list[int]
            List of GPU IDs to use
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

    # Check if we're using only a single GPU
    if not model.multi_gpu_used:
        # Use the 1st GPU in device_ids
        device = f'cuda:{device_ids[0]}'
        model.to(device=device)
        dataset.get_as(device=device)

    # Set up the Reconstructor with the Adam optimizer
    recon = cdtools.optimizer.Adam(model,dataset)

    # Perform reconstructions on either single or multi-GPU workflows.
    if TEST == 'fancy_ptycho':
        for loss in recon.optimize(50, lr=0.02, batch_size=40):
            if rank == 0:
                print(model.report())
                t_list.append(time.time() - t_start)

        for loss in recon.optimize(25, lr=0.005, batch_size=40):
            if rank == 0:
                print(model.report())
                t_list.append(time.time() - t_start)

        for loss in recon.optimize(25, lr=0.001, batch_size=40):
            if rank == 0:
                print(model.report())
                t_list.append(time.time() - t_start)

    elif TEST == 'gold_balls':
        for loss in recon.optimize(20, lr=0.005, batch_size=50):
            if rank == 0: 
                print(model.report())
                t_list.append(time.time() - t_start)

        for loss in recon.optimize(50, lr=0.002, batch_size=100):
            if rank == 0:
                print(model.report())
                t_list.append(time.time() - t_start)

        for loss in recon.optimize(100, lr=0.001, batch_size=100):
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
    if not model.multi_gpu_used:
        return time_history, loss_history


def run_test(world_sizes: int, 
             device_ids: int, 
             runs: int):
    """Runs a series of reconstructions (defined in the local function 
    `reconstruct`) using several GPUs and several trials per GPU count.

    Parameters:
        world_sizes: list[int]
            Number of GPUs to use. User can specify several GPU counts in a list.
        device_ids: list[int] or int
            List of the GPU ID numbers to use for the study
        runs: int
            How many repeat reconstructions to perform
    """
    # Load the dataset and model
    if TEST == 'fancy_ptycho':
        filename = 'cdtools/examples/example_data/lab_ptycho_data.cxi'
    elif TEST == 'gold_balls':
        filename = 'cdtools/examples/example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'

    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)
    
    if TEST == 'fancy_ptycho':
        model = cdtools.models.FancyPtycho.from_dataset(
            dataset,
            n_modes=3,
            oversampling=2, 
            probe_support_radius=120, 
            propagation_distance=5e-3,
            units='mm', 
            obj_view_crop=-50)
        
    elif TEST == 'gold_balls':
        pad = 10
        dataset.pad(pad)
        model = cdtools.models.FancyPtycho.from_dataset(
            dataset,
            n_modes=3,
            probe_support_radius=50,
            propagation_distance=2e-6,
            units='um',
            probe_fourier_crop=pad)
        model.translation_offsets.data += 0.7 * t.randn_like(model.translation_offsets)
        model.weights.requires_grad = False

    # Set up a parent/child connection to get some info from the GPU-accelerated function
    parent_conn, child_conn = mp.Pipe()
    
    # Set stuff up for plots
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)

    # Store the value of the single GPU time
    time_1gpu = 0
    std_1gpu = 0

    for world_size in world_sizes:
        # Get the GPU IDs to use
        dev_id = device_ids[0:world_size] 
        print(f'\nNumber of GPU(s): {world_size} | Using GPU IDs {*dev_id,}')

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
                                                        device_ids=dev_id)
                time_list.append(final_time)
                loss_hist_list.append(loss_history)
            else:
                # Spawn the processes
                distributed.spawn(reconstruct,
                                    model=model_copy,
                                    dataset=dataset,
                                    device_ids = dev_id,
                                    master_addr = 'localhost',
                                    master_port = '8888',
                                    timeout=300,
                                    pipe=child_conn,
                                    nccl_p2p_disable=True)
                while parent_conn.poll():
                    final_time, loss_history = parent_conn.recv()
                    time_list.append(final_time)
                    loss_hist_list.append(loss_history)
            
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

# This will execute the multi_gpu_reconstruct upon running this file
if __name__ == '__main__':
    # Define the number of GPUs to use.
    world_sizes = [1, 2, 4] 

    # Define which GPU IDs to use
    device_ids = [1, 2, 5, 7]

    # How many reconstruction runs to perform for statistics
    runs = 3

    # Run the test
    run_test(world_sizes, device_ids, runs)
    