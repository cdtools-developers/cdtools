'''This is a testing script to study how the reconstruction speed
and convergence rate scales with the number of GPUs utilized.

The test is set up so that you can run n-trials for each number of GPUs
you want to study and plot statistics of loss-versus-time as a function
of GPU counts. 

This test is based on fancy_ptycho_multi_gpu_ddp.py and fancy_ptycho.py.

'''

import cdtools
from typing import Tuple
from matplotlib import pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
import torch.multiprocessing as mp
import os
import datetime 
import time
import numpy as np

TIMEOUT = datetime.timedelta(seconds=10)   # Auto-terminate if things hang
BACKEND = 'nccl'


# Multi-GPU supported reconstruction
def multi_gpu_reconstruct(rank: int, 
                          world_size: int,
                          conn,
                          schedule=False) -> Tuple[np.ndarray, np.ndarray]:
    """Perform the reconstruction using several GPUs
    If only one GPU is used, we don't bother loading the the process group
    or doing any of the fancy stuff associated with multi-GPU operation.

    Parameters:
        rank: int
            The rank of the GPU to be used. Value should be within
            [0, world_size-1]
        world_size: int
            The total number of GPUs to use
        conn: mp.Pipe
            Connection to parent
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

    # Load the dataset
    filename = r'example_data/lab_ptycho_data.cxi'
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

    if world_size > 1:
        # We need to initialize the distributed process group
        # before calling any other method for multi-GPU usage
        init_process_group(backend=BACKEND,
                        rank=rank,
                        world_size=world_size,
                        timeout=TIMEOUT)
    
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

    # Assign devices
    device = f'cuda:{rank}'
    model.to(device=device)
    dataset.get_as(device=device)

    # Perform reconstructions on either single or multi-GPU workflows.
    if world_size > 1:
        # For multi-GPU workflows, we have to use this mess.
        model = DDP(model,
                    device_ids=[rank],  # Tells DDP which GPU the model lives in
                    output_device=rank, # Tells DDP which GPU to output to
                    find_unused_parameters=True) # TODO: Understand what this is really doing...
        barrier()

        for loss in model.module.Adam_optimize(50, 
                                            dataset, 
                                            lr=0.02, 
                                            batch_size=10,
                                            rank=rank,
                                            num_workers=world_size,
                                            schedule=schedule):
            if rank == 0:
                print(model.module.report())
                t_list.append(time.time() - t_start)
        barrier()

        for loss in model.module.Adam_optimize(50, 
                                            dataset,  
                                            lr=0.005, 
                                            batch_size=50,
                                            rank=rank,
                                            num_workers=world_size,
                                            schedule=schedule):
            if rank == 0:
                print(model.module.report())
                t_list.append(time.time() - t_start)
        # Again, set up another barrier to let all GPUs catch up
        barrier()
        # Always destroy the process group when you're done
        destroy_process_group()

        # We need to send the time_history and loss_history through
        # the child connection to the parent (sitting in the name-main block)
        if rank == 0:
            loss_history = np.array(model.module.loss_history)
            time_history = np.array(t_list)
            conn.send((time_history, loss_history))

    else:
        # For single-GPU workloads, we use the vanilla-way of performing
        # reconstructions in CDTools
        for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=10, schedule=schedule):
            print(model.report())
            t_list.append(time.time() - t_start)
        for loss in model.Adam_optimize(50, dataset,  lr=0.005, batch_size=50, schedule=schedule):
            print(model.report())
            t_list.append(time.time() - t_start)

        loss_history = np.array(model.loss_history)
        time_history = np.array(t_list)
        # Return the measured time and loss history
        return time_history, loss_history

# This will execute the multi_gpu_reconstruct upon running this file
if __name__ == '__main__':
    # We need to add some stuff to the enviromnent 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'  # You can use any open port number
    os.environ['NCCL_P2P_DISABLE'] = '1'

    # Set up a parent/child connection to get some info from the GPU-accelerated
    # function
    parent_conn, child_conn = mp.Pipe()

    # Define the number of GPUs to use.
    world_sizes = [2, 1] 

    # Define if we want to use the scheduler or not
    schedule=True

    # Define how many iterations we want to perform of the reconstructions
    # for statistics
    runs = 2

    # Write a try/except statement to help the subprocesses (and GPUs)
    # terminate gracefully. Otherwise, you may have stuff loaded on
    # several GPU even after terminating.
    try:
        for world_size in world_sizes:
            print(f'Number of GPU(s): {world_size}')
            # Make a list to store the values
            time_list = []
            loss_hist_list = []

            for i in range(runs):
                print(f'Starting run {i+1}/{runs} on {world_size} GPU(s)')
                if world_size == 1:
                    final_time, loss_history = multi_gpu_reconstruct(0, world_size,schedule)
                    time_list.append(final_time)
                    loss_hist_list.append(loss_history)
                else:
                    # Spawn the processes
                    mp.spawn(multi_gpu_reconstruct,
                             args=(world_size, child_conn, schedule),
                             nprocs=world_size,
                             join=True)
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


    except KeyboardInterrupt as e:
        # If something breaks, we try to make sure that the
        # process group is destroyed before the program fully
        # terminates
        print('Hang on a sec...')
        destroy_process_group()
    
