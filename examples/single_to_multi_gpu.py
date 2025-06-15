"""
A wrapper script intended to run single-GPU scripts as
a multi-GPU job when called by torchrun.

This script is intended to be called by torchrun. It is set
up so that the group process handling (init and destroy) and 
definition of several environmental variables are handled here. 
The reconstruction script of interest is called by simply
importing the name of the file (minus the .py extension).

Currently, this is the torchrun command I'm calling to use
4 GPUs:

torchrun --nnodes=1 --nproc_per_node=4 single_to_multi_gpu.py
"""
import os
import datetime
import torch.distributed as dist


if __name__ == '__main__':
    # Kill the process if it hangs/pauses for a certain amount
    # of time.
    timeout = datetime.timedelta(seconds=30)

    # Enable/disable NVidia Collective Communications Library (NCCL)
    # peer-to-peer communication. If you find that all your GPUs
    # are at 100% use but don't seem to be doing anything, try enabling
    # this variable.
    os.environ['NCCL_P2P_DISABLE'] = str(int(True))

    # If this script is called by torchrun, the GPU rank is
    # visible as an environment variable.
    rank = int(os.environ.get('RANK'))

    # We need to prevent each subprocess from seeing GPUs other
    # than the one it has been assigned by torchrun.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)

    # Start up the process group (needed so the different
    # subprocesses can talk with each other)
    dist.init_process_group(backend='nccl',
                            timeout=timeout)
      
    try:     
        # Run the single-GPU reconstruction script by importing it
        import fancy_ptycho

    finally:
        # Kill the process group
        dist.destroy_process_group()