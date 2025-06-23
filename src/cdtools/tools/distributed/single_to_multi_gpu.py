"""
A wrapper script to run single-GPU reconstruction scripts as
a multi-GPU job when called by torchrun.

This script is intended to be called by torchrun. It is set
up so that the group process handling (init and destroy), 
definition of several environmental variables, and actual
execution of the single-GPU script are handled by a single 
call to dist.script_wrapper.

For example, if we have the reconstruction script `reconstruct.py` and want to use
4 GPUs, we can write the following:

```
torchrun --nnodes=1 --nproc_per_node=4 single-to-multi-gpu.py --script_path=reconstruct.py
```
"""
import cdtools.tools.distributed as dist
import argparse

def get_args():
    # Define the arguments we need to pass to dist.script_wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument('--script_path', 
                        type=str, 
                        help='Single GPU script file name (with or without .py extension)')
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
    
    return parser.parse_args()


if __name__ == '__main__':
    # Get args
    args = get_args()
    
    # Pass arguments to dist.script_wrapper
    dist.wrap_single_gpu_script(script_path=args.script_path,
                                backend=args.backend,
                                timeout=args.timeout,
                                nccl_p2p_disable=bool(args.nccl_p2p_disable))