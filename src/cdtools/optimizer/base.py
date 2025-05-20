"""This module contains the base Reconstructor class for performing
optimization ('reconstructions') on ptychographic/CDI models.

The Reconstructor class is designed to resemble so-called
'Trainer' classes that (in the language of the AI/ML folks) handles
the 'training' of a model given some dataset and optimizer.

The subclasses of Reconstructor are required to implement
their own data loaders and optimizer adjusters
"""

import torch as t
from torch.utils import data as torchdata
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import threading
import queue
import time
from contextlib import contextmanager
from cdtools.tools.data import nested_dict_to_h5, h5_to_nested_dict, nested_dict_to_numpy, nested_dict_to_torch
from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
from cdtools.models import CDIModel
import torch.distributed as dist
from typing import Tuple, List, Union

__all__ = ['Reconstructor']

class Reconstructor:
    """
    Reconstructor handles the optimization ('reconstruction') of ptychographic
    models given a CDIModel (or subclass) and corresponding Ptycho2DDataset.
    
    This is a base model that defines all functions Reconstructor subclasses
    must implement.

    Parameters
    ----------
    model: CDIModel
        Model for CDI/ptychography reconstruction
    dataset: Ptycho2DDataset
        The dataset to reconstruct against
    subset : list(int) or int
        Optional, a pattern index or list of pattern indices to use

    Important attributes:
    - **model** -- Always points to the core model used.
    - **multi_gpu_used** -- Whether or not multi-GPU computation will be performed
      using a distributed data approach. This attribute will be pulled from the
      CDIModel (this flag is automatically set when using cdtools.tools.distributed.spawn).
    - **optimizer** -- A `torch.optim.Optimizer` that must be defined when initializing the
      Reconstructor subclass.
    - **scheduler** -- A `torch.optim.lr_scheduler` that may be defined during the `optimize` method.
    - **data_loader** -- A torch.utils.data.DataLoader that is defined by calling the 
      `setup_dataloader` method.
    """
    def __init__(self,
                 model: CDIModel,
                 dataset: Ptycho2DDataset,
                 subset: List[int] = None):
        # Store parameters as attributes of Reconstructor
        self.subset = subset
        self.multi_gpu_used = model.multi_gpu_used
        self.world_size = model.world_size
        self.rank = model.rank

        # Initialize attributes that must be defined by the subclasses
        self.optimizer = None       # Defined in the __init__ of the subclass as a torch.optim.Optimizer
        self.scheduler = None       # Defined as a torch.optim.lr_scheduler
        self.data_loader = None     # Defined as a torch.utils.data.DataLoader in the setup_dataloader method
        
        # Store the original model
        # TODO: Include DDP support + wrapping
        self.model = model

        # Store the dataset
        if subset is not None:
            # if subset is just one pattern, turn into a list for convenience
            if type(subset) == type(1):
                subset = [subset]
            dataset = torchdata.Subset(dataset, subset)
        
        self.dataset = dataset

        
    def setup_dataloader(self, 
                         batch_size: int = None,
                         shuffle: bool = True):
        """
        Sets up / re-initializes the dataloader. 

        Parameters
        ----------
        batch_size : int
            Optional, the size of the minibatches to use
        shuffle : bool
            Optional, enable/disable shuffling of the dataset. This option
            is intended for diagnostic purposes and should be left as True.
        """
        if self.multi_gpu_used:
            # NOTE: Multi-GPU implementation is intended for use with Adam. All other subclasses
            # and optimizers have not been tested yet.

            # First, create a sampler to load subsets of dataset to the GPUs
            self.sampler = DistributedSampler(self.dataset,
                                              num_replicas=self.world_size,
                                              rank=self.rank,
                                              shuffle=shuffle,
                                              drop_last=False)
            # Now create the dataloader
            self.data_loader = torchdata.DataLoader(self.dataset,
                                                    batch_size=batch_size//self.world_size,
                                                    num_workers=0, # Creating extra threads in children processes may cause problems. Leave this at 0.
                                                    drop_last=False,
                                                    pin_memory=False,
                                                    sampler=self.sampler)
        else:
            if batch_size is not None:
                self.data_loader = torchdata.DataLoader(self.dataset,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)
            else:
                self.data_loader = torchdata.Dataloader(self.dataset)
    

    def adjust_optimizer(self, **kwargs):
        """
        Change hyperparameters for the utilized optimizer.

        For each optimizer, the keyword arguments should be manually defined as parameters.
        """
        raise NotImplementedError()


    def optimize(self,
                 iterations: int,
                 regularization_factor: Union[float, List[float]] = None,
                 thread: bool = True,
                 calculation_width: int = 10):
        """
        Runs a round of reconstruction using the provided optimizer
        
        Formerly CDIModel.AD_optimize

        This is the basic automatic differentiation reconstruction tool
        which all the other, algorithm-specific tools, use. It is a
        generator which yields the average loss each epoch, ending after
        the specified number of iterations.

        By default, the computation will be run in a separate thread. This
        is done to enable live plotting with matplotlib during a reconstruction.
        If the computation was done in the main thread, this would freeze
        the plots. This behavior can be turned off by setting the keyword
        argument 'thread' to False.        

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        calculation_width : int
            Default 10, how many translations to pass through at once for each round of gradient accumulation. This does not affect the result, but may affect the calculation speed.

        Yields
        ------
        loss : float
            The summed loss over the latest epoch, divided by the total diffraction pattern intensity
        """

        def run_epoch(stop_event=None):
            """Runs one full epoch of the reconstruction."""
            # If we're using DistributedSampler (likely the case if you're using 
            # multiple GPUs), we need to tell it which epoch we're on. Otherwise
            # data shuffling will not work properly
            if self.multi_gpu_used: 
                self.data_loader.sampler.set_epoch(self.model.epoch)

            # First, initialize some tracking variables
            normalization = 0
            loss = 0
            N = 0
            t0 = time.time()

            # The data loader is responsible for setting the minibatch
            # size, so each set is a minibatch
            for inputs, patterns in self.data_loader:
                normalization += t.sum(patterns).cpu().numpy()
                N += 1
                def closure():
                    self.optimizer.zero_grad()

                    # We further break up the minibatch into a set of chunks.
                    # This lets us use larger minibatches than can fit
                    # on the GPU at once, while still doing batch processing
                    # for efficiency
                    input_chunks = [[inp[i:i + calculation_width]
                                     for inp in inputs]
                                    for i in range(0, len(inputs[0]),
                                                   calculation_width)]
                    pattern_chunks = [patterns[i:i + calculation_width]
                                      for i in range(0, len(inputs[0]),
                                                     calculation_width)]

                    total_loss = 0
                    for inp, pats in zip(input_chunks, pattern_chunks):
                        # This check allows for graceful exit when threading
                        if stop_event is not None and stop_event.is_set():
                            exit()

                        # Run the simulation
                        sim_patterns = self.model.forward(*inp) 

                        # Calculate the loss
                        if hasattr(self, 'mask'):
                            loss = self.model.loss(pats,sim_patterns, mask=self.model.mask)
                        else:
                            loss = self.model.loss(pats,sim_patterns)

                        # And accumulate the gradients
                        loss.backward()

                        # For multi-GPU optimization, we need to average and
                        # sync the gradients + losses across all participating
                        # GPUs with an all-reduce call.               
                        if self.multi_gpu_used:
                            for param in self.model.parameters():
                                if param.requires_grad:
                                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) 
                                    param.grad.data /= self.model.world_size
                            
                            # Sum the loss value across all devices for reporting
                            dist.all_reduce(loss, op=dist.ReduceOp.SUM) 
                        # Normalize the accumulating total loss by the numer of GPUs used
                        total_loss += loss.detach() // self.model.world_size
                        

                    # If we have a regularizer, we can calculate it separately,
                    # and the gradients will add to the minibatch gradient
                    if regularization_factor is not None \
                       and hasattr(self.model, 'regularizer'):
                        loss = self.model.regularizer(regularization_factor)
                        loss.backward()

                        # For multi-GPU optimization, we need to average and
                        # sync the gradients + losses across all participating
                        # GPUs with an all-reduce call.
                        if self.multi_gpu_used:
                            for param in self.model.parameters():
                                if param.requires_grad:
                                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) 
                                    param.grad.data /= self.model.world_size
                    
                    return total_loss

                # This takes the step for this minibatch
                loss += self.optimizer.step(closure).detach().cpu().numpy()
            
            loss /= normalization

            # We step the scheduler after the full epoch
            if self.scheduler is not None:
                self.scheduler.step(loss)

            self.model.loss_history.append(loss)
            self.model.epoch = len(self.model.loss_history)
            self.model.latest_iteration_time = time.time() - t0
            self.model.training_history += self.model.report() + '\n'
            return loss

        # We store the current optimizer as a model parameter so that
        # it can be saved and loaded for checkpointing
        self.current_optimizer = self.optimizer
        
        # If we don't want to run in a different thread, this is easy
        if not thread:
            for it in range(iterations):
                if self.model.skip_computation():
                    self.epoch = self.epoch + 1
                    if len(self.model.loss_history) >= 1:
                        yield self.model.loss_history[-1]
                    else:
                        yield float('nan')
                    continue

                yield run_epoch()
                    
                
        # But if we do want to thread, it's annoying:
        else:
            # Here we set up the communication with the computation thread
            result_queue = queue.Queue()
            stop_event = threading.Event()
            def target():
                try:
                    result_queue.put(run_epoch(stop_event))
                except Exception as e:
                    # If something bad happens, put the exception into the
                    # result queue
                    result_queue.put(e)

            # And this actually starts and monitors the thread
            for it in range(iterations):
                if self.model.skip_computation():
                    self.model.epoch = self.model.epoch + 1                    
                    if len(self.model.loss_history) >= 1:
                        yield self.model.loss_history[-1]
                    else:
                        yield float('nan')
                    continue

                calc = threading.Thread(target=target, name='calculator', daemon=True)
                try:
                    calc.start()
                    while calc.is_alive():
                        if hasattr(self.model, 'figs'):
                            self.model.figs[0].canvas.start_event_loop(0.01)
                        else:
                            calc.join()

                except KeyboardInterrupt as e:
                    stop_event.set()
                    print('\nAsking execution thread to stop cleanly - please be patient.')
                    calc.join()
                    raise e

                res = result_queue.get()

                # If something went wrong in the thead, we'll get an exception
                if isinstance(res, Exception):
                    raise res

                yield res

        # And finally, we unset the current optimizer:
        self.current_optimizer = None