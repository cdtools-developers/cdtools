"""This module contains the base Reconstructor class for performing
optimization ('reconstructions') on ptychographic/CDI models.

The Reconstructor class is designed to resemble so-called
'Trainer' classes that (in the language of the AI/ML folks) handles
the 'training' of a model given some dataset and optimizer.

The subclasses of Reconstructor are required to implement
their own data loaders and optimizer adjusters
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch as t
from torch.utils import data as td
import threading
import queue
import time
from typing import List, Union

if TYPE_CHECKING:
    from cdtools.models import CDIModel
    from cdtools.datasets import CDataset
    

__all__ = ['Reconstructor']


class Reconstructor:
    """
    Reconstructor handles the optimization ('reconstruction') of ptychographic
    models given a CDIModel (or subclass) and corresponding CDataset.

    This is a base model that defines all functions Reconstructor subclasses
    must implement.

    Parameters
    ----------
    model: CDIModel
        Model for CDI/ptychography reconstruction
    dataset: CDataset
        The dataset to reconstruct against
    optimizer: torch.optim.Optimizer
        The optimizer to use for the reconstruction
    subset : list(int) or int
        Optional, a pattern index or list of pattern indices to use

    Attributes
    ----------
    model : CDIModel
        Points to the core model used.
    optimizer : torch.optim.Optimizer
        Must be defined when initializing the Reconstructor subclass.
    scheduler : torch.optim.lr_scheduler, optional
        May be defined during the ``optimize`` method.
    data_loader : torch.utils.data.DataLoader
        Defined by calling the ``setup_dataloader`` method.
    """
    def __init__(self,
                 model: CDIModel,
                 dataset: CDataset,
                 optimizer: t.optim.Optimizer,
                 subset: Union[int, List[int]] = None):
        
        # Store parameters as attributes of Reconstructor
        self.model = model
        self.optimizer = optimizer

        # Store the dataset, clipping it to a subset if needed
        if subset is not None:
            # if subset is just one pattern, turn into a list for convenience
            if isinstance(subset, int):
                subset = [subset]
            dataset = td.Subset(dataset, subset)

        self.dataset = dataset

        # Initialize attributes that must be defined by the subclasses
        self.scheduler = None
        self.data_loader = None


    def setup_dataloader(self,
                         batch_size: int = None,
                         shuffle: bool = True):
        """
        Sets up or re-initializes the dataloader.

        Parameters
        ----------
        batch_size : int
            Optional, the size of the minibatches to use
        shuffle : bool
            Optional, enable/disable shuffling of the dataset. This option
            is intended for diagnostic purposes and should be left as True.
        """
        if batch_size is not None:
            self.data_loader = td.DataLoader(self.dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)
        else:
            self.data_loader = td.Dataloader(self.dataset)

            
    def adjust_optimizer(self, **kwargs):
        """
        Change hyperparameters for the utilized optimizer.

        For each optimizer, the keyword arguments should be manually defined
        as parameters.
        """
        raise NotImplementedError()

    
    def run_epoch(self,
                   stop_event: threading.Event = None,
                   regularization_factor: Union[float, List[float]] = None,
                   calculation_width: int = 10):
        """
        Runs one full epoch of the reconstruction. Intended to be called
        by Reconstructor.optimize.

        Parameters
        ----------
        stop_event : threading.Event
            Default None, causes the reconstruction to stop when an exception
            occurs in Optimizer.optimize.
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of
            parameters to pass the regularizer method
        calculation_width : int
            Default 10, how many translations to pass through at once for each
            round of gradient accumulation. This does not affect the result,
            but may affect the calculation speed.

        Returns
        ------
        loss : float
            The summed loss over the latest epoch, divided by the total
            diffraction pattern intensity
        """

        # Setting this as an explicit catch makes me feel more comfortable
        # exposing it as a public method. This way a user won't be confused
        # if they try to use this directly
        if self.data_loader is None:
            raise RuntimeError(
                'No data loader was defined. Please run '
                'Reconstructor.setup_dataloader() before running '
                'Reconstructor.run_epoch(), or use Reconstructor.optimize(), '
                'which does it automatically.'
            )
        
        
        # Initialize some tracking variables
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
                    if hasattr(self.model, 'mask'):
                        loss = self.model.loss(pats,
                                               sim_patterns,
                                               mask=self.model.mask)
                    else:
                        loss = self.model.loss(pats,
                                               sim_patterns)

                    # And accumulate the gradients
                    loss.backward()

                    # Normalize the accumulating total loss
                    total_loss += loss.detach()

                # If we have a regularizer, we can calculate it separately,
                # and the gradients will add to the minibatch gradient
                if regularization_factor is not None \
                        and hasattr(self.model, 'regularizer'):

                    loss = self.model.regularizer(regularization_factor)
                    loss.backward()

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

    def optimize(self,
                 iterations: int,
                 batch_size: int = 1,
                 custom_data_loader: torch.utils.data.DataLoader = None,
                 regularization_factor: Union[float, List[float]] = None,
                 thread: bool = True,
                 calculation_width: int = 10,
                 shuffle=True):
        """
        Runs a round of reconstruction using the provided optimizer

        Formerly CDIModel.AD_optimize

        This is the basic automatic differentiation reconstruction tool
        which all the other, algorithm-specific tools, use. It is a
        generator which yields the average loss each epoch, ending after
        the specified number of iterations.

        By default, the computation will be run in a separate thread. This
        is done to enable live plotting with matplotlib during a
        reconstruction.

        If the computation was done in the main thread, this would freeze
        the plots. This behavior can be turned off by setting the keyword
        argument 'thread' to False.

        The `batch_size` parameter sets the batch size for the default
        dataloader. If a custom data loader is desired, it can be passed
        in to the `custom_data_loader` argument, which will override the
        `batch_size` and `shuffle` parameters

        Please see `AdamReconstructor.optimize()` for an example of how to
        override this function when designing a subclass

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run.
        batch_size : int
            Optional, the batch size to use. Default is 1. This is typically
            overridden by subclasses with an appropriate default for the
            specific optimizer.
        custom_data_loader : torch.utils.data.DataLoader
            Optional, a custom DataLoader to use. Will override batch_size
            if set.
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of
            parameters to pass the regularizer method.
        thread : bool
            Default True, whether to run the computation in a separate thread
            to allow interaction with plots during computation.
        calculation_width : int
            Default 10, how many translations to pass through at once for each
            round of gradient accumulation. This does not affect the result,
            but may affect the calculation speed.
        shuffle : bool
            Optional, enable/disable shuffling of the dataset. This option
            is intended for diagnostic purposes and should be left as True.


        Yields
        ------
        loss : float
            The summed loss over the latest epoch, divided by the total
            diffraction pattern intensity.
        """

        if custom_data_loader is None:
            self.setup_dataloader(batch_size=batch_size, shuffle=shuffle)
        else:
            self.data_loader = custom_data_loader

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

                yield self.run_epoch(
                    regularization_factor=regularization_factor, # noqa
                    calculation_width=calculation_width,
                )

        # But if we do want to thread, it's annoying:
        else:
            # Here we set up the communication with the computation thread
            result_queue = queue.Queue()
            stop_event = threading.Event()

            def target():
                try:
                    result_queue.put(
                        self.run_epoch(
                            stop_event=stop_event,
                            regularization_factor=regularization_factor, # noqa
                            calculation_width=calculation_width,
                        )
                    )
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

                calc = threading.Thread(target=target,
                                        name='calculator',
                                        daemon=True)
                try:
                    calc.start()
                    while calc.is_alive():
                        if hasattr(self.model, 'figs'):
                            self.model.figs[0].canvas.start_event_loop(0.01)
                        else:
                            calc.join()

                except KeyboardInterrupt as e:
                    stop_event.set()
                    print('\nAsking execution thread to stop cleanly - ' +
                          'please be patient.')
                    calc.join()
                    raise e

                res = result_queue.get()

                # If something went wrong in the thead, we'll get an exception
                if isinstance(res, Exception):
                    raise res

                yield res

        # And finally, we unset the current optimizer:
        self.current_optimizer = None
