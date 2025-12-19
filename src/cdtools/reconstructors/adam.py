"""This module contains the AdamReconstructor subclass for performing
optimization ('reconstructions') on ptychographic/CDI models using
the Adam optimizer.

The Reconstructor class is designed to resemble so-called
'Trainer' classes that (in the language of the AI/ML folks) handles
the 'training' of a model given some dataset and optimizer.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch as t
from typing import Tuple, List, Union
from cdtools.reconstructors import Reconstructor

if TYPE_CHECKING:
    from cdtools.models import CDIModel
    from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset

__all__ = ['AdamReconstructor']


class AdamReconstructor(Reconstructor):
    """
    The Adam Reconstructor subclass handles the optimization ('reconstruction')
    of ptychographic models and datasets using the Adam optimizer.

    Parameters
    ----------
    model: CDIModel
        Model for CDI/ptychography reconstruction.
    dataset: Ptycho2DDataset
        The dataset to reconstruct against.
    subset : list(int) or int
        Optional, a pattern index or list of pattern indices to use.
    rank : int
        Optional, GPU rank assigned during multi-GPU operations. If this
        parameter is None, it will be redefined based on the `RANK`
        environment variable. If this environment variable doesn't exist,
        single-GPU operation will be assumed and a rank of 0 will
        automatically be assigned.
    world_size : int
        Optional, the number of participating GPUs during multi-GPU
        operations. If this parameter is None, it will be redefined based on
        the `WORLD_SIZE` environment variable. If this environment variable
        doesn't exist,single-GPU operation will be assumed and a world_size of
        1 will automatically be assigned.

    Important attributes:
    - **model** -- Always points to the core model used.
    - **optimizer** -- This class by default uses `torch.optim.Adam` to perform
        optimizations.
    - **scheduler** -- A `torch.optim.lr_scheduler` that is defined during the
        `optimize` method.
    - **data_loader** -- A torch.utils.data.DataLoader that is defined by
        calling the `setup_dataloader` method.
    """
    def __init__(self,
                 model: CDIModel,
                 dataset: Ptycho2DDataset,
                 subset: List[int] = None,
                 rank: int = None,
                 world_size: int = None):

        # Define the optimizer for use in this subclass
        optimizer = t.optim.Adam(model.parameters())

        super().__init__(model, dataset, optimizer,
                         subset=subset, rank=rank, world_size=world_size)



    def adjust_optimizer(self,
                         lr: int = 0.005,
                         betas: Tuple[float] = (0.9, 0.999),
                         amsgrad: bool = False):
        """
        Change hyperparameters for the utilized optimizer.

        Parameters
        ----------
        lr : float
            Optional, The learning rate (alpha) to use. Default is 0.005. 0.05
            is typically the highest possible value with any chance of being
            stable.
        betas : tuple
            Optional, the beta_1 and beta_2 to use. Default is (0.9, 0.999).
        amsgrad : bool
            Optional, whether to use the AMSGrad variant of this algorithm.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['betas'] = betas
            param_group['amsgrad'] = amsgrad

        
    def optimize(self,
                 iterations: int,
                 batch_size: int = 15,
                 lr: float = 0.005,
                 betas: Tuple[float] = (0.9, 0.999),
                 custom_data_loader: t.utils.data.DataLoader = None,
                 schedule: bool = False,
                 amsgrad: bool = False,
                 regularization_factor: Union[float, List[float]] = None,
                 thread: bool = True,
                 calculation_width: int = 10,
                 shuffle: bool = True):
        """
        Runs a round of reconstruction using the Adam optimizer

        Formerly `CDIModel.Adam_optimize`

        This calls the Reconstructor.optimize superclass method
        (formerly `CDIModel.AD_optimize`) to run a round of reconstruction
        once the dataloader and optimizer hyperparameters have been
        set up.
        
        The `batch_size` parameter sets the batch size for the default
        dataloader. If a custom data loader is desired, it can be passed
        in to the `custom_data_loader` argument, which will override the
        `batch_size` and `shuffle` parameters


        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run.
        batch_size : int
            Optional, the size of the minibatches to use.
        lr : float
            Optional, The learning rate (alpha) to use. Default is 0.005. 0.05
            is typically the highest possible value with any chance of being
            stable.
        betas : tuple
            Optional, the beta_1 and beta_2 to use. Default is (0.9, 0.999).
        schedule : bool
            Optional, create a learning rate scheduler
            (torch.optim.lr_scheduler._LRScheduler).
        custom_data_loader : t.utils.data.DataLoader
            Optional, a custom DataLoader to use. If set, will override
            batch_size.
        amsgrad : bool
            Optional, whether to use the AMSGrad variant of this algorithm.
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of
            parameters to pass the regularizer method.
        thread : bool
            Default True, whether to run the computation in a separate thread
            to allow interaction with plots during computation.
        calculation_width : int
            Default 10, how many translations to pass through at once for each
            round of gradient accumulation. Does not affect the result, only
            the calculation speed.
        shuffle : bool
            Optional, enable/disable shuffling of the dataset. This option
            is intended for diagnostic purposes and should be left as True.
        """
        # Update the training history
        self.model.training_history += (
            f'Planning {iterations} epochs of Adam, with a learning rate = '
            f'{lr}, batch size = {batch_size}, regularization_factor = '
            f'{regularization_factor}, and schedule = {schedule}.\n'
        )

        # The optimizer is created in self.__init__, but the
        # hyperparameters need to be set up with self.adjust_optimizer
        self.adjust_optimizer(lr=lr,
                              betas=betas,
                              amsgrad=amsgrad)

        # Set up the scheduler
        if schedule:
            self.scheduler = \
                t.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                       factor=0.2,
                                                       threshold=1e-9)
        else:
            self.scheduler = None

        # Now, we run the optimize routine defined in the base class
        return super(AdamReconstructor, self).optimize(
            iterations,
            batch_size=batch_size,
            custom_data_loader=custom_data_loader,
            regularization_factor=regularization_factor,
            thread=thread,
            calculation_width=calculation_width,
            shuffle=shuffle,
        )
