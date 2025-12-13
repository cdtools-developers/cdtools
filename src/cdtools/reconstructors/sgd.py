"""This module contains the SGDReconstructor subclass for performing
optimization ('reconstructions') on ptychographic/CDI models using
stochastic gradient descent.

The Reconstructor class is designed to resemble so-called
'Trainer' classes that (in the language of the AI/ML folks) handles
the 'training' of a model given some dataset and optimizer.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import torch as t
from typing import List, Union
from cdtools.reconstructors import Reconstructor

if TYPE_CHECKING:
    from cdtools.models import CDIModel
    from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
    

__all__ = ['SGDReconstructor']


class SGDReconstructor(Reconstructor):
    """
    The SGDReconstructor subclass handles the optimization ('reconstruction')
    of ptychographic models and datasets using the SGD optimizer.

    Parameters
    ----------
    model: CDIModel
        Model for CDI/ptychography reconstruction.
    dataset: Ptycho2DDataset
        The dataset to reconstruct against.
    subset : list(int) or int
        Optional, a pattern index or list of pattern indices to use.

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
                 subset: List[int] = None):

        # Define the optimizer for use in this subclass
        optimizer = t.optim.SGD(model.parameters())

        super().__init__(
            model,
            dataset,
            optimizer,
            subset=subset,
        )

        
    def adjust_optimizer(self,
                         lr: int = 0.005,
                         momentum: float = 0,
                         dampening: float = 0,
                         weight_decay: float = 0,
                         nesterov: bool = False):
        """
        Change hyperparameters for the utilized optimizer.

        Parameters
        ----------
        lr : float
            Optional, The learning rate (alpha) to use. Default is 0.005. 0.05
            is typically the highest possible value with any chance of being
            stable.
        momentum : float
            Optional, the length of the history to use.
        dampening : float
            Optional, dampening for the momentum.
        weight_decay : float
            Optional, weight decay (L2 penalty).
        nesterov : bool
            Optional, enables Nesterov momentum. Only applicable when momentum
            is non-zero.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['momentum'] = momentum
            param_group['dampening'] = dampening
            param_group['weight_decay'] = weight_decay
            param_group['nesterov'] = nesterov

    def optimize(self,
                 iterations: int,
                 batch_size: int = 15,
                 lr: float = 2e-7,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
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

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run.
        batch_size : int
            Optional, the size of the minibatches to use.
        lr : float
            Optional, The learning rate to use. The default is 2e-7.
        momentum : float
            Optional, the length of the history to use.
        dampening : float
            Optional, dampening for the momentum.
        weight_decay : float
            Optional, weight decay (L2 penalty).
        nesterov : bool
            Optional, enables Nesterov momentum. Only applicable when momentum
            is non-zero.
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

        # The optimizer is created in self.__init__, but the
        # hyperparameters need to be set up with self.adjust_optimizer
        self.adjust_optimizer(lr=lr,
                              momentum=momentum,
                              dampening=dampening,
                              weight_decay=weight_decay,
                              nesterov=nesterov)

        # Now, we run the optimize routine defined in the base class
        return super(SGDReconstructor, self).optimize(
            iterations,
            batch_size=batch_size,
            regularization_factor=regularization_factor,
            thread=thread,
            calculation_width=calculation_width,
        )
