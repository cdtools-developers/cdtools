"""This module contains the LBFGS Reconstructor subclass for performing
optimization ('reconstructions') on ptychographic/CDI models using
the LBFGS optimizer.

The Reconstructor class is designed to resemble so-called
'Trainer' classes that (in the language of the AI/ML folks) handles
the 'training' of a model given some dataset and optimizer.
"""
import torch as t
from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
from cdtools.models import CDIModel
from typing import Tuple, List, Union
from cdtools.reconstructors import Reconstructor

__all__ = ['LBFGS']

class LBFGS(Reconstructor):
    """
    The LBFGS Reconstructor subclass handles the optimization ('reconstruction') of 
    ptychographic models and datasets using the LBFGS optimizer.

    Parameters
    ----------
    model: CDIModel
        Model for CDI/ptychography reconstruction
    dataset: Ptycho2DDataset
        The dataset to reconstruct against
    subset : list(int) or int
        Optional, a pattern index or list of pattern indices to use
    schedule : bool
        Optional, create a learning rate scheduler (torch.optim.lr_scheduler._LRScheduler)

    Important attributes:
    - **model** -- Always points to the core model used.
    - **multi_gpu_used** -- Whether or not multi-GPU computation will be performed
      using a distributed data approach. This attribute will be pulled from the
      CDIModel (this flag is automatically set when using cdtools.tools.distributed.spawn).
    - **optimizer** -- This class by default uses `torch.optim.LBFGS` to perform
      optimizations.
    - **scheduler** -- A `torch.optim.lr_scheduler` that is defined during the `optimize` method.
    - **data_loader** -- A torch.utils.data.DataLoader that is defined by calling the 
      `setup_dataloader` method.
    """
    def __init__(self,
                 model: CDIModel,
                 dataset: Ptycho2DDataset,
                 subset: List[int] = None):

        super().__init__(model, dataset, subset)
        
        # Define the optimizer for use in this subclass
        self.optimizer = t.optim.LBFGS(self.model.parameters())
    
    def adjust_optimizer(self,
                         lr: int = 0.005,
                         history_size: int = 2,
                         line_search_fn: str = None):
        """
        Change hyperparameters for the utilized optimizer.

        Parameters
        ----------
        lr : float
            Optional, The learning rate (alpha) to use. Default is 0.005. 0.05 is 
            typically the highest possible value with any chance of being stable
        history_size : int
            Optional, the length of the history to use.
        line_search_fn : str
            Optional, either `strong_wolfe` or None
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['history_size'] = history_size
            param_group['line_search_fn'] = line_search_fn


    def optimize(self,
                 iterations: int,
                 lr: float = 0.1,
                 history_size: int = 2,
                 regularization_factor: Union[float, List[float]] = None,
                 thread: bool = True,
                 calculation_width: int = 10,
                 line_search_fn: str = None):
        """
        Runs a round of reconstruction using the LBFGS optimizer

        Formerly `CDIModel.LBFGS_optimize`
        
        This algorithm is often less stable that Adam, however in certain
        situations or geometries it can be shockingly efficient. Like all
        the other optimization routines, it is defined as a generator
        function which yields the average loss each epoch.

        NOTE: There is no batch size, because it is a usually a bad idea to use
        LBFGS on anything but all the data at onece

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        lr : float
            Optional, The learning rate (alpha) to use. Default is 0.1. 
        history_size : int
            Optional, the length of the history to use.
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass 
            the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow 
            interaction with plots during computation
        calculation_width : int
            Default 10, how many translations to pass through at once for each round of 
            gradient accumulation. Does not affect the result, only the calculation speed 
        shuffle : bool
            Optional, enable/disable shuffling of the dataset. This option
            is intended for diagnostic purposes and should be left as True.
        """
        # 1) The subset statement is contained in Reconstructor.__init__

        # 2) Set up / re-initialize the data loader. For LBFGS, we load
        #    all the data at once.
        self.setup_dataloader(batch_size=len(self.dataset))

        # 3) The optimizer is created in self.__init__, but the 
        #    hyperparameters need to be set up with self.adjust_optimizer
        self.adjust_optimizer(lr=lr, history_size=history_size, line_search_fn=line_search_fn)

        # 4) This is analagous to making a call to CDIModel.AD_optimize
        return super(LBFGS, self).optimize(iterations,
                                           regularization_factor,
                                           thread,
                                           calculation_width)