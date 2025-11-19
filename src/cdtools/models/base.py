"""This module contains the base CDIModel class for CDI Models.

The subclasses of the main CDIModel class are required to define their
own implementations of the following functions:

Loading and Saving
------------------
from_dataset
    Creates a CDIModel from an appropriate CDataset
simulate_to_dataset
    Creates a CDataset from the simulation defined in the model
save_results
    Saves out a dictionary with the recovered parameters


Simulation
----------
interaction
    Simulates exit waves from experimental parameters
forward_propagator
    The propagator from the experiment plane to the detector plane
backward_propagator
    Optional, the propagator from the detector plane to the experiment plane
measurement
    Simulates the detector readout from a detector plane wavefront
loss
    the loss function to report and use for automatic differentiation

"""

import torch as t
from torch.utils import data as torchdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker
import numpy as np
import threading
import queue
import time
from scipy import io
from contextlib import contextmanager
from cdtools.tools.data import nested_dict_to_h5, h5_to_nested_dict, nested_dict_to_numpy, nested_dict_to_torch
from cdtools.reconstructors import AdamReconstructor, LBFGSReconstructor, SGDReconstructor
from cdtools.datasets import CDataset
from typing import List, Union, Tuple
import os

__all__ = ['CDIModel']


class CDIModel(t.nn.Module):
    """This base model defines all the functions that must be exposed for a valid CDIModel subclass

    Most of the functions only raise a NotImplementedError at this level and
    must be explicitly defined by any subclass - these are noted explocitly
    in the module-level intro. The work of defining the various subclasses
    boils down to creating an appropriate implementation for this set of
    functions.
    """

    def __init__(self):
        super(CDIModel, self).__init__()

        self.loss_history = []
        self.training_history = ''
        self.epoch = 0

    def from_dataset(self, dataset):
        raise NotImplementedError()


    def interaction(self, *args):
        raise NotImplementedError()


    def forward_propagator(self, exit_wave):
        raise NotImplementedError()


    def backward_propagator(self, detector_wave):
        raise NotImplementedError()


    def measurement(self, detector_wave):
        raise NotImplementedError()


    def forward(self, *args):
        """The complete forward model

        This model relies on composing the interaction, forward propagator,
        and measurement functions which are required to be defined by all
        subclasses. It therefore should not be redefined by the subclasses.

        The arguments to this function, for any given subclass, will be
        the same as the arguments to the interaction function.
        """
        return self.measurement(self.forward_propagator(self.interaction(*args)))

    def loss(self, sim_data, real_data):
        raise NotImplementedError()

    
    def simulate_to_dataset(self, args_list):
        raise NotImplementedError()

    
    def store_detector_geometry(self, detector_geometry, dtype=t.float32):
        """Registers the information in a detector geometry dictionary

        Information about the detector geometry is passed in as a dictionary,
        but we want the various properties to be registered as buffers in
        the model. This has nice effects, for example automatically updating
        with model.to, and making it possible to automatically save them out.

        Parameters
        ----------
        detector_geometry : dict
            A dictionary containing at least the two entries 'distance' and 'basis'
        dtype : torch.dtype, default: torch.float32
            The datatype to convert the values to before registering
        """
        self.register_buffer('det_basis',
                             t.as_tensor(detector_geometry['basis'],
                                      dtype=dtype))
        
        if 'distance' in detector_geometry \
           and detector_geometry['distance'] is not None:                
            self.register_buffer('det_distance',
                                 t.as_tensor(detector_geometry['distance'],
                                          dtype=dtype))
        if 'corner' in detector_geometry \
           and detector_geometry['corner'] is not None:
            self.register_buffer('det_corner',
                                 t.as_tensor(detector_geometry['corner'],
                                          dtype=dtype))

    def get_detector_geometry(self):
        """Makes a detector geometry dictionary from the registered buffers

        This extracts a dictionary with the detector geometry data from
        the registered buffers, helpful for functions which expect the
        geometry data to be in this format.

        Returns
        -------
        detector_geometry : dict
            A dictionary containing at least the two entries 'distance' and 'basis', pulled from the model's buffers
        """
        detector_geometry = {}
        if hasattr(self, 'det_distance'):
            detector_geometry['distance'] = self.det_distance
        if hasattr(self, 'det_basis'):
            detector_geometry['basis'] = self.det_basis
        if hasattr(self, 'det_corner'):
            detector_geometry['corner'] = self.det_corner
        return detector_geometry    
    
    def save_results(self):
        """A convenience function to get the state dict as numpy arrays

        This function exists for two reasons, even though it is just a thin
        wrapper on top of t.module.state_dict(). First, because the model
        parameters for Automatic Differentiation ptychography and
        related CDI methods *are* the results, it's nice to explicitly
        recognize the role of extracting the state_dict as saving the
        results of the reconstruction

        Second, because display, further processing, long-term storage,
        etc. are often done with dictionaries of numpy arrays. So, it's useful
        to have a convenience function which does that conversion
        automatically.
        
        Returns
        -------
        results : dict
            A dictionary containing all the parameters and buffers of the model, i.e. the result of self.state_dict(), converted to numpy.
        """
        state_dict = nested_dict_to_numpy(self.state_dict())

        return {
            'state_dict': state_dict,
            'loss_history': np.array(self.loss_history),
            'epoch': self.epoch,
            'training_history': self.training_history,
            'loss_function': self.loss.__name__,
        }


    def save_to_h5(self, filename, *args):
        """Saves the results to a .mat file

        Parameters
        ----------
        filename : str
            The filename to save under
        *args
            Accepts any additional args that model.save_results needs, for this model
        """
        return nested_dict_to_h5(filename, self.save_results(*args))
    

    @contextmanager
    def save_on_exit(self, filename, *args, exception_filename=None):
        """Saves the results of the model when the context is exited

        If you wrap the main body of your code in this context manager,
        it will either save the results to a .h5 file upon completion,
        or when any exception is raised during execution.

        Parameters
        ----------
        filename : str
            The filename to save under, upon completion
        *args
            Accepts any additional args that model.save_results needs, for this model
        exception_filename : str
            Optional, a separate filename to use if an exception is raised during execution. Default is equal to filename
        """
        try:
            yield
            self.save_to_h5(filename, *args)
        except:
            if exception_filename is None:
                exception_filename = filename
            self.save_to_h5(exception_filename, *args)
            raise

    @contextmanager
    def save_on_exception(self, filename, *args):
        """Saves the results of the model if an exception occurs

        If you wrap the main body of your code in this context manager,
        it will save the results to a .h5 file if an exception is thrown.
        If the code completes without an exception, it will not save the
        results, expecting that the results are explicitly saved later

        Parameters
        ----------
        filename : str
            The filename to save under, in case of an exception
        *args
            Accepts any additional args that model.save_results needs, for this model
        """
        try:
            yield
        except:
            self.save_to_h5(filename, *args)
            print('Intermediate results saved under name:')
            print(filename, flush=True)
            raise


    def use_checkpoints(self, job_id, checkpoint_file_stem):
        self.current_checkpoint_id = 0
        self.job_id = job_id
        self.checkpoint_file_stem = checkpoint_file_stem

    def skip_computation(self):
        """Returns true if computations should be skipped due to checkpointing

        This is used internally by model.AD_optimize to make the checkpointing
        system work, but it is also useful to suppress printing when
        computations are being skipped
        """
        if (hasattr(self, 'current_checkpoint_id')
            and self.current_checkpoint_id != self.job_id):
            return True
        else:
            return False

    def save_checkpoint(self, *args, checkpoint_file=None):
        checkpoint = self.save_results(*args)
        if (hasattr(self, 'current_optimizer')
            and self.current_optimizer is not None):
            checkpoint['optimizer_state_dict'] = \
            self.current_optimizer.state_dict()

        if checkpoint_file is None:
            checkpoint_file = (
                self.checkpoint_file_stem
                + '_' + str(self.current_checkpoint_id) + '.pt'
            )

        t.save(checkpoint, checkpoint_file)
        #nested_dict_to_h5(checkpoint_file, checkpoint)
            
    
    def load_checkpoint(self, checkpoint_file=None):
        if checkpoint_file is None:
            checkpoint_file = (
                self.checkpoint_file_stem
                + '_' + str(self.current_checkpoint_id) + '.pt'
            )

        checkpoint = t.load(checkpoint_file)
        
        state_dict = nested_dict_to_torch(checkpoint['state_dict'])
        self.load_state_dict(state_dict)

        self.loss_history = list(checkpoint['loss_history'])
        self.training_history = checkpoint['training_history']
        self.epoch = checkpoint['epoch']

        if (hasattr(self, 'current_optimizer')
            and self.current_optimizer is not None):
            self.current_optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])

    
    def checkpoint(self, *args):
        if not hasattr(self, 'current_checkpoint_id'):
            raise Exception('You must initialize checkpoints with model.use_checkpoints() before calling model.checkpoint()')

        if self.current_checkpoint_id == self.job_id:
            self.save_checkpoint(*args)
            exit()
        elif self.current_checkpoint_id + 1 == self.job_id:
            self.load_checkpoint()

        self.current_checkpoint_id += 1

    def Adam_optimize(
            self,
            iterations: int,
            dataset: CDataset,
            batch_size: int = 15,
            lr: float = 0.005,
            betas: Tuple[float] = (0.9, 0.999),
            schedule: bool = False,
            amsgrad: bool = False,
            subset: Union[int, List[int]] = None,
            regularization_factor: Union[float, List[float]] = None,
            thread=True,
            calculation_width=10
    ):
        """
        Runs a round of reconstruction using the Adam optimizer from
        cdtools.reconstructors.AdamReconstructor.

        This is generally accepted to be the most robust algorithm for use
        with ptychography. Like all the other optimization routines,
        it is defined as a generator function, which yields the average
        loss each epoch.

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run.
        dataset : CDataset
            The dataset to reconstruct against.
        batch_size : int
            Optional, the size of the minibatches to use.
        lr : float
            Optional, The learning rate (alpha) to use. Defaultis 0.005. 
            0.05 is typically the highest possible value with any chance 
            of being stable.
        betas : tuple(float)
            Optional, the beta_1 and beta_2 to use. Default is (0.9, 0.999).
        schedule : bool
            Optional, whether to use the ReduceLROnPlateau scheduler.
        amsgrad : bool
            Optional, whether to use the AMSGrad variant of this algorithm.
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to use
        regularization_factor : float or list(float).
            Optional, if the model has a regularizer defined, the set of 
            parameters to pass the regularizer method.
        thread : bool
            Default True, whether to run the computation in a separate thread 
            to allow interaction with plots during computation.
        calculation_width : int
            Default 10, how many translations to pass through at once for 
            each round of gradient accumulation. Does not affect the result, 
            only the calculation speed. 
        
        """
        reconstructor = AdamReconstructor(
            model=self,
            dataset=dataset,
            subset=subset,
        )
        
        # Run some reconstructions
        return reconstructor.optimize(
            iterations=iterations,
            batch_size=batch_size,
            lr=lr,
            betas=betas,
            schedule=schedule,
            amsgrad=amsgrad,
            regularization_factor=regularization_factor, # noqa
            thread=thread,
            calculation_width=calculation_width,
        )

    def LBFGS_optimize(self,
                       iterations: int,
                       dataset: CDataset,
                       lr: float = 0.1,
                       history_size: int = 2,
                       subset: Union[int, List[int]] = None,
                       regularization_factor: Union[float, List[float]] =None,
                       thread: bool = True,
                       calculation_width: int = 10,
                       line_search_fn: str = None):
        """
        Runs a round of reconstruction using the L-BFGS optimizer from
        cdtools.reconstructors.LBFGSReconstructor.

        This algorithm is often less stable that Adam, however in certain
        situations or geometries it can be shockingly efficient. Like all
        the other optimization routines, it is defined as a generator
        function which yields the average loss each epoch.

        NOTE: There is no batch size, because it is a usually a bad idea to use
        LBFGS on anything but all the data at onece

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run.
        dataset : CDataset
            The dataset to reconstruct against.
        lr : float
            Optional, the learning rate to use.
        history_size : int
            Optional, the length of the history to use.
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to use.
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
        """
        reconstructor = LBFGSReconstructor(
            model=self,
            dataset=dataset,
            subset=subset,
        )
        
        # Run some reconstructions
        return reconstructor.optimize(
            iterations=iterations,
            lr=lr,
            history_size=history_size,
            regularization_factor=regularization_factor, # noqa
            thread=thread,
            calculation_width=calculation_width,
            line_search_fn=line_search_fn,
        )
    
    def SGD_optimize(self, 
                     iterations: int,
                     dataset: CDataset,
                     batch_size: int = None,
                     lr: float = 2e-7,
                     momentum: float = 0,
                     dampening: float = 0,
                     weight_decay: float = 0,
                     nesterov: bool = False,
                     subset: Union[int, List[int]] = None,
                     regularization_factor: Union[float, List[float]] = None,
                     thread: bool = True,
                     calculation_width: int = 10):
        """
        Runs a round of reconstruction using the SGD optimizer from
        cdtools.reconstructors.SGDReconstructor.

        This algorithm is often less stable that Adam, but it is simpler
        and is the basic workhorse of gradience descent.

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run.
        dataset : CDataset
            The dataset to reconstruct against.
        batch_size : int
            Optional, the size of the minibatches to use.
        lr : float
            Optional, the learning rate to use.
        momentum : float
            Optional, the length of the history to use.
        dampening : float
            Optional, dampening for the momentum.
        weight_decay : float
            Optional, weight decay (L2 penalty).
        nesterov : bool
            Optional, enables Nesterov momentum. Only applicable when momentum 
            is non-zero. 
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to use.
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of 
            parameters to pass the regularizer method.
        thread : bool
            Default True, whether to run the computation in a separate thread 
            to allow interaction with plots during computation.
        calculation_width : int
            Default 10, how many translations to pass through at once for each 
            round of gradient accumulation.

        """
        reconstructor = SGDReconstructor(
            model=self, 
            dataset=dataset, 
            subset=subset,
        )

        # Run some reconstructions
        return reconstructor.optimize(
            iterations=iterations,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            regularization_factor=regularization_factor, # noqa
            thread=thread,
            calculation_width=calculation_width,
        )


    def report(self):
        """Returns a string with info about the latest reconstruction iteration

        Returns
        -------
        report : str
            A string with basic info on the latest iteration
        """
        if hasattr(self, 'latest_iteration_time'):
            epoch = len(self.loss_history)
            dt = self.latest_iteration_time
            loss = self.loss_history[-1]
            msg = f'Epoch {epoch:3d} completed in {dt:0.2f} s with loss {loss:.5e}'
        else:
            msg = 'No reconstruction iterations performed yet!'

        return msg

    # By default, the plot_list is empty
    plot_list = []


    def inspect(self, dataset=None, update=True):
        """Plots all the plots defined in the model's plot_list attribute

        If update is set to True, it will update any previously plotted set
        of plots, if one exists, and then redraw them. Otherwise, it will
        plot a new set, and any subsequent updates will update the new set

        Optionally, a dataset can be passed, which will allow plotting of any
        registered plots which need to incorporate some information from
        the dataset (such as geometry or a comparison with measured data).

        Plots can be registered in any subclass by defining the plot_list
        attribute. This should be a list of tuples in the following format:
        ( 'Plot Title', function_to_generate_plot(self),
        function_to_determine_whether_to_plot(self))

        Where the third element in the tuple (a function that returns
        True if the plot is relevant) is not required.

        Parameters
        ----------
        dataset : CDataset
            Optional, a dataset matched to the model type
        update : bool, default: True
            Whether to update existing plots or plot new ones

        """
        # We find or create all the figures
        first_update = False
        if update and hasattr(self, 'figs') and self.figs:
            figs = self.figs
        elif update:
            figs = None
            self.figs = []
            first_update = True
        else:
            figs = None
            self.figs = []

        idx = 0
        for plots in self.plot_list:
            # If a conditional is included in the plot, we check whether
            # it is True
            try:
                if len(plots) >=3 and not plots[2](self):
                    continue
            except TypeError as e:
                if len(plots) >= 3 and not plots[2](self, dataset):
                    continue

            name = plots[0]
            plotter = plots[1]

            if figs is None:
                fig = plt.figure()
                self.figs.append(fig)
            else:
                fig = figs[idx]

                
            try: # We try just plotting using the simplest allowed signature
                plotter(self,fig)
                plt.title(name)
            except TypeError as e:
                # TypeError implies it wanted another argument, i.e. a dataset
                if dataset is not None:
                    try:
                        plotter(self, fig, dataset)
                        plt.title(name)
                    except Exception as e: # Don't raise errors: it's just plots
                        pass

            except Exception as e: # Don't raise errors, it's just a plot
                pass

            idx += 1

            if update:
                # This seems to update the figure without blocking.
                plt.draw()
                fig.canvas.start_event_loop(0.001)

        if first_update:
            # But this is needed the first time the figures update, or
            # they won't get drawn at all
            plt.pause(0.05 * len(self.figs))


    def save_figures(self, prefix='', extension='.pdf'):
        """Saves all currently open inspection figures.

        Note that this function is not very intelligent - so, for example,
        if multiple probe modes are being reconstructed and the probe
        plotting function allows one to scroll between different modes, it
        will simply save whichever mode happens to be showing at the moment.
        Therefore, this should not be treated as a good way of saving out
        the full state of the reconstruction.

        By default, the files will be named by the figure titles as defined
        in the plot_list. Files can be saved with any extension suported by
        matplotlib.pyplot.savefig.

        Parameters
        ----------
        prefix : str
            Optional, a string to prepend to the saved figure names
        extention : strategy
            Default is .eps, the file extension to save with.
        """

        if hasattr(self, 'figs') and self.figs:
            figs = self.figs
        else:
            return # No figures to save

        for fig in self.figs:
            fig.savefig(prefix + fig.axes[0].get_title() + extension,
                        bbox_inches = 'tight')


    def compare(self, dataset, logarithmic=False):
        """Opens a tool for comparing simulated and measured diffraction patterns

        This does what it says on the tin.
        
        Also, I am very sorry, the implementation was done while I was
        possessed by Beezlebub - do not try to fix this, if it breaks just
        kill it and start from scratch.

        Parameters
        ----------
        dataset : CDataset
            A dataset containing the simulated diffraction patterns to compare against
        logarithmic : bool, default: False
            Whether to plot the diffraction on a logarithmic scale
        """

        fig, axes = plt.subplots(1,3,figsize=(12,5.3))
        fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.96])
        axslider = plt.axes([0.15,0.06,0.75,0.03])


        def update_colorbar(im):
            if hasattr(im, 'norecurse') and im.norecurse:
                im.norecurse=False
                return

            im.norecurse=True
            im.set_clim(vmin=np.min(im.get_array()),vmax=np.max(im.get_array()))

        def update(idx):
            idx = int(idx) % len(dataset)
            fig.pattern_idx = idx
            updating = True if len(axes[0].images) >= 1 else False

            inputs, output = dataset[idx:idx+1]
            sim_data = self.forward(*inputs).detach().cpu().numpy()
            # The length of sim_data.shape changes when you're doing 
            # either a ptycho (3) or an RPI (2) reconstruction.
            # We need to make sure that sim_data is 2D.
            if len(sim_data.shape) > 2:
                sim_data = sim_data[0]

            meas_data = output.detach().cpu().numpy()[0]
            if hasattr(self, 'mask') and self.mask is not None:
                mask = self.mask.detach().cpu().numpy()
            else:
                mask = 1

            if logarithmic:
                sim_data =np.log(sim_data)/np.log(10)
                meas_data = np.log(meas_data)/np.log(10)

            if not updating:
                axes[0].set_title('Simulated')
                axes[1].set_title('Measured')
                axes[2].set_title('Difference')

                sim = axes[0].imshow(sim_data)
                meas = axes[1].imshow(meas_data * mask)
                diff = axes[2].imshow((sim_data-meas_data) * mask)

                cb1 = plt.colorbar(sim, ax=axes[0], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb1.ax.tick_params(labelrotation=20)
                cb1.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(sim))
                cb2 = plt.colorbar(meas, ax=axes[1], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb2.ax.tick_params(labelrotation=20)
                cb2.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(meas))
                cb3 = plt.colorbar(diff, ax=axes[2], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb3.ax.tick_params(labelrotation=20)
                cb3.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(diff))

            else:

                sim = axes[0].images[-1]
                sim.set_data(sim_data)
                update_colorbar(sim)

                meas = axes[1].images[-1]
                meas.set_data(meas_data * mask)
                update_colorbar(meas)

                diff = axes[2].images[-1]
                diff.set_data((sim_data-meas_data) * mask)
                update_colorbar(diff)


        # This is dumb but the slider doesn't work unless a reference to it is
        # kept somewhere...
        self.slider = Slider(axslider, 'Pattern #', 0, len(dataset)-1, valstep=1, valfmt="%d")
        self.slider.on_changed(update)

        def on_action(event):
            if not hasattr(event, 'button'):
                event.button = None
            if not hasattr(event, 'key'):
                event.key = None

            if event.key == 'up' or event.button == 'up':
                update(fig.pattern_idx - 1)
            elif event.key == 'down' or event.button == 'down':
                update(fig.pattern_idx + 1)
            self.slider.set_val(fig.pattern_idx)
            plt.draw()

        fig.canvas.mpl_connect('key_press_event',on_action)
        fig.canvas.mpl_connect('scroll_event',on_action)
        update(0)
