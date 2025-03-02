Examples
========

Included with the repository are a number of example scripts that demonstrate how various aspects of CDTools work. It is recommended to read through at least a few of them before continuing to the tutorial.

All the datasets used in these example scripts are included in the repository, and the scripts should be runnable as soon as CDTools is installed


Inspect Dataset
---------------

The first example loads and visualizes ptychography data stored in a .cxi file.

.. literalinclude:: ../../examples/inspect_dataset.py

First, data is read into a Ptycho2DDataset object, which is a subclass of a pytorch dataset that knows a bit about the structure of ptychography data. Calling :code:`dataset.inspect` generates a plot showing an overview of the ptychography scan data. On the left is a scatter plot showing the integrated detector intensity at each probe location. On the right, raw detector images are shown. The dataset can be navigated by clicking around the scatter plot on the left.

		    
Simple Ptycho
-------------

This script runs a ptychography reconstruction using the SimplePtycho model, a bare-bones ptychography model for the transmission geometry.

The purpose of the SimplePtycho model is pedagogical: there are very few situations where it would preferred to the FancyPtycho model which will be introduced later.

Because of it's simplicity, the definition of this model is much simpler than the definition of FancyPtycho, and it is therefore a good first model to look at to learn how to implement a custom ptychography model in CDTools.

.. literalinclude:: ../../examples/simple_ptycho.py

When reading this script, note the basic workflow. After the data is loaded, a model is created to match the geometry stored in the dataset with a sensible default initialization for all the parameters.

Next, the model is moved to the GPU using the :code:`model.to` function. Any device understood by :code:`torch.Tensor.to` can be specified here. The next line is a bit more subtle - the dataset is told to move patterns to the GPU before passing them to the model using the :code:`dataset.get_as` function. This function does not move the stored patterns to the GPU. If there is sufficient GPU memory, the patterns can also be pre-moved to the GPU using :code:`dataset.to`, but the speedup is empirically quite small.

Once the device is selected, a reconstruction is run using :code:`model.Adam_optimize`. This is a generator function which will yield at every epoch, to allow some monitoring code to be run.

Finally, the results can be studied using :code:`model.inspect(dataet)`, which creates or updates a set of plots showing the current state of the model parameters. :code:`model.compare(dataset)` is also called, which shows how the simulated diffraction patterns compare to the measured diffraction patterns in the dataset.


Fancy Ptycho
------------

This script runs a reconstruction on the same data, but using the workhorse FancyPtycho model, demonstrating some of it's more commonly used features

.. literalinclude:: ../../examples/fancy_ptycho.py

The :code:`FancyPtycho.from_dataset` factory function has many keyword arguments which can turn on or modify various mixins. In this case, we perform a reconstruction with:

- 3 incoherently mixing probe modes (in the vein of doi:10.1038/nature11806)
- A probe array expanded by a factor of 2 in real space, i.e. simulated on a 2x2 upsampled grid in Fourier space (in the vein of doi:10.1103/PhysRevA.87.053850)
- A circular finite support constraint applied to the probe
- An initial guess for the probe which has been propagated from its focus position

By default, FancyPtycho will also optimize over the following model parameters, each of which corrects for a specific source of errror:

:code:`model.background`
      A frame-independent detector background

:code:`model.weights`
      A frame-to-frame variation in the incoming probe intensity

:code:`model.translation_offsets`
      A frame-independent detector background

These corrections can be turned off (on) by calling :code:`model.<parameter>.requires_grad = False #(True)`.


Gold Ball Ptycho
----------------

This script shows how the FancyPtycho model might be used in a realistic situation, to perform a reconstruction on the classic `gold balls <http://www.cxidb.org/id-65.html>`_ dataset. This script also shows how to save results!

.. literalinclude:: ../../examples/gold_ball_ptycho.py

Note, in particular, the use of :code:`model.save_on_exception` and :code:`model.save_to_h5` to save the results of the reconstruction. If a different file format is required, :code:`model.save_results` will save to a pure-python dictionary.

Finally, note that there are several small adjustments made to the script to counteract particular sources of error that are present in this dataset, for example the raster grid pathology caused by the scan pattern used. Also note that not every mixin is needed every time - in this case, we turn off optimization of the :code:`weights` parameter.


Gold Ball Split
---------------

It is very common to run reconstructions on two disjoint subsets of the same dataset, as well as the full dataset. This is primarily done to estimate the resolution of a reconstruction via the Fourier ring correlation (FRC).

.. literalinclude:: ../../examples/gold_ball_split.py

This script simply divides the dataset in two, and then performs the same reconstruction on both halves of the dataset, as well as the full dataset.


Gold Ball Synthesize
--------------------

Once we have a set of three reconstructions - two half data reconstructions and a full data reconstruction, we need to calculate the resolution metrics. This script shows how that is done.

.. literalinclude:: ../../examples/gold_ball_synthesize.py


Transmission RPI
----------------

CDTools also contains a forward model for randomized probe imaging (RPI, doi:10.1364/OE.397421). This is currently not documented fully, but hopefully will be soon
