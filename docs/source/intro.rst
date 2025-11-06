Introduction to CDTools
=======================


.. only:: latex

   Introduction to CDTools
   -----------------------

CDTools is an open source python library for ptychography and CDI reconstructions, using an automatic differentiation based approach. It is distributed under an MIT (a.k.a. Expat) license.

.. code-block:: python

   from matplotlib import pyplot as plt
   from cdtools.datasets import Ptycho2DDataset
   from cdtools.models import FancyPtycho
   
   # Load a data file
   dataset = Ptycho2DDataset.from_cxi('ptycho_data.cxi')

   # Look at the data inside
   dataset.inspect()
   
   # Initialize a model from the data
   model = FancyPtycho.from_dataset(dataset)
   
   # Run a reconstruction
   for loss in model.Adam_optimize(10, dataset):
        print(model.report())
   
   # Save the results
   model.save_to_h5('ptycho_results.h5', dataset)
   
   # And look at them!
   model.inspect(dataset) # See the reconstructed object, probe, etc.
   model.compare(dataset) # See how the simulated and measured patterns compare
   plt.show()


CDTools makes it simple to load and inspect data stored in .cxi files. Reconstruction models for common geometries are included "out of the box". For more advanced users, it includes a collection of differentiable functions which can beused to construct new forward models.

The high-level interface to CDTools - datasets and models - is built on a set of lower-level tools. These lower level tools include:

- functions for accessing stored data in .cxi files
- plotting tools to visualize data and reconstructions
- basic operations, like light propagators, needed for coherent diffraction
- analysis functions for assessing the quality of reconstructions


These functions can be used alongside the high-level interface to make fun and fancy reconstruction scripts for challenging data.

If you're interested, read the docs!

