Tutorial
========

The following tutorial gives a peek under the hood, and is intended for someone who might want to write their own variant of a ptychography model or modify an existing model to meet a specific need. If you just need to use CDTools for a reconstruction, or are just starting to work with the code, the examples section is a great first introduction.

In the first section of the tutorial, we will discuss how the datasets are defined and go through the process of defining a new dataset type. Following that, we will go through the process of defining a simplified model for standard ptychography.
			   

Datasets
--------

In this section, we will write a bare-bones dataset class for 2D ptychography data to demonstrate the process of writing a new dataset class. At the end of the tutorial, we will have written the file examples/tutorial_basic_ptycho_dataset.py, which can be consulted for reference.

Basic Idea
++++++++++

At it's core, a dataset object for CDTools is just an object that implements the dataset interface from pytorch. For this reason, the base class (:code:`CDataset`) from which all the datasets are defined is itself a subclass of :code:`torch.utils.data.Dataset`. In addition, CDataset implements an extra layer that allows for a separation between the device (CPU or GPU) that the data is stored on and the device that it returns data on. This allows for GPU-based reconstructions on datasets that are too large to fit into the GPU in their entirety.

The pytorch Dataset interface is very simple. A dataset simply has to define two functions, :code:`__len__()` and :code:`__getitem__()`. Thus, we can always access the data in a Dataset :code:`mydata` using the syntax :code:`mydata[index]` or :code:`mydata[slice]`. Overriding these functions will be the first task in defining a new dataset.

In CDTools datasets, the layer that allows for separation between the device that data is stored on and the device that data is loaded onto is implemented in the :code:`__getitem__()` function. Instead of overriding this function directly, one should override the :code:`_load()` function, which is used internally by :code:`__getitem__()`.

In addition to acting as a pytorch Dataset, CDTools Datasets also work as interfaces to .cxi files. Therefore, when writing a new dataset, it is important to also override the functions :code:`to_cxi()` and :code:`from_cxi()` which handle writing to and reading from cxi files, respectively.

The final piece of the puzzle is the :code:`inspect()` method. This is not required to be defined for all datasets, however it is extremely valuable to offer a simple way of exploring a dataset visually. Therefore it is highly recommended to implement this function, which should load a plot or interactive plot that allows a user to visualize the data that they have loaded.

Writing the Skeleton
++++++++++++++++++++

We can start with the basic skeleton for this file. In addition to our standard imports, we also import the base CDataset class and the data tools. We then define an :code:`__all__` list as good practice, and set up the inheritance of our class.

.. code-block:: python

  
    import numpy as np
    import torch as t
    from matplotlib import pyplot as plt
    from cdtools.datasets import CDataset
    from cdtools.tools import data as cdtdata

    __all__ = ['BasicPtychoDataset']

    class BasicPtychoDataset(CDataset):
        """The standard dataset for a 2D ptychography scan"""
        pass


Initialization
++++++++++++++

The next thing to implement is the initialization code. Here we can leverage some of the work already done in the base CDataset class. There are a number of kinds of metadata that can be stored in a .cxi file that aren't related to the kind of experiment you're performing - sample ID, start and end times, and so on. The CDataset's initialization routine handles loading and storing these various kinds of metadata, so we can start the definition of our initialization routine by leveraging this:

.. code-block:: python

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


Of course, there is also some data that are unique to this kind of dataset. In this case, those data are the probe translations and the measured diffraction patterns. Therefore, we extend this definition to the following:

.. code-block:: python

    def __init__(self, translations, patterns, *args, **kwargs):
        """Initialize the dataset from python objects"""

        super().__init__(*args, **kwargs)
        self.translations = t.Tensor(translations).clone()
        self.patterns = t.Tensor(patterns).clone()


Dataset Interface
+++++++++++++++++

The next set of functions to write are those that plug into the dataset interface. We want :code:`len(dataset)` to return the number of diffraction patterns, which is straightforward to implement.

For the :code:`_load()` implementation, we need to consider what format the data should be returned in. The standard for all CDTools datasets is to return a tuple of (inputs, output). The inputs should always be defined as a tuple of inputs, even if there is only one input for this kind of data. As we will see later in the section on constructing models, this makes it possible to write the automatic differentiation code in a way that is applicable to every model.

In this case, our "inputs" will be a tuple of (pattern index, probe translation). This is not the only reasonable choice - it would also be possible, for example to define the input as just a pattern index (and store the probe translations in the model). For simple ptychography models with no error correction, it's also possible to just take a probe translation as an input with no index. Requiring both is the compromise that's been implemented in the default ptychography models defined with CDTools, and therefore we will follow that format here.

.. code-block:: python

    def __len__(self):
        return self.patterns.shape[0]

    def _load(self, index):
        return (index, self.translations[index]), self.patterns[index]

Remember that it's not needed to worry about what device or datatype the data is stored as here, as the relevant conversions will be performed by the :code:`__getitem()` method defined in the superclass. However, we do generally implement a method, :code:`to()`, that moves the data back and forth between devices and datatypes. This lets a user speed up data loading onto the GPU by preloading the data, for example - provided there is enough space.

.. code-block:: python

    def to(self, *args, **kwargs):
        """Sends the relevant data to the given device and dtype"""
        super().to(*args,**kwargs)
        self.translations = self.translations.to(*args, **kwargs)
        self.patterns = self.patterns.to(*args, **kwargs)

Here we can see that we first make sure to call the superclass function to handle sending any information (such as a pixel mask, or detector background) that would have been defined in CDataset to the relevant device. Then we handle the new objects that are defined specifically for this kind of dataset.


Loading and Saving
++++++++++++++++++

Now we turn to writing the tools to load and save data. First, to load the data, we override :code:`from_cxi()`, which is a factory method. In this case, we start by using the superclass to load the metadata. Then we explicitly load in and add the data that's specific to this dataset class

.. code-block:: python

    @classmethod
    def from_cxi(cls, cxi_file):
        """Generates a new CDataset from a .cxi file directly"""

        # Generate a base dataset
        dataset = CDataset.from_cxi(cxi_file)
        # Mutate the class to this subclass (BasicPtychoDataset)
        dataset.__class__ = cls

        # Load the data that is only relevant for this class
        patterns, axes = cdtdata.get_data(cxi_file)
        translations = cdtdata.get_ptycho_translations(cxi_file)

        # And now re-add it
        dataset.translations = t.Tensor(translations).clone()
        dataset.patterns = t.Tensor(patterns).clone()

        return dataset


Now to save the data, we override :code:`to_cxi()`, in a fairly self-explanatory way.

.. code-block:: python

    def to_cxi(self, cxi_file):
        """Saves out a BasicPtychoDataset as a .cxi file"""

        super().to_cxi(cxi_file)
        cdtdata.add_data(cxi_file, self.patterns, axes=self.axes)
        cdtdata.add_ptycho_translations(cxi_file, self.translations)

Note that these functions should be defined to work on h5py file objects representing the .cxi files (.cxi files are just .h5 files with a special formatting).


Inspecting
++++++++++

The final piece of the puzzle is writing a function to look at your data! This is an important thing to work on for a dataset class that you intend to use regularly, as being able to easily peruse your raw data has incalculable value. Here, we satisfy ourselves with just plotting a random diffraction pattern.

.. code-block:: python

    def inspect(self):
        """Plots a random diffraction pattern"""

        index = np.random.randint(len(self))
        plt.figure()
        plt.imshow(self.patterns[index,:,:].cpu().numpy())


Notes
+++++

This is a bare-bones class, set up to demonstrate the minimum neccessary to develop a new type of dataset class. As a result, it doesn't implement a number of things that are useful or valuable in practice (and which the default Ptycho2DDataset does implement). That includes a useful data inspector, the ability to load datasets directly from filenames, and default tweaks to how metadata such as backgrounds and masks are loaded.

	
Models
------

In this section, we will write a basic model for 2D ptychography reconstructions. At the end of this tutorial, we will have written the class defined in examples/tutorial_simple_ptycho_model.py


Basic Idea
++++++++++

Just like CDTools Datasets subclass pytorch Datasets, CDTools models subclass pytorch modules. However, the concept of a CDTools model does differ slightly from that of a pytorch module, because the CDTools models also contain a few standard methods to run automatic differentiation reconstructons on themselves.

This isn't necessarily the cleanest or most portable approach, but we've found that it feels very natural from the perspective of an end user interacting with the toolbox only through the reconstruction scripts.

The heart of each model is a :code:`model.forward()` function. In any CDTools model, this forward function maps a set of parameters describing the specific diffraction pattern to simulate to the simulated result. When it's paired with an appropriate dataset for a reconstruction, it maps from the "inputs" defined by the dataset to the "outputs".

For ptychography, this information is usually index of the exposure within the dataset (which is used to retrieve exposure-to-exposure information, like probe intensity factors) and the object translation.

A simple forward model is defined in the top level :code:`CDIModel` class from which all other models are derived, and rarely needs to be overridden. The definition is quite simple:

.. code-block:: python
		
    def forward(self, *args):
        return self.measurement(self.forward_propagator(self.interaction(*args)))

So we can see that to fully implement this forward model, we have to define the three functions :code:`model.interaction()`, :code:`model.forward_propagator()`, and :code:`model.measurement()`, which simulate conceptual stages in the diffraction process.

In addition to the core model definition, a few other functions need to be defined to make the model useful. The model needs an *initializer* to create itself from a dataset, it must have an appropriate *loss function* defined for use with automtic differentiation, a way of plotting the progress of a reconstruction, and must know how to save the results of a reconstruction in a useful format.


Writing the Skeleton
++++++++++++++++++++

Once again, we start with the basic skeleton

.. code-block:: python

   import torch as t
   from cdtools.models import CDIModel
   from cdtools import tools
   from cdtools.tools import plotting as p

    __all__ = ['SimplePtycho']

    class SimplePtycho(CDIModel):
        """A simple ptychography model to demonstrate the structure of a model
        """

Note that we imported the full tools package, as we will find ourselves using many low-level functions defined there to implement the model.


Initialization from Python
++++++++++++++++++++++++++

Two initialization functions need to be written. First, we write the :code:`__init__()` function, which initializes the model from a collection of python objects describing the system. We then write an initializer that creates a model using a dataset to initialize the various parameters.

There is no requirement for what the arguments to the initialization function of any particular model should be, only that they contain enough information to run the simulations! It should be chosen in a model-by-model basis to allow for the most sensible code.

.. code-block:: python

   def __init__(
           self,
	   wavelength,
	   probe_basis,
	   probe_guess,
	   obj_guess,
	   min_translation = [0,0],
   ):
       # We initialize the superclass
       super().__init__()
   
       # We register all the constants, like wavelength, as buffers. This
       # lets the model hook into some nice pytorch features, like using
       # model.to, and broadcasting the model state across multiple GPUs
       self.register_buffer('wavelength', t.as_tensor(wavelength))
       self.register_buffer('min_translation', t.as_tensor(min_translation))
       self.register_buffer('probe_basis', t.as_tensor(probe_basis))
       
       # We cast the probe and object to 64-bit complex tensors
       probe_guess = t.as_tensor(probe_guess, dtype=t.complex64)
       obj_guess = t.as_tensor(obj_guess, dtype=t.complex64)
       
       # We rescale the probe here so it learns at the same rate as the
       # object when using optimizers, like Adam, which set the stepsize
       # to a fixed maximum
       self.register_buffer('probe_norm', t.max(t.abs(probe_guess)))
       
       # And we store the probe and object guesses as parameters, so
       # they can get optimized by pytorch
       self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
       self.obj = t.nn.Parameter(obj_guess)

       
The first thing to notice about this model is that all the fixed, geometric information is stored with the :code:`module.register_buffer()` function. This is what makes it possible to move all the relevant tensors between devices using a single call to :code:`module.to()`, for example. It stores thetensor as an object attribute, but it also registers it so that pytorch is aware that this attribute helps to encode the state of the model.

The supporting information we need is the wavelength of the illumination, the basis of the probe array in real space, and an offset to define the zero point of the translation. 

The final two pieces of information that we need to save are the probe and object, and both of these get defined as :code:`t.nn.Parameter` objects instead of Tensors. As a result, they get registered as parameters in the pytorch module, and will therefore be optimized over in any later reconstructions. In addition, the :code:`requires_grad` flag is set to :code:`True`, which means that the information needed for gradient calculations will be stored on every Tensor that results from a calculation including a Parameter.

A list of all parameters associated with the module can be found by calling :code:`module.parameters()`.

Any additional targets of reconstruction - such as exposure-to-exposure illumination weights, translation offsets, or a detector background - would be added to the model as a parameter in a similar way.

One final note is that we actually store a scaled version of the probe. This is a specific case of a general policy designed around making it easy to use the Adam optimizer.

The Adam optimizer is designed so that the learning rate sets the maximum stepsize which will be taken in any single iteration. Therefore, it is important to make sure that *all parameters of the model are of order unity*. To enable this, we scale the probe so that the typical pixel value within the probe array is of order 1.

This is important to remember when adding additional error models. Rescaling all the parameters to have a typical amplitude near 1 is the best way to get well-behaved reconstructions.


Initialization from Dataset
+++++++++++++++++++++++++++

To initialize the object from a dataset, we need to start by extracting the relevant information from the dataset, before calling the constructor we defined above:

.. code-block:: python

    @classmethod
    def from_dataset(cls, dataset):

        # We get the key geometry information from the dataset
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # Then, we generate the probe geometry
        ewg = tools.initializers.exit_wave_geometry
        probe_basis =  ewg(det_basis, det_shape, wavelength, distance)

        # Next generate the object geometry from the probe geometry and
        # the translations
        (indices, translations), patterns = dataset[:]
        pix_translations = tools.interactions.translations_to_pixel(
            probe_basis,
            translations,
        )
        obj_size, min_translation = tools.initializers.calc_object_setup(
            det_shape,
            pix_translations,
        )

        # Finally, initialize the probe and object using this information
        probe = tools.initializers.SHARP_style_probe(dataset)
        obj = t.ones(obj_size).to(dtype=t.complex64)

        return cls(
            wavelength,
            probe_basis,
            probe,
            obj,
            min_translation=min_translation
        )

Here, we start by pulling the basic geometric information from the dataset. Then, we use a number of the basic tools to do calculations such as finding the probe basis from the detector geometry, or calculating how big our object array should be.

Once we have the basic setup ready, we then use one of the initialization functions - in this case, :code:`tools.initializers.SHARP_style_probe`, to find a sensible initialization for the probe. This particular initialization is based on the approach used in the SHARP package, where the square-root of the mean diffraction pattern intensity is used to estimate the structure of the illumination at focus.

Once all the needed information has been collected, we initialize the object.


The Forward Model
+++++++++++++++++

First, we have to implement the interaction model, as below:

.. code-block:: python

    def interaction(self, index, translations):
        
        # We map from real-space to pixel-space units
        pix_trans = tools.interactions.translations_to_pixel(
            self.probe_basis,
            translations)
        pix_trans -= self.min_translation
        
        # This function extracts the appropriate window from the object and
        # multiplies the object and probe functions
        return tools.interactions.ptycho_2D_round(
            self.probe_norm * self.probe,
            self.obj,
            pix_trans)


Here, we take input in the form of an index and a translation. Note that this input format much match the format that is output by the associated datasets that we will use for reconstruction, in this case BasicPtychoDataset.

We start by mapping the translation, given in real space, into pixel coordinates. Then, we use an "off-the-shelf" interaction model - :code:`ptycho_2d_round`, which models a standard 2D ptychography interaction, but rounds the translations to the nearest whole pixel (does not attempt subpixel translations).

The next three definitions amount to just choosing an off-the-shelf function to simulate each step in the chain.

.. code-block:: python

    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)

    def measurement(self, wavefields):
        return tools.measurements.intensity(wavefields)

    def loss(self, sim_data, real_data):
        return tools.losses.amplitude_mse(real_data, sim_data)


The forward propagator maps the exit wave to the wave at the surface of the detector, here using a far-field propagator. The measurement maps that exit wave to a measured pixel value, and the loss defines a loss function to attempt to minimize. The loss function we've chosen - the amplitude mean squared error - is the most reliable one, and can also easily be overridden by an end user.


Plotting
++++++++

The base CDIModel class has a function, :code:`model.inspect()`, which looks for a class variable called :code:`plot_list` and plots everything contained within. The plot list should be formatted as a list of tuples, with each tuple containing:

* The title of the plot
* A function that takes in the model and generates the relevant plot
* Optional, a function that takes in the model and returns whether or not the plot should be generated
  
.. code-block:: python

    # This lists all the plots to display on a call to model.inspect()
    plot_list = [
        ('Probe Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis)),
        ('Probe Phase',
         lambda self, fig: p.plot_phase(self.probe, fig=fig, basis=self.probe_basis)),
        ('Object Amplitude',
         lambda self, fig: p.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(self.obj, fig=fig, basis=self.probe_basis))
    ]

In this case, we've made use of the convenience plotting functions defined in :code:`tools.plotting`.


Saving
++++++

By default, a function :code:`model.save_results()` is defined, which returns a python dictionary with an entry, :code:`'state_dict'`, containing all the registered parameters and buffers in the model. It also contains a basic record of the model's training history. This function is used internally by :code:`model.save_to_h5()`, as well as all other convenience functions for saving results.

Sometimes, it is also useful to return a more user-friendly version of the results, such as a properly rescaled version of the probe. To make this possible, :code:`model.save_results()` is often overridden:


.. code-block:: python

    def save_results(self, dataset):
        # This will save out everything needed to recreate the object
        # in the same state, but it's not the best formatted. 
        base_results = super().save_results()

        # So we also save out the main results in a more useable format
        probe_basis = self.probe_basis.detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        wavelength = self.wavelength.cpu().numpy()

        results = {
            'probe_basis': probe_basis,
            'probe': probe,
            'obj': obj,
            'wavelength': wavelength,
        }

        return {**base_results, **results}

However, it is perfectly possible to write a new ptychography model without overriding :code:`model.save_results()`

	
Testing
+++++++

We can test this model with a simple script, in examples/tutorial_finale.py. By filling in the backend here, we've been able to create a ptychography model that can be accessed and used in reconstructions via the same interface as the models we discussed in the examples section.

.. code-block:: python

    from tutorial_basic_ptycho_dataset import BasicPtychoDataset
    from tutorial_simple_ptycho import SimplePtycho
    from h5py import File
    from matplotlib import pyplot as plt
    
    filename = 'example_data/lab_ptycho_data.cxi'
    with File(filename, 'r') as f:
        dataset = BasicPtychoDataset.from_cxi(f)    
		
    dataset.inspect()

    model = SimplePtycho.from_dataset(dataset)
    
    model.to(device='mps')#cuda')
    dataset.get_as(device='mps')#cuda')
    
    for loss in model.Adam_optimize(10, dataset):
        model.inspect(dataset)
	print(model.report())

    model.inspect(dataset)
    model.compare(dataset)
    plt.show()
    

Happy modeling!
