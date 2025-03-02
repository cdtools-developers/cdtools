General Reference
=================

The full documentation contains the details of all the functions, classes, etc. contained in CDTools - however, there are a few issues which cut across the various functions and are important enough to note in one place. This includes some definitions which are used across the packages, as well as a few conventions which are important to understand before writing code with CDTools. Because of that, it is recommended to read through this general reference page before diving into the reference documentation.


Arrays, Numpy, and Pytorch
--------------------------

By necessity, CDTools operates using a mixture of numpy arrays and pytorch tensors. This, unfortunately, sucks. Certain functions - such as the various tools designed to be used in ptychography models - only accept pytorch tensors. Other functions - mostly analysis functions - only work with numpy arrays. Where possible (for example, in many analysis functions), input is accepted in either format.

Functions that can accept either pytorch tensors or numpy arrays will have the type for the relevant inputs documented as "array", rather than "np.ndarray" or "torch.Tensor". In general, these functions will either accept all "array" type inputs as numpy arrays, or all as torch tensors. They will then return a result in a format matching that of the inputs. While many of these functions will work with mixed numpy/pytorch input, the output behavior of these functions is not, in general, defined for such a case, so it is heavily discouraged.

Please note that adherance to the conventions above is a work in progress, and you may find improperly documented or implemented functions. Please file a bug report if you do!


Unit Conventions
----------------

All physical units used everywhere are SI units, with no exception. Every unit of length is assumed to be meters, all units of energy are Joules, etc. This matches the .CXI file specification and provides easy interoperability.


Deviations from CXI Conventions
-------------------------------

There are several intentional deviations between the conventions used for the ptychography dataset classes and the .CXI file spec.

First, all probe translations are stored as translations of the probe over the object. We have found that this more closely maps onto the actual way that most ptychography experiments are run, and it is a more natural choice to use when cropping out a section of the object to multiply the probe with. However, .CXI files are defined to store translations of the object, not the probe - thus, all translations are inverted when read from a .CXI file, and again when written out to a .CXI file

Second, the convention for storing bases in CDTools is transposed from the convention used in .CXI files. This means that the basis vectors are column vectors, which we find more natural when they are used to define a coordinate system. CDTools will accept a basis stored either in this (nonstandard) format in a .CXI file, or a basis stored in the appropriate format, and any files saved using the built-in tools will produce compliant .CXI files.


