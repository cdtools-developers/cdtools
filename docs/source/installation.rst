Installation
============

CDTools supports python >=3.8 and can be installed via pip as the the `cdtools-py`_ package on `PyPI`_. If you plan to contribute to the code or need a custom environment, installation from source is also possible.

.. _`cdtools-py`: https://pypi.org/project/cdtools-py/
.. _`PyPI`: https://pypi.org/

Option 1: Installation from PyPI
--------------------------------

To install from `PyPI`_, run:

.. code:: bash
	  
   $ pip install cdtools-py

Pytorch, a major dependence of CDTools, often needs to be installed with a specific CUDA version for machine compatability. If you run into issues with pytorch, consider first installing pytorch into your environment using the instructions on `the pytorch site`_.

.. _`the pytorch site`: https://pytorch.org/get-started/locally/

Option 2: Installation from source
----------------------------------


Step 1: Download
^^^^^^^^^^^^^^^^

The source code for CDTools is hosted on `Github`_.

.. _`Github`: https://github.com/cdtools-developers/cdtools


Step 2: Install Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The major dependency for CDTools is pytorch version 2.3.0 or greater. Because the details of the pytorch installation can vary depending on platform and GPU availability, it is recommended that you first install pytorch using the instructions on `the pytorch site`_. The remaining dependencies can be installed by running the following command from the top level directory of the git repository:

.. code:: bash
	  
   $ pip install -r requirements.txt

Note that several optional dependencies used for testing and documentation will also be installed. The full set of dependencies and minimum requirements are listed below. CDTools is reguarly tested with the latest versions of these packages and with python 3.8 through 3.12.

Required dependencies:

   * `numpy <http://www.numpy.org>`_ >= 1.0
   * `scipy <http://www.scipy.org>`_ >= 1.0
   * `matplotlib <https://matplotlib.org>`_ >= 2.0
   * `pytorch <https://pytorch.org>`_ >= 2.3.0
   * `python-dateutil <https://github.com/dateutil/dateutil/>`_
   * `h5py <https://www.h5py.org/>`_ >= 2.1

Optional dependencies for running tests:

   * `pytest <https://docs.pytest.org/>`_
   * `pooch <https://www.fatiando.org/pooch/latest/>`_

Optional dependencies for building docs:

   * `sphinx <https://www.sphinx-doc.org/>`_ >= 4.3.0
   * `sphinx-argparse <https://sphinx-argparse.readthedocs.io>`_
   * `sphinx_rtd_theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_ >= 0.5.1


Step 3: Install
^^^^^^^^^^^^^^^

To install CDTools, run the following command from the top level directory of the git repository:

.. code:: bash
	  
   $ pip install -e . --no-deps

   
This will install CDTools in developer mode, so that changes to the code will propagate to the installed version immediately. This is best if you plan to actively develop CDTools. If you simply need a custom environment, you can also install CDTools in standard mode using:

.. code:: bash
	  
   $ pip install . --no-deps
   
  
Step 4: Run The Tests
^^^^^^^^^^^^^^^^^^^^^

To ensure that the installation has worked correctly, it is recommended that you run the unit tests. Execute the following command from the top level directory of the git repository:

.. code:: bash

   $ pytest

If any tests fail, make sure that you have all the noted dependencies properly installed. If you do, and things still aren't working, `open an issue on the github page <https://github.com/cdtools-developers/cdtools/issues>`_ and we'll get to the bottom of it.
