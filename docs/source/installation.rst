Installation
============

CDTools supports python >=3.9 and can be installed via pip as the the `cdtools-py`_ package on `PyPI`_. If you plan to contribute to the code or need a custom environment, installation from source is also possible.

.. _`cdtools-py`: https://pypi.org/project/cdtools-py/
.. _`PyPI`: https://pypi.org/

Option 1: Installation from PyPI
--------------------------------

To install from `PyPI`_, run:

.. code:: bash
	  
   $ pip install cdtools-py

or you can use `uv`_ for a faster installation:

.. _`uv`: https://github.com/astral-sh/uv

.. code:: bash

   $ uv pip install cdtools-py

Pytorch, a major dependence of CDTools, often needs to be installed with a specific CUDA version for machine compatability. If you run into issues with pytorch, consider first installing pytorch into your environment using the instructions on `the pytorch site`_.

.. _`the pytorch site`: https://pytorch.org/get-started/locally/

Option 2: Installation from source
----------------------------------


Step 1: Download
^^^^^^^^^^^^^^^^

The source code for CDTools is hosted on `Github`_.

.. _`Github`: https://github.com/cdtools-developers/cdtools


To download the source code, you can either clone the repository using git:

.. code:: bash

   $ git clone https://github.com/cdtools-developers/cdtools.git

or you can download a zip file of the repository from the `releases page`_.

.. _`releases page`: https://github.com/cdtools-developers/cdtools/releases


Step 2: Install
^^^^^^^^^^^^^^^

Move to the directory where you downloaded the source code. It is recommended that you create a new python virtual environment to install CDTools into.

Installation using pip and uv. Editable mode is recommended for development purposes and is added with the `-e` flag.

.. code:: bash

   $ pip install -e .

or using uv:

.. code:: bash

   $ uv pip install -e .


To install the required test and documentation dependencies as well, use:

.. code:: bash

   $ pip install -e ."[tests,docs]"

or using uv:

.. code:: bash

   $ uv pip install -e ."[tests,docs]"

CDTools is reguarly tested with the latest versions of these packages and with python 3.9 through 3.14.


Required dependencies (see pyproject.toml for all details):

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


Optional step 4: Run The Tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure that the installation has worked correctly, it is recommended that you run the unit tests. Execute the following command from the top level directory of the git repository:

.. code:: bash

   $ python -m pytest

If any tests fail, make sure that you have all the noted dependencies properly installed. If you do, and things still aren't working, `open an issue on the github page <https://github.com/cdtools-developers/cdtools/issues>`_ and we'll get to the bottom of it.
