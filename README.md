# CDTools

CDTools is a python library for ptychography and CDI reconstructions, using an Automatic Differentiation based approach.

```python
from matplotlib import pyplot as plt
from cdtools.datasets import Ptycho2DDataset
from cdtools.models import FancyPtycho

# Load a data file
dataset = Ptycho2DDataset.from_cxi('ptycho_data.cxi')

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
```

Further documentation is found [here](https://cdtools-developers.github.io/cdtools/).

# Installation

CDTools can be installed in several ways depending on your needs. For most users, installation from pypi is recommended. For developers or those who want the latest features, installation from source is available.

## Installation from pypi

CDTools can be installed via pip as the [cdtools-py](https://pypi.org/project/cdtools-py/) package on [PyPI](https://pypi.org/):

```bash
$ pip install cdtools-py
```

or using [uv](https://github.com/astral-sh/uv):

```bash
$ uv pip install cdtools-py
```

## Installation from Source

For development or to access the latest features, CDTools can be installed directly from source:


```bash
$ git clone https://github.com/cdtools-developers/cdtools.git
$ cd cdtools
$ pip install -e .
```


or using [uv](https://github.com/astral-sh/uv):

```bash
$ git clone https://github.com/cdtools-developers/cdtools.git
$ cd cdtools
$ uv pip install -e .
```

## Installing for Contributors (with tests and docs dependencies)

If you want to run the test suite or build the documentation, install with the extra dependencies:

```bash
$ pip install -e ."[tests,docs]"
```
or with uv:
```bash
$ uv pip install -e ."[tests,docs]"
```

CDTools was developed in the [photon scattering lab](https://scattering.mit.edu/) at MIT, and further development took place within the [computational x-ray imaging group](https://www.psi.ch/en/cxi) at PSI. The code is distributed under an MIT (a.k.a. Expat) license. If you would like to publish any work that uses CDTools, please contact [Abe Levitan](mailto:abraham.levitan@psi.ch).

Have a wonderful day!
