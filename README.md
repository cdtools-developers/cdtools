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

CDTools can be installed via pip as the [cdtools-py](https://pypi.org/project/cdtools-py/) package on [PyPI](https://pypi.org/):

```bash
$ pip install cdtools-py
```

Further documentation is found [here](https://cdtools-developers.github.io/cdtools/).


CDTools was developed in the [photon scattering lab](https://scattering.mit.edu/) at MIT, and further development took place within the [computational x-ray imaging group](https://www.psi.ch/en/cxi) at PSI. The code is distributed under an MIT (a.k.a. Expat) license. If you would like to publish any work that uses CDTools, please contact [Abe Levitan](mailto:abraham.levitan@psi.ch).

Have a wonderful day!
