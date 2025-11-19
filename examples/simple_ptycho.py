"""
Runs a very simple reconstruction using the SimplePtycho model, which was
designed to be an easy introduction to show how the models are made and used.

For a more realistic example of how to use cdtools for real-world data,
look at fancy_ptycho.py and gold_ball_ptycho.py, both of which use the
more powerful FancyPtycho model and include more information on how to
correct for common sources of error.
"""
import cdtools
from matplotlib import pyplot as plt

# We load an example dataset from a .cxi file
filename = 'example_data/lab_ptycho_data.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# We create a ptychography model from the dataset
model = cdtools.models.SimplePtycho.from_dataset(dataset)

# We move the model to the GPU
device = 'cuda'
model.to(device=device)
dataset.get_as(device=device)

# We run the reconstruction
for loss in model.Adam_optimize(100, dataset, batch_size=10):
    # We print a quick report of the optimization status
    print(model.report())
    # And liveplot the updates to the model as they happen
    model.inspect(dataset)

# We study the results
model.inspect(dataset)
model.compare(dataset)
plt.show()
