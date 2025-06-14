import cdtools
from matplotlib import pyplot as plt

filename = 'example_data/lab_ptycho_data.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# FancyPtycho is the workhorse model
model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3, # Use 3 incoherently mixing probe modes
    oversampling=2, # Simulate the probe on a 2xlarger real-space array
    probe_support_radius=120, # Force the probe to 0 outside a radius of 120 pix
    propagation_distance=5e-3, # Propagate the initial probe guess by 5 mm
    units='mm', # Set the units for the live plots
    obj_view_crop=-50, # Expands the field of view in the object plot by 50 pix
)

device = f'cuda:{model.rank}'
model.to(device=device)
dataset.get_as(device=device)

# The learning rate parameter sets the alpha for Adam.
# The beta parameters are (0.9, 0.999) by default
# The batch size sets the minibatch size
for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=50):
    if model.rank == 0: print(model.report())
    # Plotting is expensive, so we only do it every tenth epoch
    if model.epoch % 10 == 0:
        model.inspect(dataset)

# It's common to chain several different reconstruction loops. Here, we
# started with an aggressive refinement to find the probe, and now we
# polish the reconstruction with a lower learning rate and larger minibatch
for loss in model.Adam_optimize(50, dataset,  lr=0.005, batch_size=50):
    if model.rank == 0: print(model.report())
    if model.epoch % 10 == 0:
        model.inspect(dataset)

# This orthogonalizes the recovered probe modes
model.tidy_probes()

model.inspect(dataset)
model.compare(dataset)
plt.show()
