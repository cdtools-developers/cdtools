import cdtools
from matplotlib import pyplot as plt
import torch as t

filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# We pad the dataset with 10 pixels of zeroes around the edge. This
# data gets masked off, so it is not used for the reconstruction. This padding
# helps prevent aliasing when the probe and object get multiplied. It's a
# helpful step when there is signal present out to the edge of the detector,
# and is usually set to the radius of the probe's Fourier transform (in pixels)
pad = 10
dataset.pad(pad)

dataset.inspect()

# When the dataset is padded with zeroes and masked, the probe reconstruction
# becomes very unstable, and often develops noise at these high-frequency,
# masked off frequencies. To combat this, we simulate the probe at lower
# resolution, using the probe_fourier_crop argument. This is good practice
# in general when padding the dataset
model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3,
    probe_support_radius=50,
    propagation_distance=2e-6,
    units='um',
    probe_fourier_crop=pad 
)


# This is a trick that my grandmother taught me, to combat the raster grid
# pathology: we randomze the our initial guess of the probe positions.
# The units here are pixels in the object array.
# Try running this script with and without this line to see the difference!
model.translation_offsets.data += 0.7 * t.randn_like(model.translation_offsets)

# Not much probe intensity instability in this dataset, no need for this
model.weights.requires_grad = False

device = 'cuda'
model.to(device=device)
dataset.get_as(device=device)

# Create the reconstructor
recon = cdtools.reconstructors.AdamReconstructor(model, dataset)

# This will save out the intermediate results if an exception is thrown
# during the reconstruction
with model.save_on_exception(
        'example_reconstructions/gold_balls_earlyexit.h5', dataset):
    
    for loss in recon.optimize(20, lr=0.005, batch_size=50):
        print(model.report())
        if model.epoch % 10 == 0:
            model.inspect(dataset)

    for loss in recon.optimize(50, lr=0.002, batch_size=100):
        print(model.report())
        if model.epoch % 10 == 0:
            model.inspect(dataset)

    # We can often reset our guess of the probe positions once we have a
    # good guess of probe and object, but in this case it causes the
    # raster grid pathology to return.
    # model.translation_offsets.data[:] = 0

    # Setting schedule=True automatically lowers the learning rate if
    # the loss fails to improve after 10 epochs
    for loss in recon.optimize(100, lr=0.001, batch_size=100, schedule=True):
        print(model.report())
        if model.epoch % 10 == 0:
            model.inspect(dataset)


model.tidy_probes()

# This saves the final result
model.save_to_h5('example_reconstructions/gold_balls.h5', dataset)

model.inspect(dataset)
model.compare(dataset)
plt.show()
