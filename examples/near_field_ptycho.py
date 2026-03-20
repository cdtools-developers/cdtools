import cdtools
import torch as t
from matplotlib import pyplot as plt

filename = 'example_data/PETRAIII_P25_Near_Field_Ptycho.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

dataset.inspect()

# Setting near_field equal to True uses an angular spectrum propagator in
# lieu of the default Fourier-transform propagator for far-field ptychography.
#
# If propagation_distance is not set, it assumes that the geometry is
# a standard near-field geometry with flat illumination wavefronts, and
# pulls the sample to detector distance from dataset.distance
#
# If propagation_distance is set, it assumes a Fresnel scaling theorem
# geometry with:
#
# - distance (from the dataset): The sample-to-detector distance
# - propagation_distance: The focus-to-sample distance
# 
model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=1,
    near_field=True,
    propagation_distance=3.65e-3, # 3.65 downstream from focus
    units='um', # Set the units for the live plots
    obj_view_crop=-35,
    panel_plot_mode=True,
)

if t.cuda.is_available():
    model.to(device='cuda')
    dataset.get_as(device='cuda')

model.inspect(dataset)

recon = cdtools.reconstructors.AdamReconstructor(model, dataset)

for loss in recon.optimize(100, lr=0.04, batch_size=10):
    print(model.report())
    model.inspect(dataset, min_interval=5)

for loss in recon.optimize(50, lr=0.005, batch_size=50):
    print(model.report())
    model.inspect(dataset, min_interval=5)

# This orthogonalizes the recovered probe modes
model.tidy_probes()

model.inspect(dataset, replot_all=True)
model.compare(dataset)
plt.show()
