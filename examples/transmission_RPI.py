import cdtools
import pickle
import torch as t
from matplotlib import pyplot as plt

# First, we load an example dataset from a .cxi file
ss_filename = 'example_data/Optical_Data_ss.cxi'

with open('example_data/Optical_ptycho_incoherent.pickle', 'rb') as f:
    ptycho_results = pickle.load(f)

probe = ptycho_results['probe']
background = ptycho_results['background']

dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(ss_filename)


# Next, we create an RPI model from the dataset
# Note that we explicitly as for two incoherent probe modes
model = cdtools.models.RPI.from_dataset(dataset, probe, [500,500],
                                        background=background, n_modes=2,
                                        initialization='random',panel_plot_mode=True)


# Let's do this reconstruction on the GPU, shall we? 
if t.cuda.is_available():
    model.to(device='cuda')
    dataset.get_as(device='cuda')

# Note that the inspect step takes the vast majority of the time
# The regularization is an L2 regularizer that empirically helps accelerate
# convergence
for loss in model.Adam_optimize(30, dataset, lr=0.4, regularization_factor=[0.05,0.05]):
    model.inspect(dataset)
    print(model.report())
    

# Now we use the regularizer to damp all but the top modes
for loss in model.LBFGS_optimize(50, dataset, lr=0.4, regularization_factor=[0.001,0.1]):
    #model.inspect(dataset)
    print(model.report())

# Save results to a python dictionary
results = model.save_results()

# Finally, we plot the results
model.inspect(dataset)
model.compare(dataset)
plt.show()
