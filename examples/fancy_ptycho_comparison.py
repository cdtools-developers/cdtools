"""This script runs reconstructions using both the old
method of cdtools reconstruction (model.Adam_optimize)
and the new method based on the creation of a Reconstructor
class

"""


import cdtools
import cdtools.optimizer
import torch as t
import numpy as np
import time
import copy
from matplotlib import pyplot as plt

t.manual_seed(0)

filename = 'examples/example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# Create a dict to store loss values
losses = {}

pad = 10
dataset.pad(pad)
model_original = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3,
    probe_support_radius=50,
    propagation_distance=2e-6,
    units='um',
    probe_fourier_crop=pad)
model_original.translation_offsets.data += 0.7 * t.randn_like(model_original.translation_offsets)
model_original.weights.requires_grad = False

def reload_model():
    return copy.deepcopy(model_original)
    

# For running the optimizer class
numiter = 5

# Set stuff up for plots
fig, (ax1,ax2) = plt.subplots(1,2)

for option in ('old_method', 'optimizer'):
    time_list = []
    loss_hist_list = []

    # Iterate n-number of times for statistics
    for i in range(numiter):
        t.cuda.empty_cache()
        model = reload_model()
        device = 'cuda'
        model.to(device=device)
        dataset.get_as(device=device)
        # Construct a local time list
        local_time_list = []
        t_start = time.time()

        def report_n_record():
            print(model.report())
            local_time_list.append(time.time() - t_start)

        if option == 'optimizer':
            recon = cdtools.optimizer.Adam(model, dataset)
            for loss in recon.optimize(20, lr=0.005, batch_size=50):
                report_n_record()
            for loss in recon.optimize(20, lr=0.002, batch_size=100):
                report_n_record()
            for loss in recon.optimize(20, lr=0.001, batch_size=100):
                report_n_record()
        
        elif option == 'old_method':
            for loss in model.Adam_optimize(20, dataset, lr=0.005, batch_size=50):
                report_n_record()
            for loss in model.Adam_optimize(20, dataset, lr=0.002, batch_size=100):
                report_n_record()
            for loss in model.Adam_optimize(20, dataset, lr=0.001, batch_size=100):
                report_n_record()

        # After reconstructing, store the loss history and time values
        loss_hist_list.append(model.loss_history)
        time_list.append(local_time_list)

    # After testing either the new or old method, calculate the statistics and plot
    time_mean = np.array(time_list).mean(axis=0)/60
    time_std = np.array(time_list).std(axis=0)/60
    loss_mean = np.array(loss_hist_list).mean(axis=0)
    loss_std = np.array(loss_hist_list).std(axis=0)

    ax1.errorbar(time_mean, loss_mean, yerr=loss_std, xerr=time_std,
                    label=option)
    ax2.errorbar(np.arange(0,loss_mean.shape[0]), loss_mean, yerr=loss_std,
                label=option)

# Plot                     
fig.suptitle(f'Comparing old and new optimization refactor | {numiter} runs performed')
ax1.set_yscale('log')
ax1.set_xscale('linear')
ax2.set_yscale('log')
ax2.set_xscale('linear')
ax1.legend()
ax2.legend()
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Loss')
ax2.set_xlabel('Epochs')
plt.show()


