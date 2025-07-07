import cdtools
import time
import pickle
import os


# Script is intended to be called by distributed_speed_test.py.
# We need to know which trial number this script is running
TRIAL_NUMBER = int(os.environ.get('CDTOOLS_TRIAL_NUMBER'))

# Create a list to keep track of when each module report was printed
t_list = []
# Start counting time
t_start = time.time()

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

device = 'cuda'
model.to(device=device)
dataset.get_as(device=device)

for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=40):
    if model.rank == 0:
        print(model.report())
        t_list.append(time.time() - t_start)

for loss in model.Adam_optimize(25, dataset,  lr=0.005, batch_size=40):
    if model.rank == 0:
        print(model.report())
        t_list.append(time.time() - t_start)

for loss in model.Adam_optimize(25, dataset,  lr=0.001, batch_size=40):
    if model.rank == 0:
        print(model.report())
        t_list.append(time.time() - t_start)

# This orthogonalizes the recovered probe modes
model.tidy_probes()

# Save the model and loss history
if model.rank == 0:
    # Set up the file name:
    file_name = f'speed_test_nGPUs_{model.world_size}_TRIAL_{TRIAL_NUMBER}'
    # Grab the loss and time history
    loss_history = model.loss_history
    time_history = t_list
    # Store quantities in a dictionary
    dict = {'loss history':loss_history,
            'time history':time_history,
            'nGPUs':model.world_size,
            'trial':TRIAL_NUMBER}
    # Save the quantities
    with open (f'example_loss_data/'+file_name+'.pkl', 'wb') as f:
        pickle.dump(dict, f)

