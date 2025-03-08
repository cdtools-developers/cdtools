import cdtools
from matplotlib import pyplot as plt

# We need to import 2 additional functions
from torch.distributed import barrier
from cdtools.tools.distributed import distributed

filename = r'example_data/lab_ptycho_data.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3,
    oversampling=2, 
    probe_support_radius=120, 
    propagation_distance=5e-3,
    units='mm', 
    obj_view_crop=-50)

# Remove or comment out lines moving the dataset and model to GPU.
# This process will be handled by the cdtools.tools.distributed methods.

#device = 'cuda'
#model.to(device=device)
#dataset.get_as(device=device)


# Wrap the rest of the script inside of a function. This function will be
# distributed across several GPUs for multiprocessing at the end.
#
# CDTools multi-GPU methods expects the function to be declared as...
# 
#       def func(model, dataset, rank, world_size):
#
# ...where rank is an integer from [0, world_size-1] assigned to each
# GPU, and world_size is the total number of GPUs used.

def multi_gpu_reconstruct(model, dataset, rank, world_size):

    for loss in model.Adam_optimize(100, dataset, lr=0.02, batch_size=10):
        
        # We can still perform model.report, but we want only 1 GPU printing stuff.
        if rank == 0: 
            print(model.report())
        
        # You don't need to add the `if rank == 0` here. 
        if model.epoch % 20 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(100, dataset, lr=0.005, batch_size=50):
        if rank == 0: print(model.report())
        
        if model.epoch % 20 == 0:
            if rank == 0:
                model.inspect(dataset)

    model.tidy_probes()
    model.inspect(dataset)

    # You don't need to add the `if rank == 0` here either...
    model.compare(dataset)

    # ...but you do have to add it here.
    if rank == 0: plt.show()
    
# This will execute the multi_gpu_reconstruct upon running this file
# Here, we're...
#   - ...setting up `world_size=4` GPUs to run
#   - ...telling CDTools the machine setting up all the connections (called
#        the "rank 0 node/machine") is on address `master_addr`
#   - ...telling CDTools we have a free port on `master_port` on the machine
#        with rank 0.
#   - ...going to wait 60 seconds for the GPUs to do something before 
#        we terminate the reconstruction. If you want to inspect/compare
#        the model after reconstruction, consider increasing the timeout.
#
# If you're using a single node (single machine/computer), you can try setting
# master_addr = 'localhost'.
if __name__ == '__main__':
    distributed.spawn(multi_gpu_reconstruct, 
                      model=model,
                      dataset=dataset,
                      world_size = 8,
                      master_addr='localhost',
                      master_port='8888',
                      timeout=60)
