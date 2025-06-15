import cdtools
from matplotlib import pyplot as plt

# We need to import the distributed package from CDTools
from cdtools.tools.distributed import distributed

filename = r'example_data/lab_ptycho_data.cxi'

# Wrap the rest of the script inside of a function. This function will be
# distributed across several GPUs for multiprocessing at the end.

def multi_gpu_reconstruct():
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3,
        oversampling=2, 
        probe_support_radius=120, 
        propagation_distance=5e-3,
        units='mm', 
        obj_view_crop=-50)

    model.background.requires_grad=True

    device= f'cuda'
    model.to(device=device)
    dataset.get_as(device=device)

    for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=50):
        # We can still perform model.report, but we want only 1 GPU printing stuff.
        if model.rank == 0: 
            print(model.report())
        
        # You don't need to add the `if rank == 0` here. 
        if model.epoch % 20 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(50, dataset, lr=0.005, batch_size=50):
        if model.rank == 0: 
            print(model.report())
        
        if model.epoch % 20 == 0:
            model.inspect(dataset)

    #model.tidy_probes()
    model.inspect(dataset)

    # You don't need to add the `if rank == 0` here either...
    model.compare(dataset)
    
    # ...but you do have to add it here.
    if model.rank == 0: plt.show()
    

# This will execute the multi_gpu_reconstruct upon running this file
# Here, we're...
#   - ...setting up `world_size=4` GPUs to run
#   - ...telling CDTools the machine setting up all the connections (called
#        the "rank 0 node/machine") is on address `master_addr`
#   - ...telling CDTools we have a free port on `master_port` on the machine
#        with rank 0.
#   - ...going to wait 30 seconds for the GPUs to do something before 
#        we terminate the reconstruction. If you want to inspect/compare
#        the model after reconstruction, consider increasing the timeout.
#
# If you're using a single node (single machine/computer), you can try setting
# master_addr = 'localhost'.
if __name__ == '__main__':
    distributed.spawn(multi_gpu_reconstruct, 
                      device_ids = [0,1,2,3],
                      master_addr='localhost',
                      master_port='8888',
                      timeout=30)
