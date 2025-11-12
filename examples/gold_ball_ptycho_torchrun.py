import cdtools
from matplotlib import pyplot as plt
import torch as t

# At the beginning of the script we need to setup the multi-GPU job
# by initializing the process group and sycnronizing the RNG seed
# across all participating GPUs.
cdtools.tools.multigpu.setup()

# To avoid redundant print statements, we first grab the GPU "rank"
# (an ID number between 0 and max number of GPUs minus 1).
rank = cdtools.tools.multigpu.get_rank()

filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

pad = 10
dataset.pad(pad)
dataset.inspect()
model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3,
    probe_support_radius=50,
    propagation_distance=2e-6,
    units='um',
    probe_fourier_crop=pad
)
model.translation_offsets.data += 0.7 * t.randn_like(model.translation_offsets)
model.weights.requires_grad = False
device = 'cuda'
model.to(device=device)
dataset.get_as(device=device)

recon = cdtools.reconstructors.AdamReconstructor(model, dataset)

with model.save_on_exception(
        'example_reconstructions/gold_balls_earlyexit.h5', dataset):

    for loss in recon.optimize(20, lr=0.005, batch_size=50):
        # We ensure that only the GPU with rank of 0 runs print statement.
        if rank == 0:
            print(model.report())

        # But we don't need to do rank checking for any plotting- or saving-
        # related methods; this checking is handled internernally.
        if model.epoch % 10 == 0:
            model.inspect(dataset)

    for loss in recon.optimize(50, lr=0.002, batch_size=100):
        if rank == 0:
            print(model.report())

        if model.epoch % 10 == 0:
            model.inspect(dataset)

    for loss in recon.optimize(100, lr=0.001, batch_size=100, schedule=True):
        if rank == 0:
            print(model.report())
        if model.epoch % 10 == 0:
            model.inspect(dataset)

# After the reconstruction is completed, we need to cleanup things by
# destroying the process group.
cdtools.tools.multigpu.cleanup()

model.tidy_probes()
model.save_to_h5('example_reconstructions/gold_balls.h5', dataset)
model.inspect(dataset)
model.compare(dataset)
plt.show()
