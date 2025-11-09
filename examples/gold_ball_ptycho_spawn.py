import cdtools
from matplotlib import pyplot as plt
import torch as t

# If you're noticing that the multi-GPU job is hanging (especially with 100%
# GPU use across all participating devices), you might want to try disabling
# the environment variable NCCL_P2P_DISABLE.
import os
os.environ['NCCL_P2P_DISABLE'] = str(int(True))


# The entire reconstruction script needs to be wrapped in a function
def reconstruct(rank, world_size):

    # In the multigpu setup, we need to explicitly define the rank and
    # world_size. The master address and master port should also be
    # defined if we don't specify the `init_method` parameter.
    cdtools.tools.multigpu.setup(rank=rank,
                                 world_size=world_size,
                                 master_addr='localhost',
                                 master_port='6666')

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
    model.translation_offsets.data += 0.7 * \
        t.randn_like(model.translation_offsets)
    model.weights.requires_grad = False

    # We need to manually define the rank parameter for the model, or else
    # all plots will be duplicated by the number of GPUs used.
    model.rank = rank

    device = 'cuda'
    model.to(device=device)
    dataset.get_as(device=device)

    # Rank and world_size also needs to be explicitly defined here
    recon = cdtools.reconstructors.AdamReconstructor(model,
                                                     dataset,
                                                     rank=rank,
                                                     world_size=world_size)

    with model.save_on_exception(
            'example_reconstructions/gold_balls_earlyexit.h5', dataset):

        for loss in recon.optimize(20, lr=0.005, batch_size=50):
            if rank == 0:
                print(model.report())
            if model.epoch % 10 == 0:
                model.inspect(dataset)

        for loss in recon.optimize(50, lr=0.002, batch_size=100,
                                   schedule=True):
            if rank == 0:
                print(model.report())

            if model.epoch % 10 == 0:
                model.inspect(dataset)

        for loss in recon.optimize(100, lr=0.001, batch_size=100,
                                   schedule=True):
            if rank == 0:
                print(model.report())
            if model.epoch % 10 == 0:
                model.inspect(dataset)

    cdtools.tools.multigpu.cleanup()

    model.tidy_probes()
    model.save_to_h5('example_reconstructions/gold_balls.h5', dataset)
    model.inspect(dataset)
    model.compare(dataset)
    plt.show()


if __name__ == '__main__':
    # Specify the number of GPUs we want to use, then spawn the multi-GPU job
    ngpus = 2
    t.multiprocessing.spawn(reconstruct, args=(ngpus,), nprocs=ngpus)
