import cdtools
import torch as t

# If you're noticing that the multi-GPU job is hanging (especially with 100%
# GPU use across all participating devices), you might want to try disabling
# the environment variable NCCL_P2P_DISABLE.
import os
os.environ['NCCL_P2P_DISABLE'] = str(int(True))


# For running a speed test, we need to add an additional `conn` parameter
# that the speed test uses to send loss-versus-time curves to the
# speed test function.
def reconstruct(rank, world_size, conn):

    # We define the setup in the same manner as we did in the spawn example
    # (speed test uses spawn)
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

    # Unless you're plotting data within the reconstruct function, setting
    # the rank parameter is not necessary.
    # model.rank = rank

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

        # It is recommended to comment out/remove all plotting-related methods
        # for the speed test.
        for loss in recon.optimize(20, lr=0.005, batch_size=50):
            if rank == 0:
                print(model.report())

        for loss in recon.optimize(50, lr=0.002, batch_size=100):
            if rank == 0:
                print(model.report())

        for loss in recon.optimize(100, lr=0.001, batch_size=100,
                                   schedule=True):
            if rank == 0:
                print(model.report())

    # We now use the conn parameter to send the loss-versus-time data to the
    # main process running the speed test.
    conn.send((model.loss_times, model.loss_history))

    # And, as always, we need to clean up at the end.
    cdtools.tools.multigpu.cleanup()


if __name__ == '__main__':
    # We call the run_speed_test function instead of calling
    # t.multiprocessing.spawn. We specify the number of runs
    # we want to perform per GPU count along with how many
    # GPU counts we want to test.
    #
    # Here, we test both 1 and 2 GPUs using 3 runs for both.
    # The show_plots will report the mean +- standard deviation
    # of loss-versus-time/epoch curves across the 3 runs.
    # The plot will also show the mean +- standard deviation of
    # the GPU-dependent speed ups across the 3 runs.
    cdtools.tools.multigpu.run_speed_test(reconstruct,
                                          gpu_counts=(1, 2),
                                          runs=3,
                                          show_plot=True)
