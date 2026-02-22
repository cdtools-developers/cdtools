import cdtools
from cdtools.tools import multigpu
import pytest
import os
import subprocess
import torch as t

"""
This file contains several tests that are relevant to running multi-GPU
operations in CDTools.
"""


def reconstruct(rank, world_size, conn):
    """
    An example reconstruction script to test the performance of 1 vs 2 GPU
    operation.
    """
    filename = os.environ.get('CDTOOLS_TESTING_GOLD_BALL_PATH')
    cdtools.tools.multigpu.setup(rank=rank,
                                 world_size=world_size)
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

    device = 'cuda'
    model.to(device=device)
    dataset.get_as(device=device)

    recon = cdtools.reconstructors.AdamReconstructor(model,
                                                     dataset,
                                                     rank=rank,
                                                     world_size=world_size)

    for loss in recon.optimize(10, lr=0.005, batch_size=50):
        if rank == 0 and model.epoch == 10:
            print(model.report())
    conn.send((model.loss_times, model.loss_history))
    cdtools.tools.multigpu.cleanup()


@pytest.mark.multigpu
def test_plotting_saving_torchrun(lab_ptycho_cxi,
                                  multigpu_script,
                                  tmp_path,
                                  show_plot):
    """
    Run a multi-GPU test via torchrun on a script that executes several
    plotting and file-saving methods from CDIModel and ensure they run
    without failure.

    Also, make sure that only 1 GPU is generating the plots.

    If this test fails, one of three things happened:
        1) Either something failed while multigpu_script_2 was called
        2) Somehow, something aside from Rank 0 saved results
        3) multigpu_script_2 was not able to save all the data files
           we asked it to save.
    """
    # Run the test script, which generates several files that either have
    # the prefix
    cmd = ['torchrun',
           '--standalone',
           '--nnodes=1',
           '--nproc_per_node=2',
           multigpu_script]

    child_env = os.environ.copy()
    child_env['CDTOOLS_TESTING_DATA_PATH'] = lab_ptycho_cxi
    child_env['CDTOOLS_TESTING_TMP_PATH'] = str(tmp_path)
    child_env['CDTOOLS_TESTING_SHOW_PLOT'] = str(int(show_plot))

    try:
        subprocess.run(cmd, check=True, env=child_env)
    except subprocess.CalledProcessError:
        # The called script is designed to throw an exception.
        # TODO: Figure out how to distinguish between the engineered error
        # in the script versus any other error.
        pass

    # Check if all the generated file names only have the prefix 'RANK_0'
    filelist = [f for f in os.listdir(tmp_path)
                if os.path.isfile(os.path.join(tmp_path, f))]

    assert all([file.startswith('RANK_0') for file in filelist])
    print('All files have the RANK_0 prefix.')

    # Check if plots have been saved
    if show_plot:
        print('Plots generated: ' +
              f"{sum([file.startswith('RANK_0_test_plot') for file in filelist])}") # noqa
        assert any([file.startswith('RANK_0_test_plot') for file in filelist])
    else:
        print('--plot not enabled. Checks on plotting and figure saving' +
              ' will not be conducted.')

    # Check if we have all five data files saved
    file_output_suffix = ('test_save_checkpoint.pt',
                          'test_save_on_exit.h5',
                          'test_save_on_except.h5',
                          'test_save_to.h5',
                          'test_to_cxi.h5')

    print(f'{sum([file.endswith(file_output_suffix) for file in filelist])}'
          + ' out of 5 data files have been generated.')
    assert sum([file.endswith(file_output_suffix) for file in filelist]) \
        == len(file_output_suffix)


@pytest.mark.multigpu
def test_reconstruction_quality_spawn(gold_ball_cxi,
                                      show_plot):
    """
    Run a multi-GPU speed test based on gold_ball_ptycho_speedtest.py
    and make sure the final reconstructed loss using 2 GPUs is similar
    to 1 GPU.

    This test requires us to have 2 NVIDIA GPUs and makes use of the
    multi-GPU speed test.

    If this test fails, it indicates that the reconstruction quality is
    getting noticably worse with increased GPU counts. This may be a symptom
    of a synchronization/broadcasting issue between the different GPUs.
    """
    # Make the gold_ball_cxi file path visible to the reconstruct function
    os.environ['CDTOOLS_TESTING_GOLD_BALL_PATH'] = gold_ball_cxi

    loss_mean_list, loss_std_list, \
        _, _, speed_up_mean_list, speed_up_std_list\
        = multigpu.run_speed_test(fn=reconstruct,
                                  gpu_counts=(1, 2),
                                  runs=3,
                                  show_plot=show_plot)

    # Make sure that the final loss values between the 1 and 2 GPU tests
    # are comprable to within 1 std of each other.
    single_gpu_loss_mean = loss_mean_list[0][-1]
    single_gpu_loss_std = loss_std_list[0][-1]
    double_gpu_loss_mean = loss_mean_list[1][-1]
    double_gpu_loss_std = loss_std_list[1][-1]

    single_gpu_loss_min = single_gpu_loss_mean - single_gpu_loss_std
    single_gpu_loss_max = single_gpu_loss_mean + single_gpu_loss_std
    multi_gpu_loss_min = double_gpu_loss_mean - double_gpu_loss_std
    multi_gpu_loss_max = double_gpu_loss_mean + double_gpu_loss_std

    has_loss_overlap = \
        min(single_gpu_loss_max, multi_gpu_loss_max)\
        > max(single_gpu_loss_min, multi_gpu_loss_min)

    assert has_loss_overlap

    # Make sure the loss mean falls below 3.2e-4. The values of losses I
    # recorded at the time of testing were <3.19 e-4.
    assert double_gpu_loss_mean < 3.2e-4

    # Make sure that we have some speed up...
    assert speed_up_mean_list[0] < speed_up_mean_list[1]
