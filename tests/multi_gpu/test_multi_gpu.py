import cdtools.tools.distributed as dist
import pytest
import os
import subprocess

"""
This file contains several tests that are relevant to running multi-GPU
operations in CDTools.
"""


@pytest.mark.multigpu
def test_plotting_and_saving(lab_ptycho_cxi,
                             multigpu_script_2,
                             tmp_path,
                             show_plot):
    """
    Run a multi-GPU test on a script that executes several plotting and
    file-saving methods from CDIModel and ensure they run without failure.

    Also, make sure that only 1 GPU is generating the plots.

    If this test fails, one of three things happened:
        1) Either something failed while multigpu_script_2 was called
        2) Somehow, something aside from Rank 0 saved results
        3) multigpu_script_2 was not able to save all the data files
           we asked it to save.
    """
    # Pass the cxi directory to the reconstruction script
    # Define a temporary directory

    # Run the test script, which generates several files that either have
    # the prefix
    cmd = ['torchrun',
           '--standalone',
           '--nnodes=1',
           '--nproc_per_node=2',
           '-m',
           'cdtools.tools.distributed.single_to_multi_gpu',
           '--backend=nccl',
           '--timeout=30',
           '--nccl_p2p_disable=1',
           multigpu_script_2]

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
def test_reconstruction_quality(lab_ptycho_cxi,
                                multigpu_script_1,
                                tmp_path,
                                show_plot):
    """
    Run a multi-GPU test based on fancy_ptycho_speed_test.py and make
    sure the final reconstructed loss using 2 GPUs is similar to 1 GPU.

    This test requires us to have 2 NVIDIA GPUs and makes use of the
    multi-GPU speed test.

    If this test fails, it indicates that the reconstruction quality is
    getting noticably worse with increased GPU counts. This may be a symptom
    of a synchronization/broadcasting issue between the different GPUs.
    """
    # Pass the cxi directory to the reconstruction script
    os.environ['CDTOOLS_TESTING_DATA_PATH'] = lab_ptycho_cxi

    # Set up and run a distributed speed test
    world_sizes = [1, 2]
    runs = 5
    file_prefix = 'speed_test'

    # Define a temporary directory
    temp_dir = str(tmp_path)

    results = dist.run_speed_test(world_sizes=world_sizes,
                                  runs=runs,
                                  script_path=multigpu_script_1,
                                  output_dir=temp_dir,
                                  file_prefix=file_prefix,
                                  show_plot=show_plot,
                                  delete_output_files=True)

    # Ensure that both single and 2 GPU results produce losses lower than
    # a threshold value of 0.0013. This is the same threshold used in
    # test_fancy_ptycho.py
    loss_mean = results[0]
    assert loss_mean[0] < 0.0013
    assert loss_mean[1] < 0.0013

    # Check if the two losses are similar to each other by seeing if their
    # mean +- standard deviation intervals overlap with each other
    loss_std = results[1]
    single_gpu_loss_min = loss_mean[0] - loss_std[0]
    single_gpu_loss_max = loss_mean[0] + loss_std[0]
    multi_gpu_loss_min = loss_mean[1] - loss_std[1]
    multi_gpu_loss_max = loss_mean[1] + loss_std[1]
    has_overlap_loss = \
        min(single_gpu_loss_max, multi_gpu_loss_max)\
        > max(single_gpu_loss_min, multi_gpu_loss_min)

    print(f'Single GPU final loss: {loss_mean[0]} +- {loss_std[0]}')
    print(f'Two GPU final loss: {loss_mean[1]} +- {loss_std[1]}')
    print('Overlap between mean +- std of the single/multi GPU losses: ' +
          f'{has_overlap_loss}')

    assert has_overlap_loss

    # Also make sure that we actually get some kind of speed up with
    # multiple GPUs...
    speed_mean = results[2]
    speed_std = results[3]

    single_gpu_speed_min = speed_mean[0] - speed_std[0]
    single_gpu_speed_max = speed_mean[0] + speed_std[0]
    multi_gpu_speed_min = speed_mean[1] - speed_std[1]
    multi_gpu_speed_max = speed_mean[1] + speed_std[1]
    has_overlap_speed = \
        min(single_gpu_speed_max, multi_gpu_speed_max)\
        > max(single_gpu_speed_min, multi_gpu_speed_min)

    print(f'Single GPU runtime: {speed_mean[0]} +- {speed_std[0]}')
    print(f'Two GPU runtime: {speed_mean[1]} +- {speed_std[1]}')
    print('Overlap between the mean +- std of the single/multi GPU runtimes: '
          + f'{has_overlap_speed}')

    assert speed_mean[0] < speed_mean[1]
    assert not has_overlap_speed

    # Clear the environment variable we created here
    os.environ.pop('CDTOOLS_TESTING_DATA_PATH')
