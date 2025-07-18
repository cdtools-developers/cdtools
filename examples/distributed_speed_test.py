from cdtools.tools.distributed import run_speed_test

# Define the number of GPUs to use for the test. We always need to include
# a single GPU in the test.
#
# Here, we will run trials with 1 and 2 GPUs.
world_sizes = [1, 2]

# We will run 3 trials per GPU to collect statistics on loss-versus-epoch/time
# data as well as runtime speedup.
runs = 3

# We will perform a speed test on a reconstruction script modified to run
# a speed test (see fancy_ptycho_speed_test.py)
script_path = 'fancy_ptycho_speed_test.py'

# When we run the modified script with the speed test, a pickle dump file
# will be generated after each trial. The file contains data about loss-vs-time
# measured for the trial with one or several GPUs used.
output_dir = 'example_loss_data'

# Define the file name prefix. The file will have the following name:
# `<file_prefix>_nGPUs_<world_size>_TRIAL_<run number>.pkl`
file_prefix = 'speed_test'

# We can plot several curves showing what the loss-versus/epoch curves look
# like for each GPU count. The plot will also show the relative runtime
# speed-up relative to the single-GPU runtime.
show_plot = True

# We can also delete the pickle dump files after each trial run has been
# completed and stored by `run_speed_test`
delete_output_file = True

# Run the test. This speed test will return several lists containing the
# means and standard deviations of the final recorded losses and runtime
# speed ups calculated over several trial runs. Each entry index maps to
# the GPU count specified by `world_sizes`.
final_loss_mean, final_loss_std, speed_up_mean, speed_up_std = \
    run_speed_test(world_sizes=world_sizes,
                   runs=runs,
                   script_path=script_path,
                   output_dir=output_dir,
                   file_prefix=file_prefix,
                   show_plot=show_plot,
                   delete_output_files=delete_output_file)
