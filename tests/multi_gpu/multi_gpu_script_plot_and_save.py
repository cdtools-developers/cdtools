import cdtools
from cdtools.tools import multigpu
import os
from matplotlib import pyplot as plt

rank = multigpu.get_rank()
world_size = multigpu.get_world_size()
cdtools.tools.multigpu.setup(rank=rank, world_size=world_size)

filename = os.environ.get('CDTOOLS_TESTING_DATA_PATH')
savedir = os.environ.get('CDTOOLS_TESTING_TMP_PATH')
SHOW_PLOT = bool(int(os.environ.get('CDTOOLS_TESTING_SHOW_PLOT')))

print('DONT CLOSE ANY OF THE FIGURES OR THE TEST WILL FAIL!')
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3,
    oversampling=2,
    probe_support_radius=120,
    propagation_distance=5e-3,
    units='mm',
    obj_view_crop=-50,
)

device = 'cuda'
model.to(device=device)
dataset.get_as(device=device)

# Test Ptycho2DDataset.inspect
if SHOW_PLOT:
    dataset.inspect()

# Test Ptycho2DDataset.to_cxi
filename_to_cxi = os.path.join(savedir,
                               f'RANK_{model.rank}_test_to_cxi.h5')
dataset.to_cxi(filename_to_cxi)

# Test CDIModel.save_to_h5
filename_save_to_h5 = os.path.join(savedir,
                                   f'RANK_{model.rank}_test_save_to.h5')
model.save_to_h5(filename_save_to_h5, dataset)

# Test CDIModel.save_on_exit(), CDIModel.inspect()
filename_save_on_exit = os.path.join(savedir,
                                     f'RANK_{model.rank}_test_save_on_exit.h5')

with model.save_on_exit(filename_save_on_exit, dataset):
    for loss in model.Adam_optimize(5, dataset, lr=0.02, batch_size=40):
        if rank == 0:
            print(model.report())
            if SHOW_PLOT:
                model.inspect(dataset)

if SHOW_PLOT:
    # Test CDIModel.compare(dataset)
    model.compare(dataset)

    # Test CDIModel.save_figures()
    filename_save_figures = os.path.join(savedir,
                                         f'RANK_{model.rank}_test_plot_')
    model.save_figures(prefix=filename_save_figures,
                       extension='.png')

    plt.close('all')

# Test CDIModel.save_checkpoint
filename_save_checkpoint = \
    os.path.join(savedir, f'RANK_{model.rank}_test_save_checkpoint.pt')
model.save_checkpoint(dataset, checkpoint_file=filename_save_checkpoint)

# Test CDIModel.save_on_exception()
filename_save_on_except = \
    os.path.join(savedir, f'RANK_{model.rank}_test_save_on_except.h5')

with model.save_on_exception(filename_save_on_except, dataset):
    for loss in model.Adam_optimize(10, dataset, lr=0.02, batch_size=40):
        if rank == 0 and model.epoch <= 10:
            print(model.report())
        elif model.epoch > 10:
            raise Exception('This is a deliberate exception raised to ' +
                            'test save on exception')

cdtools.tools.multigpu.cleanup()
