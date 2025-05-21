import pytest
import cdtools
import torch as t

import cdtools
from matplotlib import pyplot as plt

@pytest.mark.slow
def test_lab_ptycho(lab_ptycho_cxi, reconstruction_device, show_plot):

    print('\nTesting performance on the standard transmission ptycho dataset')
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(lab_ptycho_cxi)
    
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3, 
        oversampling=2,
        exponentiate_obj=True,
        dm_rank=2,
        probe_support_radius=120,
        propagation_distance=5e-3, 
        units='mm', 
        obj_view_crop=-50,
        use_qe_mask=True, # test this in the case where no qe mask is defined
    )
    
    print('Running reconstruction on provided reconstruction_device,',
          reconstruction_device)
    model.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    for loss in model.Adam_optimize(70, dataset, lr=0.02, batch_size=10):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(50, dataset,  lr=0.005, batch_size=50):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)
            
    model.tidy_probes()

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)

    # If this fails, the reconstruction has gotten worse
    assert model.loss_history[-1] < 0.001


@pytest.mark.slow
def test_gold_balls(gold_ball_cxi, reconstruction_device, show_plot):

    print('\nTesting performance on the standard gold balls dataset')
    
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(gold_ball_cxi)

    pad = 10
    dataset.pad(pad)

    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3,
        probe_support_radius=50,
        propagation_distance=2e-6,
        units='um',
        probe_fourier_crop=pad
    )

    model.translation_offsets.data += \
        0.7 * t.randn_like(model.translation_offsets)

    # Not much probe intensity instability in this dataset, no need for this
    model.weights.requires_grad = False

    print('Running reconstruction on provided --reconstruction_device,',
          reconstruction_device)
    model.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)
    
    for loss in model.Adam_optimize(20, dataset, lr=0.005, batch_size=50):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(50, dataset, lr=0.002, batch_size=100):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(100, dataset, lr=0.001, batch_size=100,
                                    schedule=True):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)
            
    model.tidy_probes()

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)

    # This just comes from running a reconstruction when it was working well
    # and choosing a rough value. If it triggers this assertion error,
    # something changed to make the final quality worse!
    assert model.loss_history[-1] < 0.0001


