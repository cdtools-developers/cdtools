import pytest
import torch as t

import cdtools

# Force all reconstructions to use the same RNG seed
t.manual_seed(0)


@pytest.mark.slow
def test_simple_ptycho(lab_ptycho_cxi, reconstruction_device, show_plot):
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(lab_ptycho_cxi)

    model = cdtools.models.SimplePtycho.from_dataset(dataset)

    model.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    for loss in model.Adam_optimize(100, dataset, batch_size=10):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)

    # If this fails, the reconstruction got worse
    assert model.loss_history[-1] < 0.013
