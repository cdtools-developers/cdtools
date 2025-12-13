import pytest
import torch as t

import cdtools

# Force all reconstructions to use the same RNG seed
t.manual_seed(0)


def test_center_probe(lab_ptycho_cxi):
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(lab_ptycho_cxi)
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3,
        fourier_probe=False
    )
    base_probe = model.probe.detach().clone()
    model.center_probes()
    centered_probe = model.probe.detach().clone()

    fourier_model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3,
        fourier_probe=True,
    )

    fourier_model.probe.data = cdtools.tools.propagators.far_field(
        base_probe
    )

    fourier_model.probe.detach().clone()
    fourier_model.center_probes()
    fourier_centered_probe = fourier_model.probe.detach().clone()
    ifft_fourier_centered_probe = cdtools.tools.propagators.inverse_far_field(
        fourier_centered_probe)

    # So we know the code had to do something
    assert not t.allclose(base_probe, centered_probe)
    # And checking that they both do the same thing, whether or not
    # fourier_probe was set to True
    assert t.allclose(
        centered_probe,
        ifft_fourier_centered_probe,
        atol=1e-4,
        rtol=1e-3
    )


@pytest.mark.slow
def test_lab_ptycho(lab_ptycho_cxi, reconstruction_device, show_plot):

    print('\nTesting performance on the standard transmission ptycho dataset')
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(lab_ptycho_cxi)

    # Test the masking system
    dataset.mask[110:115,65:70] = 0
    dataset.patterns[...,~dataset.mask] = t.max(dataset.patterns)
    
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3,
        oversampling=2,
        dm_rank=2,
        exponentiate_obj=True,
        probe_support_radius=120,
        propagation_distance=5e-3,
        units='mm',
        obj_view_crop=-50,
        use_qe_mask=True,  # test this in the case where no qe mask is defined
    )

    print('Running reconstruction on provided reconstruction_device,',
          reconstruction_device)
    model.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=10):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)

    for loss in model.Adam_optimize(50, dataset, lr=0.005, batch_size=50):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)
            
    for loss in model.Adam_optimize(25, dataset, lr=0.001, batch_size=50):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)

    model.tidy_probes()

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)

    # If this fails, the reconstruction has gotten worse
    assert model.loss_history[-1] < 0.0013

