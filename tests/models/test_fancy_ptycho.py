import pytest
import time
import torch as t
from matplotlib import pyplot as plt

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

def test_lab_ptycho_data_loading(lab_ptycho_cxi):
    
    print('\nTesting a few unusual data loading scenarios.')
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(lab_ptycho_cxi)
    
    # Test that it will properly load an initialization for the weights
    # from the intensities with OPRP on
    dataset.intensities = t.rand(len(dataset))
    
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=4,
        dm_rank=1,
    )

    # And test the case without OPRP    
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=2,
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
        panel_plot_mode=True, # test with panel plot mode,
        plot_level=4, # test with all plots
    )

    print('Running reconstruction on provided reconstruction_device,',
          reconstruction_device)
    #model.to(device=reconstruction_device)
    #dataset.get_as(device=reconstruction_device)

    for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=10):
        print(model.report())
        if show_plot:
            model.inspect(dataset, min_interval=10)

    for loss in model.Adam_optimize(50, dataset, lr=0.005, batch_size=50):
        print(model.report())
        if show_plot:
            model.inspect(dataset, min_interval=10)
            
    for loss in model.Adam_optimize(25, dataset, lr=0.001, batch_size=50):
        print(model.report())
        if show_plot:
            model.inspect(dataset, min_interval=10)

    model.tidy_probes()

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)
        time.sleep(3)
        plt.close('all')

    # If this fails, the reconstruction has gotten worse
    assert model.loss_history[-1] < 0.0013


@pytest.mark.slow
def test_near_field_ptycho(near_field_ptycho_cxi, reconstruction_device, show_plot):

    print('\nTesting performance on the standard transmission ptycho dataset')
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(near_field_ptycho_cxi)
    
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=1,
        near_field=True,
        propagation_distance=3.65e-3, # 3.65 downstream from focus
        panel_plot_mode=False, # test without panel plot mode
    )

    print('Running reconstruction on provided reconstruction_device,',
          reconstruction_device)
    model.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    for loss in model.Adam_optimize(100, dataset, lr=0.04, batch_size=10):
        print(model.report())
        if show_plot:
            model.inspect(dataset, min_interval=10)

    for loss in model.Adam_optimize(50, dataset, lr=0.005, batch_size=50):
        print(model.report())
        if show_plot:
            model.inspect(dataset, min_interval=10)

    model.tidy_probes()

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)
        time.sleep(3)
        plt.close('all')

    # If this fails, the reconstruction has gotten worse
    assert model.loss_history[-1] < 0.005


def test_fancy_ptycho_from_results_dict(lab_ptycho_cxi, tmp_path):
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(lab_ptycho_cxi)

    t.manual_seed(42)
    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=2,
    )

    # Verify original_translations is stored after from_dataset
    assert hasattr(model, 'original_translations')
    assert model.original_translations is not None

    # Run a few epochs to get non-trivial state
    for loss in model.Adam_optimize(5, dataset, batch_size=10):
        pass

    # Test from_results_dict with in-memory dict (no dataset argument needed)
    results_dict = model.save_results()
    loaded_model = cdtools.models.FancyPtycho.from_results_dict(results_dict)

    # Test from_results_h5 via a temporary file
    h5_path = str(tmp_path / 'fancy_ptycho_test.h5')
    model.save_to_h5(h5_path)
    loaded_model_h5 = cdtools.models.FancyPtycho.from_results_h5(h5_path)

    # Verify training metadata is restored
    assert loaded_model.epoch == model.epoch
    assert loaded_model.loss_history == model.loss_history

    # Verify original_translations round-trips correctly
    assert t.allclose(
        loaded_model.original_translations,
        model.original_translations,
    )

    # Verify all parameters and buffers are restored exactly
    original_sd = model.state_dict()
    loaded_sd = loaded_model.state_dict()
    loaded_h5_sd = loaded_model_h5.state_dict()
    for key in original_sd:
        assert t.allclose(original_sd[key].float(), loaded_sd[key].float()), \
            f'from_results_dict: state_dict mismatch for key {key}'
        assert t.allclose(original_sd[key].float(), loaded_h5_sd[key].float()), \
            f'from_results_h5: state_dict mismatch for key {key}'

    # Verify forward pass produces identical output
    (indices, translations), patterns = dataset[:5]
    with t.no_grad():
        original_out = model(indices, translations)
        loaded_out = loaded_model(indices, translations)
        loaded_h5_out = loaded_model_h5(indices, translations)

    assert t.allclose(original_out, loaded_out), \
        'from_results_dict: forward pass output mismatch'
    assert t.allclose(original_out, loaded_h5_out), \
        'from_results_h5: forward pass output mismatch'
