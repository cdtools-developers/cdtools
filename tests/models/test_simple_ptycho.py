import pytest
import time
import torch as t
from matplotlib import pyplot as plt

import cdtools

# Force all reconstructions to use the same RNG seed
t.manual_seed(0)


def test_simple_ptycho_from_results_dict(lab_ptycho_cxi, tmp_path):
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(lab_ptycho_cxi)

    t.manual_seed(42)
    model = cdtools.models.SimplePtycho.from_dataset(dataset)

    # Run a few epochs to get non-trivial state
    for loss in model.Adam_optimize(5, dataset, batch_size=10):
        pass

    # Test from_results_dict with in-memory dict (no dataset argument needed)
    results_dict = model.save_results()
    loaded_model = cdtools.models.SimplePtycho.from_results_dict(results_dict)

    # Test from_results_h5 via a temporary file
    h5_path = str(tmp_path / 'simple_ptycho_test.h5')
    model.save_to_h5(h5_path)
    loaded_model_h5 = cdtools.models.SimplePtycho.from_results_h5(h5_path)

    # Verify training metadata is restored
    assert loaded_model.epoch == model.epoch
    assert loaded_model.loss_history == model.loss_history

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


@pytest.mark.slow
def test_simple_ptycho(lab_ptycho_cxi, reconstruction_device, show_plot):
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(lab_ptycho_cxi)

    model = cdtools.models.SimplePtycho.from_dataset(dataset)

    model.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    for loss in model.Adam_optimize(100, dataset, batch_size=10):
        print(model.report())
        if show_plot:
            model.inspect(dataset, min_interval=10)

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)
        time.sleep(3)
        plt.close('all')

    # If this fails, the reconstruction got worse
    assert model.loss_history[-1] < 0.013
