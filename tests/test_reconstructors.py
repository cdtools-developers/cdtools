import pytest
import cdtools
import torch as t
import numpy as np
import pickle
from matplotlib import pyplot as plt
from copy import deepcopy


@pytest.mark.slow
def test_Adam_gold_balls(gold_ball_cxi, reconstruction_device, show_plot):
    """
    This test checks out several things with the Au particle dataset
        1) Calls to Reconstructor.adjust_optimizer is updating the
           hyperparameters
        2) We are only using the single-GPU dataloading method
        3) Ensure `recon.model` points to the original `model`
        4) Reconstructions performed by `Adam.optimize` and
           `model.Adam_optimize` calls produce identical results when
           run over one round of optimization.
        5) The quality of the reconstruction remains below a specified
           threshold.
        5) Ensure that the FancyPtycho model works fine and dandy with the
           Reconstructors.
    """

    print('\nTesting performance on the standard gold balls dataset ' +
          'with reconstructors.AdamReconstructor')

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

    model.translation_offsets.data += 0.7 * \
        t.randn_like(model.translation_offsets)
    model.weights.requires_grad = False

    # Make a copy of the model
    model_recon = deepcopy(model)
    model.to(device=reconstruction_device)
    model_recon.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    # ******* Reconstructions with AdamReconstructor.optimize *******
    print('Running reconstruction using AdamReconstructor.optimize' +
          ' on provided reconstruction_device,', reconstruction_device)

    recon = cdtools.reconstructors.AdamReconstructor(model=model_recon,
                                                     dataset=dataset)
    t.manual_seed(0)

    # Run a reconstruction
    epoch_tup = (20, 50, 100)
    lr_tup = (0.005, 0.002, 0.001)
    batch_size_tup = (50, 100, 100)

    for i, iterations in enumerate(epoch_tup):
        for loss in recon.optimize(iterations,
                                   lr=lr_tup[i],
                                   batch_size=batch_size_tup[i]):
            print(model_recon.report())
            if show_plot and model_recon.epoch % 10 == 0:
                model_recon.inspect(dataset)

        # Check hyperparameter update
        assert recon.optimizer.param_groups[0]['lr'] == lr_tup[i]
        assert recon.data_loader.batch_size == batch_size_tup[i]

    # Ensure that recon does not have sampler as an attribute (only used in
    # multi-GPU)
    assert not hasattr(recon, 'sampler')

    # Ensure recon.model points to the original model
    assert id(model_recon) == id(recon.model)

    model_recon.tidy_probes()

    if show_plot:
        model_recon.inspect(dataset)
        model_recon.compare(dataset)

    # ******* Reconstructions with CDIModel.Adam_optimize *******
    print('Running reconstruction using CDIModel.Adam_optimize on provided' +
          ' reconstruction_device,', reconstruction_device)
    t.manual_seed(0)

    # We only need to test the first loop to ensure it's identical
    for i, iterations in enumerate(epoch_tup[:1]):
        for loss in model.Adam_optimize(iterations,
                                        dataset,
                                        lr=lr_tup[i],
                                        batch_size=batch_size_tup[i]):
            print(model.report())
            if show_plot and model.epoch % 10 == 0:
                model.inspect(dataset)

    model.tidy_probes()

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)

    # Ensure equivalency between the model reconstructions during the first
    # pass, where they should be identical
    assert np.allclose(model_recon.loss_history[:epoch_tup[0]], model.loss_history[:epoch_tup[0]])

    # Ensure reconstructions have reached a certain loss tolerance. This just
    # comes from running a reconstruction when it was working well and
    # choosing a rough value. If it triggers this assertion error, something
    # changed to make the final quality worse!
    assert model_recon.loss_history[-1] < 0.0001


@pytest.mark.slow
def test_LBFGS_RPI(optical_data_ss_cxi,
                   optical_ptycho_incoherent_pickle,
                   reconstruction_device,
                   show_plot):
    """
    This test checks out several things with the transmission RPI dataset
        1) Calls to Reconstructor.adjust_optimizer is updating the
           hyperparameters
        2) Ensure `recon.model` points to the original `model`
        3) Reconstructions performed by `LBFGS.optimize` and
           `model.LBFGS_optimize` calls produce identical results when
           run over one round of reconstruction.
        4) The quality of the reconstruction remains below a specified
           threshold.
        5) Ensure that the RPI model works fine and dandy with the
           Reconstructors.
    """
    with open(optical_ptycho_incoherent_pickle, 'rb') as f:
        ptycho_results = pickle.load(f)

    probe = ptycho_results['probe']
    background = ptycho_results['background']

    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(optical_data_ss_cxi)
    model = cdtools.models.RPI.from_dataset(dataset, probe, [500, 500],
                                            background=background, n_modes=2,
                                            initialization='random')

    # Prepare two sets of models for the comparative reconstruction
    model_recon = deepcopy(model)

    model.to(device=reconstruction_device)
    model_recon.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    # ******* Reconstructions with LBFGSReconstructor.optimize ******
    print('Running reconstruction using LBFGSReconstructor.' +
          'optimize on provided reconstruction_device,', reconstruction_device)

    recon = cdtools.reconstructors.LBFGSReconstructor(model=model_recon,
                                                      dataset=dataset)
    t.manual_seed(0)

    # Run a reconstruction
    reg_factor_tup = ([0.05, 0.05], [0.001, 0.1])
    epoch_tup = (30, 50)
    for i, iterations in enumerate(epoch_tup):
        for loss in recon.optimize(iterations,
                                   lr=0.4,
                                   regularization_factor=reg_factor_tup[i]):
            if show_plot and i == 0:
                model_recon.inspect(dataset)
            print(model_recon.report())

        # Check hyperparameter update (or lack thereof)
        assert recon.optimizer.param_groups[0]['lr'] == 0.4

    if show_plot:
        model_recon.inspect(dataset)
        model_recon.compare(dataset)

    # Check model pointing
    assert id(model_recon) == id(recon.model)

    # ******* Reconstructions with CDIModel.LBFGS_optimize ******
    print('Running reconstruction using CDIModel.LBFGS_optimize.' +
          'optimize on provided reconstruction_device,', reconstruction_device)
    t.manual_seed(0)
    for i, iterations in enumerate(epoch_tup[:1]):
        for loss in model.LBFGS_optimize(iterations,
                                         dataset,
                                         lr=0.4,
                                         regularization_factor=reg_factor_tup[i]): # noqa
            if show_plot and i == 0:
                model.inspect(dataset)
            print(model.report())

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)

    # Check loss equivalency between the two reconstructions
    assert np.allclose(model.loss_history[:epoch_tup[0]], model_recon.loss_history[:epoch_tup[0]])

    # The final loss when testing this was 2.28607e-3. Based on this, we set
    # a threshold of 2.3e-3 for the tested loss. If this value has been
    # exceeded, the reconstructions have gotten worse.
    assert model_recon.loss_history[-1] < 0.0023


@pytest.mark.slow
def test_SGD_gold_balls(gold_ball_cxi, reconstruction_device, show_plot):
    """
    This test checks out several things with the Au particle dataset
        1) Calls to Reconstructor.adjust_optimizer is updating the
           hyperparameters
        3) Ensure `recon.model` points to the original `model`
        4) Reconstructions performed by `SGD.optimize` and
           `model.SGD_optimize` calls produce identical results
           when run over one round of reconstruction.
        5) The quality of the reconstruction remains below a specified
           threshold.
        5) Ensure that the FancyPtycho model works fine and dandy with the
           Reconstructors.

    The hyperparameters used in this test are not optimized to produce
    a super-high-quality reconstruction. Instead, I just need A reconstruction
    to do some kind of comparative assessment.
    """
    print('\nTesting performance on the standard gold balls dataset ' +
          'with reconstructors.SGDReconstructor')

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

    model.translation_offsets.data += 0.7 * \
        t.randn_like(model.translation_offsets)
    model.weights.requires_grad = False

    # Make a copy of the model
    model_recon = deepcopy(model)
    model.to(device=reconstruction_device)
    model_recon.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    # ******* Reconstructions with SGDReconstructor.optimize *******
    print('Running reconstruction using SGDReconstructor.optimize' +
          ' on provided reconstruction_device,', reconstruction_device)

    recon = cdtools.reconstructors.SGDReconstructor(model=model_recon,
                                                    dataset=dataset)
    t.manual_seed(0)

    # Run a reconstruction
    epochs = 50
    lr = 0.00000005
    batch_size = 40

    for loss in recon.optimize(epochs,
                               lr=lr,
                               batch_size=batch_size):
        print(model_recon.report())
        if show_plot and model_recon.epoch % 10 == 0:
            model_recon.inspect(dataset)

    # Check hyperparameter update
    assert recon.optimizer.param_groups[0]['lr'] == lr
    assert recon.data_loader.batch_size == batch_size

    # Ensure that recon does not have sampler as an attribute (only used in
    # multi-GPU)
    assert not hasattr(recon, 'sampler')

    # Ensure recon.model points to the original model
    assert id(model_recon) == id(recon.model)

    model_recon.tidy_probes()

    if show_plot:
        model_recon.inspect(dataset)
        model_recon.compare(dataset)

    # ******* Reconstructions with cdtools.CDIModel.SGD_optimize *******
    print('Running reconstruction using CDIModel.SGD_optimize on provided' +
          ' reconstruction_device,', reconstruction_device)
    t.manual_seed(0)

    for loss in model.SGD_optimize(epochs,
                                   dataset,
                                   lr=lr,
                                   batch_size=batch_size):
        print(model.report())
        if show_plot and model.epoch % 10 == 0:
            model.inspect(dataset)

    model.tidy_probes()

    if show_plot:
        model.inspect(dataset)
        model.compare(dataset)

    # Ensure equivalency between the model reconstructions
    assert np.allclose(model_recon.loss_history[-1], model.loss_history[-1])

    # The final loss when testing this was 7.12188e-4. Based on this, we set
    # a threshold of 7.2e-4 for the tested loss. If this value has been
    # exceeded, the reconstructions have gotten worse.
    assert model.loss_history[-1] < 0.00072
