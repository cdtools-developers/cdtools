import pytest
import cdtools
import torch as t
import numpy as np
from copy import deepcopy

@pytest.mark.slow
def test_gold_balls(gold_ball_cxi, reconstruction_device, show_plot):
    """
    This test checks out several things with the Au particle dataset
        1) Calls to Reconstructor.adjust_optimizer is updating the hyperparameters
        2) We are only using the single-GPU dataloading method
        3) Ensure `recon.model` points to the original `model` 
        4) Reconstructions performed by `Adam.optimize` and `model.Adam_optimize`
           calls produce identical results.
        5) The quality of the reconstruction remains below a specified threshold.
    """

    print('\nTesting performance on the standard gold balls dataset with reconstructors.Adam')
    
    # Setup dataset and model
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

    model.translation_offsets.data += 0.7 * t.randn_like(model.translation_offsets)
    model.weights.requires_grad = False

    # Make a copy of the model
    model_recon = deepcopy(model)

    # Load models and datasets to devices
    model.to(device=reconstruction_device)
    model_recon.to(device=reconstruction_device)
    dataset.get_as(device=reconstruction_device)

    # Make sure that we're not going to perform reconstructions on the same model
    assert id(model_recon) != id(model)

    ######### Reconstructions with cdtools.reconstructors.Adam.optimize #########
    #############################################################################

    print('Running reconstruction using cdtools.reconstructors.Adam.optimize on provided reconstruction_device,',
          reconstruction_device)

    recon = cdtools.reconstructors.Adam(model=model_recon, dataset=dataset)
    t.manual_seed(0)
    for loss in recon.optimize(20, lr=0.005, batch_size=50):
        print(model_recon.report())
        if show_plot and model_recon.epoch % 10 == 0:
            model_recon.inspect(dataset)

    # Test 1a: Ensure that the Adam.optimizer.param_groups learning rate and
    #          batch size got updated
    assert recon.optimizer.param_groups[0]['lr'] == 0.005
    assert recon.data_loader.batch_size == 50

    # Test 2: Ensure that recon does not have sampler as an attribute (for multi-GPU)
    assert not hasattr(recon, 'sampler')

    for loss in recon.optimize(50, lr=0.002, batch_size=100):
        print(model_recon.report())
        if show_plot and model_recon.epoch % 10 == 0:
            model_recon.inspect(dataset)

    # Test 1b: Ensure that the Adam.optimizer.param_groups learning rate and
    #          batch size got updated
    assert recon.optimizer.param_groups[0]['lr'] == 0.002
    assert recon.data_loader.batch_size == 100

    for loss in recon.optimize(100, lr=0.001, batch_size=100,
                                    schedule=True):
        print(model_recon.report())
        if show_plot and model_recon.epoch % 10 == 0:
            model_recon.inspect(dataset)
    
    # Test 1c: Ensure that the Adam.optimizer.param_groups learning rate and
    #          batch size got updated
    assert recon.optimizer.param_groups[0]['lr'] == 0.001
    assert recon.data_loader.batch_size == 100

    # Test 3:  Ensure recon.model points to the original model 
    assert id(model_recon) == id(recon.model)        
    
    model_recon.tidy_probes()

    if show_plot:
        model_recon.inspect(dataset)
        model_recon.compare(dataset)

    loss_recon = model_recon.loss_history[-1]

    ############ Reconstructions with cdtools.CDIModel.Adam_optimize ############
    #############################################################################
    print('Running reconstruction using CDIModel.Adam_optimize on provided reconstruction_device,',
          reconstruction_device)
    t.manual_seed(0)

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

    loss_model = model.loss_history[-1]

    # Test 4: Ensure equivalency between the model reconstructions
    assert np.allclose(loss_recon, loss_model)

    # Test 5: Ensure reconstructions have reached a certain loss tolerance
    #         This just comes from running a reconstruction when it was working well
    #         and choosing a rough value. If it triggers this assertion error,
    #         something changed to make the final quality worse!
    assert loss_model < 0.0001
