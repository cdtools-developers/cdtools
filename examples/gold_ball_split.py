import cdtools
import torch as t

filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

pad = 10
dataset.pad(pad)

# This splits the dataset into a pseudorandomly chosen set of two disjoint
# datasets. The partitioning is drawn from a saved list, so the split is
# deterministic
dataset_1, dataset_2 = dataset.split()

datasets = [dataset_1, dataset_2, dataset]
labels = ['half_1', 'half_2', 'full']

for label, dataset in zip(labels, datasets):
    print(f'Working on dataset {label}')

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
    
    model.weights.requires_grad = False
    
    device = 'cuda'
    model.to(device=device)
    dataset.get_as(device=device)

    # Create the reconstructor
    recon = cdtools.reconstructors.AdamReconstructor(model, dataset)

    # For batched reconstructions like this, there's no need to live-plot
    # the progress
    for loss in recon.optimize(20, lr=0.005, batch_size=50):
        print(model.report())

    for loss in recon.optimize(50, lr=0.002, batch_size=100):
        print(model.report())

    for loss in recon.optimize(100, lr=0.001, batch_size=100,
                                    schedule=True):
        print(model.report())

        
    model.tidy_probes()

    model.save_to_h5(f'example_reconstructions/gold_balls_{label}.h5', dataset)
