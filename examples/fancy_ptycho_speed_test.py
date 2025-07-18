import cdtools


# To modify fancy_ptycho.py for a multi-GPU speed test, we need to enclose the
# entire reconstruction script in a function. The function then needs to be
# decorated with cdtools.tools.distributed.report_speed_test. The decorator
# allows data to be saved and read by the multi-GPU speed test function
# which we will use to run this script.
@cdtools.tools.distributed.report_speed_test
def main():
    filename = 'example_data/lab_ptycho_data.cxi'
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3,
        oversampling=2,
        probe_support_radius=120,
        propagation_distance=5e-3,
        units='mm',
        obj_view_crop=-50
    )

    device = 'cuda'
    model.to(device=device)
    dataset.get_as(device=device)

    # Remove or comment out plotting existing plotting statements
    for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=40):
        # Optional: ensure that only a single GPU prints a report by
        # adding an if statement. Without this, the print statement will
        # be called by all participating GPUs, resulting in multiple copies
        # of the printed model report.
        if model.rank == 0:
            print(model.report())

    for loss in model.Adam_optimize(25, dataset,  lr=0.005, batch_size=40):
        if model.rank == 0:
            print(model.report())

    for loss in model.Adam_optimize(25, dataset,  lr=0.001, batch_size=40):
        if model.rank == 0:
            print(model.report())

    model.tidy_probes()

    # We need to return the model so the data can be saved by the decorator.
    return model


# We also need to include this if-name-main block at the end
if __name__ == '__main__':
    main()
