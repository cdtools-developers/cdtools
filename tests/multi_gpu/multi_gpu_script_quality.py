import cdtools
import os


@cdtools.tools.distributed.report_speed_test
def main():
    filename = os.environ.get('CDTOOLS_TESTING_DATA_PATH')
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

    # FancyPtycho is the workhorse model
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

    for loss in model.Adam_optimize(50, dataset, lr=0.02, batch_size=40):
        if model.rank == 0 and model.epoch % 10:
            print(model.report())
    for loss in model.Adam_optimize(25, dataset,  lr=0.005, batch_size=40):
        if model.rank == 0 and model.epoch % 10:
            print(model.report())
    for loss in model.Adam_optimize(25, dataset,  lr=0.001, batch_size=40):
        if model.rank == 0 and model.epoch % 10:
            print(model.report())

    model.tidy_probes()
    return model


if __name__ == '__main__':
    main()
