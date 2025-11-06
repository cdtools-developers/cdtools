import numpy as np
import torch as t

from cdtools.tools import initializers
from cdtools.datasets import Ptycho2DDataset


def test_exit_wave_geometry():
    # First test a simple case where nothing need change
    basis = t.Tensor([[0, -30e-6, 0],
                      [-20e-6, 0, 0]]).transpose(0, 1)
    shape = t.Size([73, 56])
    wavelength = 1e-9
    distance = 1.
    rs_basis = initializers.exit_wave_geometry(basis, shape, wavelength, distance)

    assert t.allclose(rs_basis[0, 1], t.Tensor([-8.928571428571428e-07]))
    assert t.allclose(rs_basis[1, 0], t.Tensor([-4.5662100456621004e-07]))


def test_calc_object_setup():
    # First just try a simple case
    probe_shape = t.Size([120, 57])
    translations = t.rand((30, 2)) * 300
    t_max = t.max(translations, dim=0)[0]
    t_min = t.min(translations, dim=0)[0]
    obj_shape, min_translation = initializers.calc_object_setup(probe_shape, translations)
    exp_shape = t.ceil(t_max - t_min).to(t.int32) + t.Tensor(list(probe_shape)).to(t.int32)
    assert t.allclose(min_translation, t_min)

    assert obj_shape == t.Size(exp_shape)

    # Then add some padding
    padding = 5
    obj_shape, min_translation = initializers.calc_object_setup(probe_shape, translations, padding=padding)
    assert t.allclose(min_translation, t_min - padding)
    assert obj_shape == t.Size(exp_shape + 2 * padding)


def test_gaussian():
    # Generate gaussian as a numpy array (square array)
    shape = [10, 10]
    sigma = [2.5, 2.5]

    center = ((shape[0] - 1) / 2, (shape[1] - 1) / 2)
    y, x = np.mgrid[:shape[0], :shape[1]]
    np_result = 10 * np.exp(-0.5 * ((x - center[1]) / sigma[1])**2 - 0.5 * ((y - center[0]) / sigma[0])**2)
    init_result = initializers.gaussian(shape, sigma, amplitude=10).numpy()
    assert np.allclose(init_result, np_result)

    # Generate gaussian as a numpy array (rectangular array)
    shape = [10, 5]
    sigma = [2.5, 3]
    center = ((shape[0] - 1) / 2, (shape[1] - 1) / 2)
    y, x = np.mgrid[:shape[0], :shape[1]]
    np_result = np.exp(-0.5 * ((x - center[1]) / sigma[1])**2
                       - 0.5 * ((y - center[0]) / sigma[0])**2)
    init_result = initializers.gaussian(shape, sigma).numpy()
    assert np.allclose(init_result, np_result)

    # Generate gaussian with curvature
    shape = [20, 30]
    sigma = [2.5, 5]
    curvature = [1, 0.6]
    center = ((shape[0] - 1) / 2 + 3, (shape[1] - 1) / 2 - 1.4)
    y, x = np.mgrid[:shape[0], :shape[1]]
    np_result = (10 + 0j) * np.exp(-0.5 * ((x - center[1]) / sigma[1])**2 - 0.5 * ((y - center[0]) / sigma[0])**2)
    np_result *= np.exp(0.5j * curvature[1] * (x - center[1])**2 + 0.5j * curvature[0] * (y - center[0])**2)
    init_result = initializers.gaussian(shape, sigma, center=center, curvature=curvature, amplitude=10).numpy()
    assert np.allclose(init_result, np_result)


def test_gaussian_probe(ptycho_cxi_1):
    dataset = Ptycho2DDataset.from_cxi(ptycho_cxi_1[0])

    det_basis = t.Tensor(dataset.detector_geometry['basis'])
    det_shape = t.Size(dataset.patterns.shape[-2:])
    wavelength = dataset.wavelength
    distance = dataset.detector_geometry['distance']
    basis = initializers.exit_wave_geometry(det_basis,
                                            det_shape,
                                            wavelength,
                                            distance)
    # Basis is around 60nm in the i(y) direction, 85nm in the j(x) direction
    # Full window is therefore about 15 um in i(y) and 20 um in the j(x) dir

    # Come up with a roughly matching set of probe parameters
    sigma = 5e-7

    # Build a stage explicitly with numpy to compare against
    x = (np.arange(256) - 127.5) * (-basis[0, 1]).numpy()
    y = (np.arange(256) - 127.5) * (-basis[1, 0]).numpy()
    Xs, Ys = np.meshgrid(x, y)
    Rs = np.sqrt(Xs**2 + Ys**2)

    # Now we first test the non-propagated probe
    np_probe = np.exp(- 1 / (2 * sigma**2) * Rs**2)

    normalization = 0
    for params, im in dataset:
        normalization += np.sum(im.cpu().numpy())
    normalization /= len(dataset)

    normalization_1 = np.sqrt(normalization / np.sum(np.abs(np_probe)**2))

    probe = initializers.gaussian_probe(
        dataset, basis, det_shape, sigma).numpy()

    assert np.allclose(probe, normalization_1 * np_probe)

    # And then a propagated probe
    z = 1e-4  # nm
    k = 2 * np.pi / wavelength
    w0 = np.sqrt(2) * sigma
    zr = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + (z / zr)**2)
    Rz = z * (1 + (zr / z)**2)
    np_probe = np.exp(-Rs**2 / wz**2) * np.exp(-1j * k * Rs**2 / (2 * Rz))

    normalization_2 = np.sqrt(normalization / np.sum(np.abs(np_probe)**2))

    probe = initializers.gaussian_probe(dataset, basis, det_shape, sigma,
                                        propagation_distance=z).numpy()
    assert np.allclose(probe, normalization_2 * np_probe)


def test_SHARP_style_probe(ptycho_cxi_1):
    # This code will probably change and honestly it doesn't need to
    # be exactly the final thing. So just test that the function doesn't
    # throw an error.
    dataset = Ptycho2DDataset.from_cxi(ptycho_cxi_1[0])
    det_basis = t.Tensor(dataset.detector_geometry['basis'])
    det_shape = t.Size(dataset.patterns.shape[-2:])
    wavelength = dataset.wavelength
    distance = dataset.detector_geometry['distance']

    basis = initializers.exit_wave_geometry(det_basis,
                                            det_shape,
                                            wavelength,
                                            distance)

    assert basis.shape == t.Size([3, 2])

    probe = initializers.SHARP_style_probe(dataset)
    assert probe.shape == t.Size([256, 256])

    probe = initializers.SHARP_style_probe(dataset, propagation_distance=20e-6)
    assert probe.shape == t.Size([256, 256])


def test_RPI_spectral_init():
    # I think we can only really meaningfully test that it doesn't throw errors,
    # since the original implementation is in numpy and there aren't any clear
    # cases that can be calculated analytically.

    pattern = np.random.rand(230, 253).astype(np.float32)
    probe = np.random.rand(230, 253).astype(np.complex64)
    obj_shape = [37, 53]
    mask = t.Tensor(np.random.rand(*pattern.shape) > 0.04)
    background = t.as_tensor(np.random.rand(*pattern.shape), dtype=t.float32) * 0.05

    probe = t.as_tensor(probe)
    pattern = t.as_tensor(pattern)

    obj = initializers.RPI_spectral_init(pattern, probe, obj_shape)
    assert list(obj.shape) == [1] + obj_shape

    obj = initializers.RPI_spectral_init(pattern, probe, obj_shape,
                                         n_modes=2, mask=mask)
    assert list(obj.shape) == [2] + obj_shape

    obj = initializers.RPI_spectral_init(pattern, probe, obj_shape,
                                         n_modes=2, background=background)
    assert list(obj.shape) == [2] + obj_shape

    obj = initializers.RPI_spectral_init(pattern, probe, obj_shape,
                                         n_modes=2, mask=mask,
                                         background=background)
    assert list(obj.shape) == [2] + obj_shape
