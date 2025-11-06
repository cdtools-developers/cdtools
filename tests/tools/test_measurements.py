import torch as t
import numpy as np

from cdtools.tools import measurements


def test_intensity():
    wavefields = t.rand((5, 10, 10)) + 1j * t.rand((5, 10, 10))
    epsilon = 1e-6
    np_result = np.abs(wavefields.numpy())**2 + epsilon
    assert t.allclose(measurements.intensity(wavefields, epsilon=epsilon),
                      t.as_tensor(np_result))

    # Test single field case
    assert t.allclose(measurements.intensity(wavefields[0], epsilon=epsilon),
                      t.as_tensor(np_result[0]))

    det_slice = np.s_[3:, 5:8]
    assert t.allclose(measurements.intensity(wavefields, det_slice, epsilon=epsilon),
                      t.as_tensor(np_result[(np.s_[:],) + det_slice]))

    # Test single field case
    assert t.allclose(measurements.intensity(wavefields[0], det_slice, epsilon=epsilon),
                      t.as_tensor(np_result[0][det_slice]))

    # With oversampling on
    np_oversampling_result = (np_result[:, ::2, ::2] + np_result[:, 1::2, ::2] + np_result[:, ::2, 1::2] + np_result[:, 1::2, 1::2]) / 4

    # With multiple fields
    assert t.allclose(measurements.intensity(wavefields, epsilon=epsilon, oversampling=2),
                      t.as_tensor(np_oversampling_result,))

    # With a single field
    assert t.allclose(measurements.intensity(wavefields[0], epsilon=epsilon, oversampling=2),
                      t.as_tensor(np_oversampling_result[0],))


def test_incoherent_sum():

    # With no explicit slice given

    wavefields = t.rand((5, 4, 10, 10)) + 1j * t.rand((5, 4, 10, 10))
    epsilon = 1e-6
    np_result = np.sum(np.abs(wavefields.numpy())**2, axis=-3) + epsilon
    assert t.allclose(measurements.incoherent_sum(wavefields, epsilon=epsilon),
                      t.as_tensor(np_result))
    # Test single field case
    assert t.allclose(measurements.incoherent_sum(wavefields[0, :], epsilon=epsilon),
                      t.as_tensor(np_result[0]))

    # With a slice given
    det_slice = np.s_[3:, 5:8]
    assert t.allclose(measurements.incoherent_sum(wavefields, det_slice, epsilon=epsilon),
                      t.as_tensor(np_result[(np.s_[:],) + det_slice]))
    # Test single field case
    assert t.allclose(measurements.incoherent_sum(wavefields[0, :], det_slice, epsilon=epsilon),
                      t.as_tensor(np_result[0][det_slice]))

    # With oversampling on
    np_oversampling_result = (np_result[:, ::2, ::2] + np_result[:, 1::2, ::2] + np_result[:, ::2, 1::2] + np_result[:, 1::2, 1::2]) / 4

    # With multiple fields
    assert t.allclose(measurements.incoherent_sum(wavefields, epsilon=epsilon, oversampling=2),
                      t.as_tensor(np_oversampling_result,))

    # With a single field
    assert t.allclose(measurements.incoherent_sum(wavefields[0, :], epsilon=epsilon, oversampling=2),
                      t.as_tensor(np_oversampling_result[0],))


def test_quadratic_background():
    # test with intensity
    wavefields = t.rand((5, 10, 10)) + 1j * t.rand((5, 10, 10))
    epsilon = 1e-6
    background = t.rand((10, 10))
    np_result = np.abs(wavefields.numpy())**2 + background.numpy()**2 + epsilon
    det_slice = np.s_[3:, 5:8]

    result = measurements.quadratic_background(wavefields, background[det_slice],
                                               detector_slice=det_slice,
                                               epsilon=epsilon,
                                               measurement=measurements.intensity)
    assert t.allclose(result, t.tensor(np_result[(np.s_[:],) + det_slice]))

    # test with incoherent sum but no slice and no stack
    wavefields = t.rand((4, 10, 10)) + 1j * t.rand((4, 10, 10))
    np_result = np.sum(np.abs(wavefields.numpy())**2, axis=0)
    np_result += background.numpy()**2
    result = measurements.quadratic_background(wavefields, background,
                                               epsilon=epsilon,
                                               measurement=measurements.incoherent_sum)
    assert t.allclose(result, t.tensor(np_result))
