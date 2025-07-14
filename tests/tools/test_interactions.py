import numpy as np
from numpy.fft import fftshift, ifftshift
from numpy import fft
import pytest
import torch as t

from cdtools.tools import interactions


# Have a random probe and a random object and test the two
# functions for a variety of overlaps

# Also I want a probe that's just a single pixel
#

@pytest.fixture(scope='module')
def random_probe():
    return np.random.rand(256, 256) * np.exp(2j * np.pi * np.random.rand(256, 256))


@pytest.fixture(scope='module')
def random_obj():
    return np.random.rand(900, 900) * np.exp(2j * np.pi * np.random.rand(900, 900))


@pytest.fixture(scope='module')
def single_pixel_probe(scope='module'):
    probe = np.zeros((256, 256), dtype=np.complex128)
    probe[128, 128] = 1
    return probe


def test_translations_to_pixel():
    # First, try the case where everything is ones and simple
    basis = t.Tensor([[0, -1, 0], [-1, 0, 0]]).t()
    translations = t.rand((10, 3))
    output = interactions.translations_to_pixel(basis, translations)
    assert t.allclose(output, -translations[:, :2].flip(1))

    # Next, try a case with a single translation
    translation = t.rand((3))
    output = interactions.translations_to_pixel(basis, translation)
    assert t.allclose(output, -translation[:2].flip(0))

    # Then, try a case with no surface normal but with a real conversion
    basis = t.Tensor([[0, -2, 0], [-1, 0, 0.1]]).t()
    translations = t.rand((10, 3))
    output = interactions.translations_to_pixel(basis, translations)
    basis_vectors_inv = t.pinverse(basis)
    translations[:, 2] = 0  # manually project off z component
    assert t.allclose(output, t.mm(translations, basis_vectors_inv.t()))

    # Finally, try a case with a known surface normal (reflection)
    basis = t.Tensor([[0, -1, 0], [0, 0, 1]]).t()
    surface_normal = t.Tensor([np.sqrt(2), 0, -np.sqrt(2)])
    translations = t.rand((10, 3))
    output = interactions.translations_to_pixel(basis, translations,
                                                surface_normal=surface_normal)
    exp_translations = t.stack((-translations[:, 1], translations[:, 0]), dim=1)
    assert t.allclose(output, exp_translations)


def test_pixel_to_translations():
    # First, try the case where everything is ones and simple
    basis = t.Tensor([[0, -1, 0], [-1, 0, 0]]).t()
    translations = t.rand((10, 3))
    translations[:, 2] = 0
    output = interactions.translations_to_pixel(basis, translations)
    roundtrip = interactions.pixel_to_translations(basis, output)
    assert t.allclose(translations, roundtrip)

    # Next, try a case with a single translation
    translation = t.rand((3))
    translation[2] = 0
    output = interactions.translations_to_pixel(basis, translation)
    roundtrip = interactions.pixel_to_translations(basis, output)
    assert t.allclose(translation, roundtrip)

    # Then, try a case with no surface normal but with a real conversion
    basis = t.Tensor([[0, -2, 0], [-1, 0, 0.1]]).t()
    translations = t.rand((10, 3))
    translations[:, 2] = 0  # manually project off z component
    output = interactions.translations_to_pixel(basis, translations)
    roundtrip = interactions.pixel_to_translations(basis, output)
    assert t.allclose(translations, roundtrip)

    # Finally, try a case with a known surface normal (reflection)
    basis = t.Tensor([[0, -1, 0], [0, 0, 1]]).t()
    surface_normal = t.Tensor([np.sqrt(2), 0, -np.sqrt(2)])
    translations = t.rand((10, 3))
    translations[:, 2] = 0  # manually project off z component
    output = interactions.translations_to_pixel(basis, translations,
                                                surface_normal=surface_normal)
    roundtrip = interactions.pixel_to_translations(basis, output,
                                                   surface_normal=surface_normal)
    assert t.allclose(translations, roundtrip)


def test_project_translations_to_sample():
    # First, try the case where everything is ones and simple
    basis = t.Tensor([[0, -1, 0], [-1, 0, 0]]).t()
    translations = t.rand((10, 3))
    pixels, props = interactions.project_translations_to_sample(basis, translations)

    assert np.allclose(pixels[:, 0].numpy(), -translations[:, 1])
    assert np.allclose(pixels[:, 1].numpy(), -translations[:, 0])
    assert np.allclose(props.numpy(), -translations[:, 2:].numpy())

    # Next, a simple tilt along one axis. This is a 45 degree rotation
    # around the positive y-axis
    # Thus, y-axis translations are unaffected, but x-axis translations
    # induce a motion of 1/sqrt(2) in the j- pixel space, as well as
    # creating a propagation (negative propagation for positive x)
    basis = t.Tensor([[0, -1e-3, 0], [-np.sqrt(2) * 1e-3, 0, np.sqrt(2) * 1e-3]]).t()
    translations = t.rand((10, 3))
    pixels, props = interactions.project_translations_to_sample(basis, translations)

    print(props.numpy())
    print(-translations[:, 2:].numpy() - translations[:, :1].numpy())
    assert np.allclose(pixels[:, 0].numpy(), -translations[:, 1] * 1e3)
    assert np.allclose(pixels[:, 1].numpy(), -translations[:, 0] * 1e3 / np.sqrt(2))
    assert np.allclose(props.numpy(), -translations[:, 2:].numpy() - translations[:, :1].numpy())

    # Finally, we check a non-orthogonal case


def test_ptycho_2D_round(random_probe, random_obj):
    # Test a stack of images
    translations = np.random.rand(10, 2) * 500
    exit_waves_np = [random_probe * random_obj[tr[0]:tr[0] + random_probe.shape[0],
                     tr[1]:tr[1] + random_probe.shape[1]] for
                     tr in np.round(translations).astype(int)]
    exit_waves_t = interactions.ptycho_2D_round(t.as_tensor(random_probe),
                                                t.as_tensor(random_obj),
                                                t.as_tensor(translations))
    assert np.allclose(exit_waves_t.numpy(), exit_waves_np)

    # Test the single wave case
    exit_wave_t = interactions.ptycho_2D_round(t.as_tensor(random_probe),
                                               t.as_tensor(random_obj),
                                               t.as_tensor(translations[0]))
    assert np.allclose(exit_wave_t.numpy(), exit_waves_np[0])


def test_ptycho_2D_linear(single_pixel_probe, random_obj):

    # For this one, I just want to check one translation, but
    # I need to check both formats
    translations = np.array([[46.7, 53.2]])
    translation = np.array([46.7, 53.2])

    exit_waves_probe = interactions.ptycho_2D_linear(
        t.as_tensor(single_pixel_probe),
        t.as_tensor(random_obj),
        t.as_tensor(translations),
        shift_probe=True)

    exit_wave_probe = interactions.ptycho_2D_linear(
        t.as_tensor(single_pixel_probe),
        t.as_tensor(random_obj),
        t.tensor(translation),
        shift_probe=True)

    # Check that the outputs match
    assert t.allclose(exit_waves_probe[0], exit_wave_probe)

    exit_waves_obj = interactions.ptycho_2D_linear(t.as_tensor(single_pixel_probe), t.as_tensor(random_obj), t.tensor(translations), shift_probe=False)

    exit_wave_obj = interactions.ptycho_2D_linear(
        t.as_tensor(single_pixel_probe),
        t.as_tensor(random_obj),
        t.tensor(translation),
        shift_probe=False)

    # Check that the outputs match
    assert t.allclose(exit_waves_obj[0], exit_wave_obj)

    # For the shifted probe, we should find 4 pixels with intensity
    exit_waves_probe = t.as_tensor(exit_waves_probe)[0]

    probe_shift = np.array([[0.3 * 0.8, 0.3 * 0.2],
                            [0.7 * 0.8, 0.7 * 0.2]])
    obj_section = random_obj[128 + 46:128 + 48,
                             128 + 53:128 + 55]
    exit_section = exit_waves_probe[128:130, 128:130]
    assert np.allclose(probe_shift * obj_section, exit_section)

    # For the shifted obj, we should find one pixel with intensity
    exit_waves_obj = t.as_tensor(exit_waves_obj)[0]
    obj_shift = np.array([[0.3 * 0.8, 0.3 * 0.2],
                          [0.7 * 0.8, 0.7 * 0.2]])
    obj_section = random_obj[128 + 46:128 + 48,
                             128 + 53:128 + 55]
    exit_pixel = exit_waves_obj[128, 128]
    assert np.isclose(np.sum(obj_shift * obj_section), exit_pixel)

    # Test for a single translation


def test_ptycho_2D_sinc(single_pixel_probe, random_obj):

    # For this one, I just want to check one translation, but
    # I need to check both formats
    translations = np.array([[46.7, 53.2]])
    translation = np.array([46.7, 53.2])

    exit_waves_probe = interactions.ptycho_2D_sinc(
        t.as_tensor(single_pixel_probe),
        t.as_tensor(random_obj),
        t.as_tensor(translations),
        shift_probe=True)

    exit_wave_probe = interactions.ptycho_2D_sinc(
        t.as_tensor(single_pixel_probe),
        t.as_tensor(random_obj),
        t.as_tensor(translation),
        shift_probe=True)

    # Check that the outputs match
    assert t.allclose(exit_waves_probe[0], exit_wave_probe)

    # Now we explicitly define what the sinc interpolated array should
    # look like
    xs = np.arange(256) - 128
    Ys, Xs = np.meshgrid(xs, xs)
    sinc_probe = np.sinc(Xs) * np.sinc(Ys)
    # Just check that the unshifted probe is correct
    assert np.allclose(single_pixel_probe, sinc_probe)

    sinc_shifted_probe = np.sinc(Xs - 0.7) * np.sinc(Ys - 0.2)
    obj_section = random_obj[46:46 + 256,
                             53:53 + 256]
    exit_wave_np = sinc_shifted_probe * obj_section

    exit_wave_torch = exit_wave_probe.numpy()

    # The fidelity isn't great due to the FFT-based approach, so we need
    # a pretty relaxed condition
    assert np.max(np.abs(exit_wave_np - exit_wave_torch)) < 0.005


def test_RPI_interaction(random_probe, random_obj):

    random_obj1 = random_obj[:79, :68] * 0 + 1
    random_probe1 = random_probe * 0 + 1
    t_random_obj1 = t.as_tensor(random_obj1)
    t_random_probe1 = t.as_tensor(random_probe1)
    t_output1 = interactions.RPI_interaction(t_random_probe1, t_random_obj1)

    obj1_fourier = fftshift(fft.fft2(ifftshift(random_obj1), norm='ortho'))
    obj1_ups = np.zeros(random_probe1.shape[:2]).astype(np.complex128)
    obj1_ups[random_probe1.shape[0] // 2 - 79 // 2:
             -(random_probe1.shape[0] - 79 - (random_probe1.shape[0] // 2 - 79 // 2)),
             (random_probe1.shape[1] - 68) // 2:
             (random_probe1.shape[1] - 68) // 2 + 68] = obj1_fourier
    output1 = random_probe1 * fftshift(fft.ifft2(ifftshift(obj1_ups),
                                                 norm='ortho'))

    output1 = output1 * np.sqrt(output1.shape[-2] * output1.shape[-1] / (random_obj1.shape[-2] * random_obj1.shape[-1]))

    assert np.allclose(t_output1, output1)

    random_obj2 = np.stack([random_obj[:64, :89]] * 3)
    random_probe2 = random_probe[3:, 5:]
    t_random_obj2 = t.as_tensor(random_obj2)
    t_random_probe2 = t.as_tensor(random_probe2)
    t_output2 = interactions.RPI_interaction(t_random_probe2, t_random_obj2)

    obj2_fourier = fftshift(fft.fft2(ifftshift(random_obj2), norm='ortho'))
    obj2_ups = np.zeros((3,) + random_probe2.shape[:2]).astype(np.complex128)
    obj2_ups[:, (random_probe2.shape[0] - 64) // 2:
             (random_probe2.shape[0] - 64) // 2 + 64,
             (random_probe2.shape[1] - 89) // 2:
             (random_probe2.shape[1] - 89) // 2 + 89] = obj2_fourier
    output2 = random_probe2 * fftshift(fft.ifft2(ifftshift(obj2_ups),
                                                 norm='ortho'))

    output2 = output2 * np.sqrt(output2.shape[-2] * output2.shape[-1] / (random_obj2.shape[-2] * random_obj2.shape[-1]))

    assert np.allclose(t_output2, output2)
