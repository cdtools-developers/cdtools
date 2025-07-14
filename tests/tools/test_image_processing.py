import numpy as np
import torch as t
from scipy import ndimage

from cdtools.tools import image_processing, interactions


def test_centroid():
    # Test single im
    im = t.rand((30, 40))
    sp_centroid = ndimage.center_of_mass(im.numpy())
    centroid = image_processing.centroid(im)
    assert t.allclose(centroid, t.Tensor(sp_centroid))

    # Test stack o' ims
    ims = t.rand((5, 30, 40))
    sp_centroids = [ndimage.center_of_mass(im.numpy())
                    for im in ims]
    centroids = image_processing.centroid(ims)
    assert t.allclose(centroids, t.Tensor(sp_centroids))


def test_centroid_sq():
    # Test single im
    im = t.rand((30, 40))
    sp_centroid = ndimage.center_of_mass(im.numpy()**2)
    centroid = image_processing.centroid_sq(im)
    assert t.allclose(centroid, t.Tensor(sp_centroid))

    # Test complex with multiple ims
    ims = t.rand((5, 30, 40)) + 1j * t.rand((5, 30, 40))
    np_ims = ims.numpy()
    sp_centroids = [ndimage.center_of_mass(np.abs(im)**2)
                    for im in np_ims]
    centroids = image_processing.centroid_sq(ims, comp=True)
    assert t.allclose(centroids, t.Tensor(np.array(sp_centroids)))


def test_sinc_subpixel_shift():

    im = np.zeros((512, 512), dtype=np.complex128)
    im[256, 256] = 1

    # test it by creating a single pixel object and seeing that it is
    # shifted correctly
    xs = np.arange(512) - 256
    Ys, Xs = np.meshgrid(xs, xs)
    sinc_im = np.sinc(Xs - 0.3) * np.sinc(Ys - 0.6)

    torch_im = t.as_tensor(im)
    test_im = image_processing.sinc_subpixel_shift(torch_im, (0.3, 0.6))

    # The fidelity isn't great due to the FFT-based approach, so we need
    # a pretty relaxed condition
    assert np.max(np.abs(sinc_im - test_im.numpy())) < 0.005


def test_find_pixel_shift():

    # Test two real ims
    big_im = t.rand((30, 70))
    im1 = big_im[3:, :-20]
    im2 = big_im[:-3, 20:]
    assert t.all(image_processing.find_pixel_shift(im1, im2) == t.LongTensor([-3, 20]))

    # Test a real and complex im
    big_im = t.rand((30, 70))
    im1 = big_im[:-5, 10:].to(dtype=t.complex64)
    im2 = big_im[5:, :-10]
    assert t.all(image_processing.find_pixel_shift(im1, im2) == t.LongTensor([5, -10]))
    assert t.all(image_processing.find_pixel_shift(im2, im1) == t.LongTensor([-5, 10]))

    # Test two complex ims
    big_im = t.rand((45, 45)) + 1j * t.rand((45, 45))
    im1 = big_im[:-5, :-4]
    im2 = big_im[5:, 4:]
    assert t.all(image_processing.find_pixel_shift(im1, im2) == t.LongTensor([5, 4]))


def test_find_subpixel_shift():
    # We can do this by creating a test probe and a test object
    test_probe = t.rand((70, 70)) + 1j * t.rand((70, 70))
    test_obj = t.ones((300, 300)) + 1j * t.rand((300, 300))

    shift = t.tensor((0.8, 0.75))

    im = interactions.ptycho_2D_sinc(test_probe, test_obj, shift, multiple_modes=False)

    retrieved_shift = image_processing.find_subpixel_shift(im, test_probe, search_around=(0, 0), resolution=50)
    # tolerance of 0.03 on this measurement
    assert t.all(t.abs(shift - retrieved_shift) < 0.03)


def test_find_shift():

    # We can do this by creating a test probe and a test object
    test_probe = t.rand((200, 200)) + 1j * t.rand((200, 200))
    test_obj = t.ones((300, 300)) + 1j * t.rand((300, 300))

    shift = t.tensor((0.8, 0.75))

    im = interactions.ptycho_2D_sinc(test_probe, test_obj, shift,
                                     multiple_modes=False)[:-40, :-6]

    retrieved_shift = image_processing.find_shift(im, test_probe[40:, 6:], resolution=50)
    # tolerance of 0.03 on this measurement
    assert t.all(t.abs(shift + t.Tensor((40, 6)) - retrieved_shift) < 0.03)


def test_convolve_1d():
    test_image = np.random.rand(400, 300)
    # test_image = np.hstack((np.ones((400,150)),np.zeros((400,150))))
    xs = np.linspace(-100, 100, 300)
    kernel = 1 / (1 + xs**2)

    # First, we test with everything real, dim=1
    convolved = image_processing.convolve_1d(t.as_tensor(test_image),
                                             t.as_tensor(kernel), dim=1)

    np_result = np.abs(np.fft.ifft(np.fft.fft(test_image, axis=1) * np.fft.fft(np.fft.ifftshift(kernel)), axis=1))
    assert np.allclose(convolved.numpy(), np_result)

    xs = np.linspace(-100, 100, 400)
    kernel = 1 / (1 + xs**2)

    # Then with dim=0, and a non-fftshifted kernel
    convolved = image_processing.convolve_1d(t.as_tensor(test_image),
                                             t.as_tensor(np.fft.ifftshift(kernel)),
                                             fftshift_kernel=False)

    np_result = np.abs(np.fft.ifft(np.fft.fft(test_image, axis=0) * np.fft.fft(np.fft.ifftshift(kernel))[:, None], axis=0))
    assert np.allclose(convolved.numpy(), np_result)

    # And finally with complex input
    convolved = image_processing.convolve_1d(t.as_tensor(test_image, dtype=t.complex64),
                                             t.as_tensor(kernel, dtype=t.complex64)).numpy()

    np_result = np.fft.ifft(np.fft.fft(test_image, axis=0) * np.fft.fft(np.fft.ifftshift(kernel))[:, None], axis=0)
    assert np.allclose(convolved, np_result)
