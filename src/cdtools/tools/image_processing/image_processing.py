"""Contains functions for basic image processing needs

This module contains two kinds of image processing tools. The first type
is specific tools for calculating commonly needed metrics (such as the
centroid of an image), directly on complex-valued torch tensors. The second
kind of tools perform common image manipulations on torch tensors, in such
a way that it is safe to include them in automatic differentiation models.
"""

import numpy as np
import torch as t
from cdtools.tools import propagators

__all__ = ['hann_window', 'centroid', 'centroid_sq', 'sinc_subpixel_shift',
           'find_subpixel_shift', 'find_pixel_shift', 'find_shift',
           'convolve_1d', 'fourier_upsample', 'center']

def hann_window(im):
    """ Applies a hann window to a 2D image to apodize it

    TODO: update for pytorch
    
    Parameters
    ----------
    im: np.array
        A numpy array to apply a hann window to

    Returns
    -------
    apodidzed : np.array
        The image, apodized by a hann window
    """
    Xs, Ys = np.mgrid[:im.shape[0],:im.shape[1]]
    Xhann = np.sin(np.pi*Xs/(im.shape[1]-1))**2
    Yhann = np.sin(np.pi*Ys/(im.shape[0]-1))**2
    Hann = Xhann * Yhann
    return im * Hann

def centroid(im, dims=2):
    """Returns the centroid of an image or a stack of images

    By default, the last two dimensions are used in the calculation
    and the remainder of the dimensions are passed through.

    Beware that the meaning of the centroid is not well defined if your
    image contains values less than 0

    Parameters
    ----------
    im : torch.Tensor
        An image or stack of images to calculate from
    dims : int 
        Default 2, how many trailing dimensions to calculate with
    
    Returns
    -------
    centroid : torch.Tensor
        An (i,j) index or stack of indices
    """
    # For some reason this needs to be a list
    indices = [t.arange(im.shape[-dims+i]).to(t.float32) for i in range(dims)]
    indices = t.meshgrid(*indices, indexing='ij')

    use_dims = [-dims+i for i in range(dims)]
    divisor = t.sum(im, dim=use_dims)
    centroids = [t.sum(index * im, dim=use_dims) / divisor
                 for index in indices]

    return t.stack(centroids,dim=-1)


def centroid_sq(im, dims=2, comp=False):
    """Returns the centroid of the square of an image or stack of images

    By default, the last two dimensions are used in the calculation
    and the remainder of the dimensions are passed through.

    If the "comp" flag is set, it will be assumed that the last dimension
    represents the real and imaginary part of a complex number, and the
    centroid will be calculated for the magnitude squared of those numbers

    Parameters
    ----------
    im : torch.Tensor
        An image or stack of images to calculate from
    dims : int 
        Default 2, how many trailing dimensions to calculate for
    comp : bool
        Default is False, whether the data represents complex numbers

    Returns
    -------
    centroid: torch.Tensor
        An (i,j) index or stack of indices
    """
    if comp:
        im_sq = t.abs(im)**2
    else:
        im_sq = im**2

    return centroid(im_sq, dims=dims)


def sinc_subpixel_shift(im, shift):
    """Performs a subpixel shift with sinc interpolation on the given tensor

    The subpixel shift is done circularly via a multiplication with a linear
    phase mask in Fourier space.

    Parameters
    ----------
    im : torch.Tensor
        A complex-valued tensor to perform the subpixel shift on
    shift : array
        A length-2 array describing the shift to perform, in pixels

    Returns
    -------
    shifted_im : torch.Tensor
        The subpixel shifted tensor
    """

    i = t.arange(im.shape[0]) - im.shape[0]//2
    j = t.arange(im.shape[1]) - im.shape[1]//2
    I,J = t.meshgrid(i,j, indexing='ij')
    I = 2 * np.pi * I.to(t.float32) / im.shape[0]
    J = 2 * np.pi * J.to(t.float32) / im.shape[1]
    I = I.to(dtype=im.dtype,device=im.device)
    J = J.to(dtype=im.dtype,device=im.device)

    fft_im = t.fft.fftshift(t.fft.fft2(im),dim=(-2,-1))
    shifted_fft_im = fft_im * t.exp(1j * (-shift[0]*I - shift[1]*J))
    return t.fft.ifft2(t.fft.ifftshift(shifted_fft_im, dim=(-2,-1)))



def find_subpixel_shift(im1, im2, search_around=(0,0), resolution=10):
    """Calculates the subpixel shift between two images by maximizing the autocorrelation

    This function only searches in a 2 pixel by 2 pixel box around the
    specified search_around parameter. The calculation is done using the
    approach outlined in "Efficient subpixel image registration algorithms",
    Optics Express (2008) by Manual Guizar-Sicarios et al.

    Parameters
    ----------
    im1 : torch.Tensor
        The first real or complex-valued torch tensor
    im2 : torch.Tensor
        The second real or complex-valued torch tensor
    search_around : array
        Default (0,0), the shift to search in the vicinity of
    resolution : int
        Default is 10, the fraction of a pixel to calculate to

    Returns
    -------
    shift : torch.Tensor
        The relative shift (i,j) needed to best map im1 onto im2
    """

    #
    # Here's my approach, perhaps it's a little unconventional. I will first
    # calculate the phase correlation function as found in ____ (cite a paper
    # defining it). This is strongly peaked, so I can take a small window
    # of say, 10x10 pixels, and then do a sinc interpolation of that area
    # using an FFT with upsampling by a factor of resolution in reciprocal
    # space
    #
    cor_fft = t.fft.fft2(im1) * t.conj(t.fft.fft2(im2))

    # Not sure if this is more or less stable than just the correlation
    # maximum - requires some testing
    cor = t.fft.ifft2(cor_fft / t.abs(cor_fft))


    # Now, I need to shift the array to pull out a contiguous window
    # around the correlation maximum
    try:
        search_around = search_around.cpu()
    except:
        search_around = t.as_tensor(search_around)

    window_size = 15
    shift_zero = tuple(-search_around + t.tensor([window_size,window_size]))
    cor_window = t.roll(cor, shift_zero, dims=(-2,-1))[...,:2*window_size,:2*window_size]

    # Now we upsample this window
    cor_window_fft = t.fft.fftshift(t.fft.fft2(cor_window),dim=(-2,-1))
    upsampled = t.zeros(tuple(t.tensor(cor_window_fft.shape) * resolution),
                        dtype=cor.dtype,device=cor.device)

    upsampled[...,:2*window_size,:2*window_size] = cor_window_fft
    upsampled = t.roll(upsampled,(-window_size,-window_size),dims=(0,1))
    upsampled = t.roll(t.abs(t.fft.ifft2(upsampled))**2,
                       (-window_size*resolution,-window_size*resolution),
                       dims=(0,1))


    # And we extract the shift from the window
    sh = t.as_tensor(upsampled.shape, device=upsampled.device)
    cormax = t.as_tensor([t.div(t.argmax(upsampled), sh[1],
                                rounding_mode='floor'),
                          t.argmax(upsampled) % sh[1]],
                         device=upsampled.device)

    sh_over_2 = t.div(sh,2,rounding_mode='floor')
    subpixel_shift = ((cormax + sh_over_2) % sh - sh_over_2).to(dtype=upsampled.dtype)

    return search_around.to(device=upsampled.device, dtype=upsampled.dtype) + \
        subpixel_shift / resolution


def find_pixel_shift(im1, im2):
    """Calculates the integer pixel shift between two images by maximizing the autocorrelation

    This function simply takes the circular correlation with an FFT and
    returns the position of the maximum of that correlation. This corresponds
    to the amount that im1 would have to be shifted by to line up best with
    im2

    Parameters
    ----------
    im1 : torch.Tensor
        The first real or complex-valued torch tensor
    im2 : torch.Tensor
        The second real or complex-valued torch tensor

    Returns
    -------
    shift : torch.Tensor
        The integer-valued shift (i,j) that best maps im1 onto im2 
    """
    cor_fft = t.fft.fft2(im1) * t.conj(t.fft.fft2(im2))

    # Not sure if this is more or less stable than just the correlation
    # maximum - requires some testing
    cor = t.abs(t.fft.ifft2(cor_fft / t.abs(cor_fft)))


    sh = t.as_tensor(cor.shape,device=im1.device)
    cormax = t.tensor([t.div(t.argmax(cor),sh[1],rounding_mode='floor'),
                       t.argmax(cor) % sh[1]]).to(device=im1.device)

    sh_over_2 = t.div(sh,2,rounding_mode='floor')
    return (cormax + sh_over_2) % sh - sh_over_2



def find_shift(im1, im2, resolution=10):
    """Calculates the shift between two images by maximizing the autocorrelation

    This function starts by calculating the maximum shift to integer
    pixel resolution, and then searchers the nearby area to calculate a
    subpixel shift

    Parameters
    ----------
    im1 : torch.Tensor
        The first real or complex-valued torch tensor
    im2 : torch.Tensor
        The second real or complex-valued torch tensor
    resolution : int
        Default is 10, the fraction of a pixel to calculate to

    Returns
    -------
    shift : torch.Tensor
        The relative shift (i,j) needed to best map im1 onto im2
    """
    integer_shift = find_pixel_shift(im1,im2)
    subpixel_shift = find_subpixel_shift(im1, im2, search_around=integer_shift,
                                         resolution=resolution)

    return subpixel_shift


def convolve_1d(image, kernel, dim=0, fftshift_kernel=True):
    """Convolves an image with a 1d kernel along a specified dimension

    The convolution is a circular convolution calculated using a Fourier
    transform. The calculation is done so the input remains differentiable
    with respect to the output.

    If the image has a final dimension of 2, it is assumed to be complex.
    Otherwise, the image is assumed to be real. The image and kernel
    must either both be real or both be complex.
    
    Parameters
    ----------
    image : torch.Tensor
        The image to convolve
    kernel : torch.Tensor
        The 1d kernel to convolve with
    dim : int
        Default 0, the dimension to convolve along
    fftshift_kernel : bool
        Default True, whether to fftshift the kernel first.
    
    Returns
    -------
    convolved_im : torch.Tensor
        The convolved image
    """

    
    if fftshift_kernel:
        kernel = t.fft.ifftshift(kernel,dim=(-1,))

    # We have to transpose the relevant dimension to -1 before using the fft,
    # which expects to operate on the final dimension
    trans_im = t.transpose(image, dim, -1)
        
    # Take a correlation
    fft_im = t.fft.fft(trans_im)
    fft_kernel = t.fft.fft(kernel)
    trans_conv = t.fft.ifft(fft_im * fft_kernel)

    conv_im = t.transpose(trans_conv, dim, -1)

    return conv_im


def fourier_upsample(ims, upsample_factor=2, preserve_mean=False):
    # If preserve_mean is true, it preserves the mean pixel intensity
    # otherwise, it preserves the total summed intensity
    
    upsampled = t.zeros(ims.shape[:-2]+(upsample_factor*ims.shape[-2],
                                        upsample_factor*ims.shape[-1]),
                           dtype=ims.dtype,
                           device=ims.device)
    
    left = [((upsample_factor-1)*ims.shape[-2])//2,
            ((upsample_factor-1)*ims.shape[-1])//2]
    right = [left[0]+ims.shape[-2],
             left[1]+ims.shape[-1]]
    
    upsampled[...,left[0]:right[0],left[1]:right[1]] = propagators.far_field(ims)
    if preserve_mean:
        upsampled *= upsample_factor
    return propagators.inverse_far_field(upsampled)


def center(image, image_dims=2, use_power=True, iterations=4):
    """Automatically centers an image or stack of images
    
    This function is designed with probes for ptychography in mind, so the
    default centering method is to place the centroid of the magnitude-squared
    of the input image on the central pixel.

    # TODO I should place the centroid actually at the zero-frequency
    # pixel, rather than the center of the array, so this can be used in
    # Fourier space also.
    
    Because it's intended for probes, the function will sum over all extra
    dimensions beyond <image_dims> when calculating the centroid, and shift
    all the images by the same amount. It does *not* calculate a separate
    shift for each image in the stack

    It also centers by using circular shifts. This means that, after calculating
    the centroid position and shifting by that amount, the centroid will not
    be perfectly centered. To counteract this, multiple iterations are run,
    by default 4

    
    Parameters
    ----------
    image : torch.Tensor
        The ... x N x M image to center
    image_dims : int
        Default 2, the number of dimensions to center along
    use_power : bool
        Default True, whether to use the square of the magnitude
    iterations : int
        Default 4, the number of iterations to do
    
    Returns
    -------
    centered_im : torch.Tensor
        The centered image
    
    """
    # Make sure we dont screw with the input image
    image = t.clone(image)

    if image_dims !=2:
        raise NotImplementedError('Implementing centerings with dimension != '
                                  '2 requires modifying some other functions '
                                  'and is not yet implemented.')
    
    if image_dims > image.ndim:
        raise IndexError('Number of image dimensions cannot exceed the '
                         'dimensionality of the input')

    im_shape = image.shape
    # This adds an extra dimension if the image dimensionality is equal to
    # image_dims, and ravels any extra dimensions
    reshaped_im = image.reshape([-1, ] + list(im_shape[-image_dims:]))

    for i in range(iterations):
        if use_power:
            to_center = t.sum(t.abs(reshaped_im)**2, dim=0)
        else:
            to_center = t.sum(t.abs(reshaped_im), dim=0)
        
        im_centroid = centroid(to_center)
        
        for i in range(reshaped_im.shape[0]):
            reshaped_im[i] = sinc_subpixel_shift(
                reshaped_im[i],
                (-im_centroid[0] + im_shape[-2] / 2,
                 -im_centroid[1] + im_shape[-1] / 2))
            
    return reshaped_im.reshape(im_shape)
