"""This module contains tools to simulate various measurement models

All the measurements here are safe to use in an automatic differentiation
model. There exist tools to simulate detectors with finite saturation
thresholds, backgrounds, and more.
"""

import torch as t
import numpy as np
from torch.nn.functional import avg_pool2d

#
# This file will host tools to turn a propagated wavefield into a measured
# intensity pattern on a detector
#

__all__ = ['intensity', 'incoherent_sum', 'quadratic_background']


def intensity(wavefield, detector_slice=None, epsilon=1e-7, saturation=None, oversampling=1, simulate_finite_pixels=False):
    """Returns the intensity of a wavefield
    
    The intensity is defined as the magnitude squared of the
    wavefront. If a detector slice is given, the returned array
    will only include that slice from the simulated wavefront.
    
    Parameters
    ----------
    wavefield : torch.Tensor
        A JxMxN stack of complex-valued wavefields
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor
        A real MxN array storing the wavefield's intensities
    """
    if simulate_finite_pixels:
        inverse_fft = t.fft.fftshift(t.fft.ifft2(wavefield), dim=(-2,-1))
        pad1l = wavefield.shape[-2]//2
        pad1r = wavefield.shape[-2] - pad1l
        pad2l = wavefield.shape[-1]//2
        pad2r = wavefield.shape[-1] - pad2l
        padded = t.nn.functional.pad(inverse_fft, (pad1l, pad1r, pad2l, pad2r))
        upsampled_field = t.fft.fft2(t.fft.ifftshift(padded, dim=(-2,-2)))
        upsampled_intensity = t.abs(upsampled_field)**2
        ifft_intensity = t.fft.fftshift(t.fft.ifft2(upsampled_intensity), dim=(-2,-1))
        # Now we take a sinc function
        xs = t.arange(ifft_intensity.shape[-2])
        xs = (xs / t.max(xs)) * 2 - 1
        ys = t.arange(ifft_intensity.shape[-1])
        ys = (ys / t.max(ys)) * 2 - 1
        Xs, Ys = t.meshgrid(xs, ys, indexing='ij')
        mask = (t.special.sinc(Xs) * t.special.sinc(Ys)).to(device=ifft_intensity.device)
        blurred_intensity = t.fft.fft2(t.fft.ifftshift(mask * ifft_intensity, dim=(-2,-2)))
        output = blurred_intensity[...,::2,::2]
    else:
        output = t.abs(wavefield)**2

        
    
    # Now we apply oversampling
    if oversampling != 1:
        if wavefield.dim() == 2:
            output = avg_pool2d(output.unsqueeze(0), oversampling)[0]
        else:
            output = avg_pool2d(output, oversampling)

    # Then we grab the detector slice
    if detector_slice is not None:
        if wavefield.dim() == 2:
            output = output[detector_slice]
        else:
            output = output[(np.s_[:],) + detector_slice]

    # And now saturation    
    if saturation is None:
        return output + epsilon
    else:
        return t.clamp(output + epsilon,0,saturation)
            
            
def incoherent_sum(wavefields, detector_slice=None, epsilon=1e-7, saturation=None, oversampling=1, simulate_finite_pixels=False):
    """Returns the incoherent sum of the intensities of the wavefields
    
    The intensity is defined as the sum of the magnitudes squared of
    the wavefields. If a detector slice is given, the returned array
    will only include that slice from the simulated wavefronts.

    The (-3) index is the set of incoherently adding patterns, and any
    indexes further to the front correspond to the set of diffraction patterns
    to measure. The (-2) and (-1) indices are the wavefield, and the final
    index is the complex index
    
    Parameters
    ----------
    wavefields : torch.Tensor
        An LxJxMxNx stack of complex wavefields
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor 
        A real LXMxN array storing the incoherently summed intensities
    """
    if simulate_finite_pixels:
        inverse_fft = t.fft.fftshift(t.fft.ifft2(wavefields), dim=(-2,-1))
        pad1l = wavefields.shape[-2]//2
        pad1r = wavefields.shape[-2] - pad1l
        pad2l = wavefields.shape[-1]//2
        pad2r = wavefields.shape[-1] - pad2l
        padded = t.nn.functional.pad(inverse_fft, (pad1l, pad1r, pad2l, pad2r))
        upsampled_field = t.fft.fft2(t.fft.ifftshift(padded, dim=(-2,-2)))
        upsampled_intensity = t.sum(t.abs(upsampled_field)**2, dim=-3)
        ifft_intensity = t.fft.fftshift(t.fft.ifft2(upsampled_intensity), dim=(-2,-1))
        # Now we take a sinc function
        xs = t.arange(ifft_intensity.shape[-2])
        xs = (xs / t.max(xs)) * 2 - 1
        ys = t.arange(ifft_intensity.shape[-1])
        ys = (ys / t.max(ys)) * 2 - 1
        Xs, Ys = t.meshgrid(xs, ys, indexing='ij')
        mask = (t.special.sinc(Xs) * t.special.sinc(Ys)).to(device=ifft_intensity.device)
        blurred_intensity = t.fft.fft2(t.fft.ifftshift(mask * ifft_intensity, dim=(-2,-2)))
        output = t.abs(blurred_intensity[...,::2,::2])
    else:
        output = t.sum(t.abs(wavefields)**2,dim=-3) 

    # Now we apply oversampling
    if oversampling != 1:
        if wavefields.dim() == 3:
            output = avg_pool2d(output.unsqueeze(0), oversampling)[0]
        else:
            output = avg_pool2d(output, int(oversampling))
            
    # Then we grab the detector slice
    if detector_slice is not None:
        if wavefields.dim() == 3:
            output = output[detector_slice]
        else:
            output = output[(np.s_[:],) + detector_slice]
        
    if saturation is None:
        return output + epsilon
    else:
        return t.clamp(output + epsilon,0,saturation)


def quadratic_background(
        wavefield,
        background,
        *args,
        detector_slice=None,
        measurement=intensity,
        epsilon=1e-7,
        qe_mask=None,
        saturation=None,
        oversampling=1,
        simulate_finite_pixels=False
):
    """Returns the intensity of a wavefield plus a background
    
    The intensity is calculated via the given measurment function 
    Then, the square of the given background is added. This kind
    of background model is commonly used to enforce positivity of the
    background model.

    Parameters
    ----------
    wavefield : torch.Tensor
        A JxMxN stack of complex-valued wavefields
    background : torch.Tensor
        An tensor storing the square root of the detector background
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    measurement : function
        Default is measurements.intensity, the measurement function to use.
    qe_mask : torch.Tensor
        A tensor storing the per-pixel quantum efficiency (up to an unknown global scaling factor)
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor
        A real MxN array storing the wavefield's intensities
    """
    
    raw_intensity = measurement(
        wavefield,
        *args,
        detector_slice=detector_slice,
        epsilon=epsilon,
        oversampling=oversampling,
        simulate_finite_pixels=simulate_finite_pixels
    )

    if qe_mask is None:
        output = raw_intensity + background**2
    else:
        output = (qe_mask * raw_intensity) + background**2
        
    # This has to be done after the background is added, hence we replicate
    # it here
    if saturation is None:
        return output
    else:
        return t.clamp(output,0,saturation)
