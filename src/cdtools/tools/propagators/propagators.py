"""This module contains various propagators for light fields

All the functions here are designed for use in an automatic differentiation
ptychography model. Each function implements a different propagator.
"""

import torch as t
from torch.nn.functional import grid_sample
import numpy as np
from matplotlib import pyplot as plt

__all__ = ['far_field', 'near_field',
           'generate_angular_spectrum_propagator',
           'inverse_far_field', 'inverse_near_field',
           'generate_high_NA_k_intensity_map',
           'high_NA_far_field',
           'generate_generalized_angular_spectrum_propagator']

def far_field(wavefront):
    """Implements a far-field propagator in torch

    This accepts a torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield,
    and returns the far-field propagated version of it assuming it matches the
    detector dimensions. It assumes that the
    propagation is purely far-field, without checking that the geometry
    is consistent with that assumption.


    It also assumes that the real space wavefront is stored in an array
    [i,j] where i corresponds to the y-axis and j corresponds to the
    x-axis, with the origin following the CS standard of being in the
    upper right. The zero frequency component of the propagated wavefield is
    shifted to the center of the array.

    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts to be propagated
    
    Returns
    -------
    propagated : torch.Tensor
        The JxNxMx2 propagated wavefield
    """
    
    shifted = t.fft.ifftshift(wavefront, dim=(-1,-2))
    propagated = t.fft.fft2(shifted, norm='ortho')
    return t.fft.fftshift(propagated, dim=(-1,-2))


def inverse_far_field(wavefront):
    """Implements the inverse of the far-field propagator in torch

    This accepts a torch tensor, where the last dimension
    represents the real and imaginary components of the propagated wavefield,
    and returns the un-propagated array.

    It assumes that the real space wavefront is stored in an array
    [i,j] where i corresponds to the y-axis and j corresponds to the
    x-axis, with the origin following the CS standard of being in the
    upper right. The zero frequency component of the propagated wavefield is
    assumed to be the center of the array.

    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts propagated to the far-field
    
    Returns
    -------
    propagated : torch.Tensor
        The JxNxMx2 exit wavefield
    """
    shifted = t.fft.ifftshift(wavefront, dim=(-1,-2))
    propagated = t.fft.ifft2(shifted, norm='ortho')
    return t.fft.fftshift(propagated, dim=(-1,-2))


def generate_high_NA_k_intensity_map(sample_basis, det_basis,det_shape,distance, wavelength, *args, lens=False, **kwargs):
    """Generates k-space and intensity maps to allow for high-NA far-field propagation of light

    At high numerical apertures or for very tilted samples, the simple
    linear map between location on the detector and location in k-space
    starts to break down. In addition, at angles above roughly 15 degrees,
    a correction is needed to account for the decreasing solid angle
    intersected by each pixel on the detector.
    
    This function generates a map which can be used to apply both corrections
    via the high_NA_far_field propagator. The k-map which is output is
    defined as a map between pixel location on the detector and locations
    in the output of the standard, low-NA far-field propagated wavefield.
    The output coordinate system is defined to run from -1 to 1 in both
    directions - this allows for compatibility with pytorch's grid_sample
    function. Some detector pixels may be mapped to values outside the
    rangel [-1,1], depending on the respective sample and detector geometries.
    This is most likely to be the case if the major correction is due to a
    tilted sample.

    The intensity map is simply an object, the shape of the detector, which
    encodes intensity corrections between 0 and 1 per pixel.

    If the optional "lens" parameter is set to True, the intensity map will
    be set to a uniform map, and the distortion of Fourier space due to the
    flat nature of the detector (that is, the portion of the distortion
    that exists even if the sample is not tilted) will be disabled. This is
    to account for the fact that a good, infinity-conjugate imaging lens
    will do it's best to correct for these abberations in the lens. Of course,
    the lens will not be perfect, but in such a case it is a better
    approximation to assume that the lens is perfect than to assume that it
    is not there at all.

    Parameters
    ----------
    sample_basis: array
        The 3x2 sample basis, in real space
    det_basis: array
        The 3x2 detector basis, in real space
    det_shape: array
        The length-2 shape of the detector array, (N,M)
    distance: float
        The sample-to-detector distance
    wavelength: float
        The wavelength of light being propagated
    lens: bool
        Whether the diffraction pattern is formed by a lens or not.

    Returns
    -------
    k_map : torch.Tensor
        An NxMx2 tensor mapping detector pixels to locations in the low-NA propagated wavefield
    intensity_map : torch.Tensor
        An NxM tensor encoding the high-NA intensity correction
    """

    # First we convert things to numpy
    try:
        sample_basis = sample_basis.detach().cpu().numpy()
    except:
        pass

    try:
        det_basis = det_basis.detach().cpu().numpy()
    except:
        pass

    det_shape = np.array(tuple(det_shape))

    try:
        distance = distance.detach().cpu().numpy().ravel()[0]
    except:
        pass

    try:
        wavelength = wavelength.detach().cpu().numpy().ravel()[0]
    except:
        pass

    # The next order of business is to calculate the k values associated with
    # each pixel.

    i_arr = np.arange(det_shape[0])
    i_arr = i_arr - np.mean(i_arr)
    j_arr = np.arange(det_shape[1])
    j_arr = j_arr - np.mean(j_arr)
    Is, Js = np.meshgrid(i_arr,j_arr,indexing='ij')
    samp_det_vec = np.cross(det_basis[:,0],det_basis[:,1])
    samp_det_vec *= distance / np.linalg.norm(samp_det_vec)

    # This could potentially correct for a mistake in the implied
    # propagation direction (e.g. choosing e^ikx instead of e^-ikx)
    # This appears to be correct, based on empirical evidence from
    # a grazing incidence reflection experiment at 10 degrees
    # on the optical table
    samp_det_vec *= -1

    if lens == False:
        # This correctly reproduces the sample-to-each-pixel vectors
        # in the case where the diffraction pattern is actually formed
        # by Fraunhoffer diffraction
        Rs = np.tensordot(det_basis,np.stack([Is,Js]),axes=1) \
            + samp_det_vec[:,None,None]
    else:
        # This forms a distorted set of vectors designed to produce the
        # correct Fourier space map in the case where an imaging lens is
        # used in the 2f geometry. One should not read too much meaning
        # into these vectors, they are simply set up to produce the
        # correct final K-map
        Rs = np.tensordot(det_basis,np.stack([Is,Js]),axes=1)#
        Rs += (samp_det_vec / np.linalg.norm(samp_det_vec))[:,None,None] * \
            np.sqrt(np.sum((samp_det_vec)**2)-np.sum(Rs**2,axis=0))[None,:,:]
        
    k0 = 2*np.pi/wavelength
    
    R_magnitudes = np.linalg.norm(Rs, axis=0)
    Ks = k0 * Rs / R_magnitudes

    # My attempt at seeing what happens if I flip the Ks
    #Ks *= -1

    
    # This is the cosine of the angle with the detector normal
    intensity_map = np.tensordot(samp_det_vec/(k0*distance),Ks,axes=1)

    # And we add the factor of 1/R^2 which accounts for the increasing
    # distance from the source
    intensity_map = np.min(R_magnitudes)**2 * (intensity_map / R_magnitudes**2)
    
    if lens:
        # Set the intensity map to be uniform if a lens is being used
        intensity_map = np.ones_like(intensity_map)
    
    intensity_map = t.Tensor(intensity_map).to(*args, **kwargs)

    
    # This accounts for the implied phase ramp along the exit wave direction
    # In other words, it prevents the diffraction pattern from sliding off the
    # detector when the sample is tilted but represented by an object with
    # uniform phase.
    Ks -= k0 * samp_det_vec[:,None,None] / distance

    # A potential alternative when Ks are flipped
    #Ks += k0 * samp_det_vec[:,None,None] / distance

    
    # Now we move on to finding the conversion into k-space
    # for the sample grid. It turns out we can do this by multiplying
    # them with the real space basis (dual of the reciprocal space
    # basis is the real space basis). In fact, because we want to return
    # values scaled to the overall size of the k-space window, we don't
    # even need the shape of the sample array

    k_map = np.tensordot(2*sample_basis.transpose()[::-1,:] / (2*np.pi),Ks,axes=1)
    k_map = t.Tensor(np.moveaxis(k_map,0,2)).to(*args, **kwargs)
    
    # Potentially we need a correction to account for the discrete nature
    # of the FFT
    
    # And finally, we need to convert the results to pytorch
    
    return k_map, intensity_map




def high_NA_far_field(wavefront, k_map, intensity_map=None):
    """Performs a far-field propagation step including a correction for high-NA scenarios

    Two major corrections need to be performed when propagating light fields
    into the far field at high numerical aperture or when the sample is
    tilted as compared to the detector. The first correction is a deviation
    from the linear relationship between detector position and spatial
    frequency in the near field. This is accounted for with the k_map
    argument, as generated by the generate_high_NA_k_intensity_map 
    function.

    The second correction is the change in the solid angle which each pixel
    subtends at high NA. This is accounted for with an optional intensity
    map. This is kept optional because some detectors - specifically, those
    for penetrating radiation - may either not need a correction or need
    a different correction due to the volumetric nature of the pixels.
    
    If the k-map maps any pixels on the detector to pixels outside of the
    k-space range of the wavefront, these will be set to zero. This is in
    keeping with the typical assumption that the sample is band-limited to
    the Nyquist frequency for the array on which it is sampled.

    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts propagated to the far-field
    k_map : torch.Tensor
        The NxMx2 map accounting for high NA distortion, as generated by generate_high_NA_k_intensity_map
    intensity_map : torch.Tensor
        The optional NxM tensor accounting for the intensity variation across the detector

    
    Returns
    -------
    propagated : torch.Tensor
        The JxNxMx2 exit wavefield


    """
    low_NA_wavefield = far_field(wavefront)
    # I'm going to need to separately interpolate the real and complex parts
    # This can be done
    
    k_map = k_map[None,:,:,:]
    # Will only work for a 4D wavefile stack.
    #plt.figure()
    #plt.pcolormesh(k_map[0,:,:,0].cpu().numpy(),k_map[0,:,:,1].cpu().numpy(),
    #               np.ones_like(k_map[0,:-1,:-1,0].cpu().numpy()))
    #plt.show()
    def process_wavefield_stack(low_NA_wavefield):
        # grid_sample doesn't work on complex-valued wavefields
        real_output = grid_sample(low_NA_wavefield[None,:,:,:].real,k_map,mode='bilinear',padding_mode='zeros', align_corners=False)
        imag_output = grid_sample(low_NA_wavefield[None,:,:,:].imag,k_map,mode='bilinear',padding_mode='zeros', align_corners=False)

        result = real_output[0,:,:,:] + 1j * imag_output[0,:,:,:]
    
        if intensity_map is not None:
            result = result * intensity_map[None,:,:]

        return result

    original_dim = wavefront.dim()
    if original_dim == 2:
        result = process_wavefield_stack(low_NA_wavefield[None,:,:])
        return result[0,:,:]
    if original_dim == 3:
        result = process_wavefield_stack(low_NA_wavefield)
        return result
    if original_dim == 4:
        result = []
        for i in range(low_NA_wavefield.size()[0]):
            result.append(process_wavefield_stack(low_NA_wavefield[i,:,:,:]))
        return t.stack(result)
    else:
        raise IndexError('Wavefield had incorrect number of dimensions')



def generate_angular_spectrum_propagator(shape, spacing, wavelength, z, *args, bandlimit=None, **kwargs):
    """Generates an angular-spectrum based near-field propagator from experimental quantities
    
    This function generates an angular-spectrum based near field
    propagator that will work on torch Tensors. The function is structured
    this way - to generate the propagator first - because the
    generation of the propagation mask is a bit expensive and if this
    propagator is used in a reconstruction program, then it will be best
    to calculate this mask once and reuse it.

    Formally, this propagator is the the fourier transform of the convolution
    kernel for light propagation in free space
    
    If the optional bandlimit parameter is set, the propagator will be set
    to zero beyond an explicit bandlimiting frequency. This is helpful if the
    propagator will be used in a repeated multiply/propagate framework such
    as a multislice algorithm, where it helps to prevent aliasing.
    
    Parameters
    ----------
    shape : array
        The shape of the arrays to be propagated
    spacing : array
        The pixel size in each dimension of the arrays to be propagated
    wavelength : float
        The wavelength of light to simulate propagation of
    z : float
        The distance to simulate propagation over
    bandlimit : float
        Optional, a fraction of the full detector radius beyond which to set the propagator to zero.

    Returns
    -------
    propagator : torch.Tensor
        A phase mask which accounts for the phase change that each plane wave will undergo.
    """
    
    # Internally, the generalized propagation function is used, so we start
    # by creating an appropriate basis
    # This creates a real-valued tensor which matches the kind of complex
    # number dtype requested in **kwargs
    if 'dtype' in kwargs:
        basis = t.real(t.zeros([3,2], **kwargs))
    else:
        basis = t.zeros([3,2], dtype=t.float32)
    spacing = t.as_tensor(spacing, dtype=basis.dtype)

    basis[1,0] = -spacing[0]
    basis[0,1] = -spacing[1]
    # And similarly, the offset is just z along the z direction
    #offset = t.tensor([0,0,z], dtype=basis.dtype)
    offset = t.zeros(3, dtype=basis.dtype)
    offset[2] = z
    
    # And we call the generalized function! 
    propagator = generate_generalized_angular_spectrum_propagator(
        shape,
        basis,
        wavelength,
        offset, 
        **kwargs)
    if z < 0:
        propagator = t.conj(propagator)


    # Bandlimiting is not implemented in the generalized function, because it
    # has a less clear meaning in that setting, so we apply it here instead
    if bandlimit is not None:
        # No need to multiply by 2pi
        ki = 2 * np.pi * t.fft.fftfreq(shape[0],spacing[0])
        kj = 2 * np.pi * t.fft.fftfreq(shape[1],spacing[1])
        Ki, Kj = t.meshgrid(ki,kj, indexing='ij')
        min_radius = min(t.max(ki),t.max(kj))
        Rs = t.sqrt((Ki/t.max(ki))**2 + (Kj/t.max(kj))**2)
        propagator = propagator * (Rs < bandlimit)
        
    return propagator



def generate_generalized_angular_spectrum_propagator(shape, basis, wavelength, offset_vector, *args, propagation_vector=None, propagate_along_offset=False, **kwargs):
    """Generates an angular-spectrum based near-field propagator from experimental quantities

    This function generates an angular-spectrum based near field
    propagator that will work on torch Tensors. The function is structured
    this way - to generate the propagator first - because the
    generation of the propagation mask is a bit expensive and if this
    propagator is used in a reconstruction program, it will be best
    to calculate this mask once and then reuse it it.

    Formally, this propagator is the fourier transform of the convolution
    kernel for light propagation in free space. It will map a light field
    at an input plane, with the size and shape defined by the shape and basis
    inputs, and map it to a plane of the same size and shape offset by the
    offset vector. It is designed to work on any wavefield defined on an
    array of parallelograms.


    In addition, if the propagation_vector is set, there is an assumed
    phase ramp  applied to the wavefield before propagation, defined such
    that a feature with uniform phase will propagate along the direction of
    the propagation vector. This will also remove the phase variation along
    the propagation direction, because it makes the most physical sense to
    regard this choice as removing the dominant phase variation in 3D, allowing
    for the generation of a smoothly varying wavefield over 3D volumes.
    This decision provides the best numerical stability and allows for the
    simple setup of light fields copropagating with the coordinate system.
    
    If the propagate_along_offset option is set to True, then the propagation
    vector will be set equal to the offset vector. This overrides the
    propagation_vector option

    Note that, unlike in the case of the simple angular spectrum propagator,
    the direction of "forward propagation" is defined by the propagation
    vector, or (if the propagation vector is not defined), the offset
    vector. Therefore, in the simple case of a perpendicular offset,
    there will be no difference between using an offset vector or the
    negative of the  offset vector. This is because, for the light
    propagation problem to be well posed, the assumption must be made that
    light only passes through the plane of the known wavefield in one
    direction. We always assume that light passes through the initial plane
    travelling in the direction of the final plane.

    Practically, if one wants to simulate inverse propagation, there are 
    two possible approaches. First, one can use the inverse_near_field 
    function, which simulates the inverse propagation problem and therefore
    will naturally simulate propagation in the opposite direction. Second,
    one can explicitly include a propagation_vector argument, in the direction
    opposite to the offset vector. Note for both cases that this function
    will only return propagators which suppress evanescent waves. Thus,
    propagating forward and then backward by either of these two methods
    will lead to a supression of evanescant waves. If you need a propagator
    that will cause evanescent waves to undergo exponential growth, good
    for you, but this function will not provide it for you.    

    Parameters
    ----------
    shape : array
        The shape of the arrays to be propagated
    basis : array
        The (2x3) set of basis vectors describing the array to be propagated
    wavelength : float
        The wavelength of light to simulate propagation of
    offset_vector : array
        The displacement to propagate the wavefield along.
    propagation_vector : array
        The vector along which to include an implied phase ramp to propagate uniform phase features along, if set
    propagate_along_offset : bool
        Overrides propagation_vector, sets the propagation vector to equal the offset vector if set.

    Returns
    -------
    propagator : torch.Tensor
        A phase mask which accounts for the phase change that each plane wave will undergo.
    """

    # make sure everything is in pytorch, and set the propagation vector
    # appropriately if propagate_along_offset is chosen
    basis = t.as_tensor(basis)
    offset_vector = t.as_tensor(offset_vector, dtype=basis.dtype)

    if propagate_along_offset:
        propagation_vector = offset_vector
        
    if propagation_vector is not None:
        propagation_vector = t.as_tensor(propagation_vector, dtype=basis.dtype)

    #
    # In this section, we calculate the wavevectors associated with each
    # pixel in Fourier space. This is the meat of the function
    #
    
    # First we calculate a dual basis for the real space grid
    inv_basis =  t.linalg.pinv(basis).transpose(0,1)

    # Then we calculate the frequencies in (i,j) space
    ki = 2 * np.pi * t.fft.fftfreq(shape[0], dtype=inv_basis.dtype)
    kj = 2 * np.pi * t.fft.fftfreq(shape[1], dtype=inv_basis.dtype)
    K_ij = t.stack(t.meshgrid(ki,kj, indexing='ij'))
    
    # Now we convert these to frequencies in reciprocal space
    # These frequencies span the 2D plane of the input wavefield,
    # hence K_ip for "in-plane"
    K_ip = t.tensordot(inv_basis, K_ij, dims=1)


    # Now, we need to generate the out-of-plane direction, so we can
    # expand these Ks to the full Ks in 3D reciprocal space.
    
    # This is broken down into 2 steps to avoid floating point underflow
    # which was a real problem that showed up for electron ptycho
    b1_dir = basis[:,0] / t.linalg.norm(basis[:,0])
    b2_dir = basis[:,1] / t.linalg.norm(basis[:,1])
    perpendicular_dir = t.cross(b1_dir, b2_dir, dim=-1)
    # Note that we cannot use in-place operations if we want to be able to
    # use automatic differentiation successfully
    perpendicular_dir = perpendicular_dir / t.linalg.norm(perpendicular_dir)

    # We set the sign of the propagation direction appropriately
    if propagation_vector is not None:
        perpendicular_dir = perpendicular_dir \
            * t.sign(t.dot(perpendicular_dir,propagation_vector))
    else:
        perpendicular_dir = perpendicular_dir * \
            t.sign(t.dot(perpendicular_dir,offset_vector))
    
    # Then, if we have a propagation vector, we shift the in-plane
    # components of all the pixels to be centered around the in-plane
    # component of the propagation vector.
    if propagation_vector is not None:
        prop_dir = (propagation_vector /
                    t.linalg.norm(propagation_vector))
        K_0 = 2*np.pi / wavelength * (prop_dir - perpendicular_dir)
        K_0_ip = K_0 - t.dot(perpendicular_dir,K_0) * perpendicular_dir
        
        K_ip = K_ip + K_0_ip[:,None,None]
    else:
        K_0 = t.zeros_like(offset_vector)

    # Now, we have accurate in-plane values for K, so we can calculate the
    # out-of-plane part. We start by calculating it's squared magnitude
    K_oop_squared = (2*np.pi/wavelength)**2 - t.linalg.norm(K_ip,dim=0)**2
    
    # Then, we take the square root and assign it the appropriate direction,
    # adding to get the full 3D wavevectors. Note that we convert to complex
    # before the square root to appropriately map negative numbers to
    # complex frequencies

    # Below was the old, naive approach, that had numerical stability
    # issues when K_oop >> K_ip, and the subtle changes in K_oop were
    # clippping because K_ip was so much larger.
    
    # K = K_ip + perpendicular_dir[:,None,None] \
    #     * t.sqrt(t.complex(K_oop_squared,t.zeros_like(K_oop_squared)))
    
    # The correct algorithm for sqrt(1-x**2) - 1 is
    # - x**2 / (sqrt(1-x**2) + 1)
    # This one doesn't have numerical stability issues near x=0.
    # We still use the complex-valued trick in the square root, which
    # still generates the appropriate complex valued frequencies corresponding
    # to exponential decay if the in-plane Ks get larger than the
    # wavenumber of the light.

    # So, it seems like this is a silly way to calculate K_oop (why
    # not just take the square root of K_oop_squared?), but it is much
    # more numerically stable. Also note that this has one big difference from
    # the old method, which (by default) preserved the 

    K_oop_squared_complex = t.complex(K_oop_squared,
                                      t.zeros_like(K_oop_squared))

    K_oop = - t.linalg.norm(K_ip, dim=0)**2 / \
        (t.sqrt(K_oop_squared_complex) + (2*np.pi/wavelength))
    K = K_ip + perpendicular_dir[:,None,None] * K_oop

    # This corrects for the propagation direction, if set. If not, K_0 is
    # just 0
    K_m_K_0 = K - K_0[:,None,None]
    # print(propagation_vector)

    # 
    # In this section, we take the inner product of the calcualted
    # wavevectors with the offset vector, to get the phase shift
    # experienced by each plane wave.
    #
    
    # We need to convert to complex because K is complex
    offset_vector = t.complex(offset_vector,t.zeros_like(offset_vector))


    # We actually calculate the phase mask
    phase_mask = t.tensordot(offset_vector,K_m_K_0, dims=1)
    
    # And we take the conjugate if needed, which makes sure that the
    # imaginary part is always positive (supressing evanescent waves)
    if propagation_vector is not None \
       and t.sign(t.dot(propagation_vector,offset_vector.real)) == -1:
        phase_mask = t.conj(phase_mask)

    # We return the final mask, in the form asked for!
    return t.exp(1j*phase_mask).to(*args,**kwargs)
    

def near_field(wavefront, angular_spectrum_propagator):
    """ Propagates a wavefront via the angular spectrum method

    This function accepts an 3D torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield, and
    returns the near-field propagated version of it. It does this
    using the supplied angular spectrum propagator, which is a premade
    phase mask representing the Fourier transform of the kernel for
    light propagation in the desired geometry.


    Parameters
    ----------
    wavefront : torch.Tensor
        The (Leading Dims)xNxM stack of complex wavefronts to be propagated
    angular_spectrum_propagator : torch.Tensor
        The NxM phase mask to be applied during propagation

    Returns
    -------
    propagated : torch.Tensor
        The propagated wavefront 
    """
    return t.fft.ifft2(angular_spectrum_propagator * t.fft.fft2(wavefront))



def inverse_near_field(wavefront, angular_spectrum_propagator):
    """ Inverse propagates a wavefront via the angular spectrum method

    This function accepts an 3D torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield, and
    returns the near-field propagated version of it. It does this
    using the supplied angular spectrum propagator, which is a premade
    phase mask.

    It propagates the wave using the complex conjugate of the supplied
    phase mask. This corresponds to propagation backward across the original
    propagation region - however, the treatment of evanescent waves is such
    that evanescent waves will decay both during the forward propagation and
    inverse propagation. This is done for reasons of numerical stability,
    as the choice to magnify evanescent waves during the inverse propagation
    process will quickly lead to magnification of any small amount of noise at 
    frequencies larger than k_0, and in most typical situations will even
    lead to overflow of the floating point range. If evanescent waves need
    to be treated appropriately for any reason, it is recommended to use the
    "magnify_evanescent" option in the appropriate helper function used to
    generate the propagation phase mask. In this case, evanescent waves will
    be magnified both when used with the forward and inverse near field
    functions


    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts to be propagated
    angular_spectrum_propagator : torch.Tensor
        The NxM phase mask to be applied during propagation

    Returns
    -------
    propagated : torch.Tensor
        The inverse propagated wavefront
    """
    return t.fft.ifft2(t.fft.fft2(wavefront)
                       * t.conj(angular_spectrum_propagator))



# I think it would be worthwhile to implement an FFT-DI based strategy as
# well, especially for probe initialization where the propagation distance
# can be large relative to what the angular spectrum method can reliably handle


