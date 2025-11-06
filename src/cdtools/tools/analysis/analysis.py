"""Contains basic functions for analyzing the results of reconstructions

The functions in this module are designed to work either with pytorch tensors
or numpy arrays, so they can be used either directly after reconstructions
on the attributes of the models themselves, or after-the-fact once the
data has been stored in numpy arrays.
"""

import torch as t
import numpy as np
from cdtools.tools import image_processing as ip
import cdtools
from scipy import linalg as sla
from scipy import special
from scipy import optimize as opt
from scipy import spatial
import warnings

__all__ = [
    'product_svd',
    'orthogonalize_probes',
    'standardize',
    'synthesize_reconstructions',
    'calc_consistency_prtf',
    'calc_deconvolved_cross_correlation',
    'calc_frc',
    'calc_vn_entropy',
    'calc_mode_power_fractions',
    'calc_rms_error',
    'calc_fidelity',
    'calc_generalized_rms_error',
    'remove_phase_ramp',
    'remove_amplitude_exponent',
    'standardize_reconstruction_set',
    'standardize_reconstruction_pair',
    'calc_spectral_info',
]


def product_svd(A, B):
    """ Computes the SVD of A @ B

    This function uses a method which uses a QR decomposition of
    A and B to calculate the final reduced SVD, without explicitly
    calculating the full matrix. The output is defined such that
    A B = U S Vh, and as a reduced SVD

    Parameters
    ----------
    A : array
        An nxr matrix
    B : array
        An rxm matrix

    Returns
    -------
    U : array
        An nxr matrix of left singular vectors
    S : array
        An length-r array, t.diag(S) is the diagonal matrix of singular values
    Vh : array
        And rxm matrix of the conjugate-transposed right singular vectors
    """
    # Handle the case of numpy input
    return_np = False
    if isinstance(A, np.ndarray):
        A = t.as_tensor(A)
        return_np = True
    if isinstance(B, np.ndarray):
        B = t.as_tensor(B)
        return_np = True

    # We take a QR decomposition of the two matrices
    Qa, Ra = t.linalg.qr(A)
    Qb, Rb = t.linalg.qr(B.conj().transpose(0,1))

    # And now we take the SVD of the product of the two R matrices
    U, S, Vh = t.linalg.svd(t.matmul(Ra, Rb.conj().transpose(0,1)),
                            full_matrices=False)

    # And build back the final SVD of the product matrix
    U_final = t.matmul(Qa, U)
    Vh_final = t.matmul(Vh, Qb.conj().transpose(0,1))

    if return_np:
        U_final = U_final.numpy()
        S = S.numpy()
        Vh_final = Vh_final.numpy()
    
    return U_final, S, Vh_final


def orthogonalize_probes(
        probes,
        weight_matrix=None,
        n_probe_dims=2,
        return_reexpressed_weights=False,
):
    """ Orthogonalizes a set of incoherently mixing probes
    
    This function takes any set of probe modes for mixed-mode ptychography,
    which are considered to define a mutual coherence function, and returns
    an orthogonalized set probe modes which refer to the same mutual coherence
    function. The orthogonalized modes are extracted via a singular value
    decomposition and are unique up to a global per-mode phase factor.

    If a weight matrix is explicitly given, the function will instead
    orthogonalize the light field defined by weight_matrix @ probes. It
    accomplishes this via a method which avoids explicitly constructing this
    potentially large matrix. This can be useful for Orthogonal Probe
    Relaxation ptychography. In this case, one may have a large stacked
    matrix of shot-to-shot weights but a small basis set of probes.

    In addition to returning the orthogonalized probe modes, this function
    also returns a re-expression of the original weight matrix in the basis
    of the orthogonalized probe modes, such that:

    reexpressed_weight_matrix @ orthogonalized_probes = weight_matrix @ probes.

    This re-expressed weight matrix is guaranteed to have orthonormalized rows,
    such that:

    reexpressed_weight_matrix^\\dagger @ reexpressed_weight_matrix = I.

    There is usually no reason to use the re-expressed weight matrix.
    However, it can be useful in situations where the individual rows in the 
    weight matrix have a specific meaning, such as an exposure number, which
    should be preserved.
    
    Warning! The shape of the output orthogonalized_probes may not be equal to
    the shape of input probes, when a weight matrix is used. If the input
    weight matrix has m < l rows, where l is the number of probe modes,
    then the output orthogonalized probes will have length m, not length l.

    Parameters
    ----------
    probes : array
        An l x (<n_probe_pix>) array representing a stack of l probes
    weight_matrix : array
        Optional, an m x l weight matrix further elaborating on the state
    n_probe_dims : int
        Default is 2, the number of trailing dimensions for each probe state

    Returns
    -------
    orthogonalized_probes : array
        A min(m,l) x (<n_probe_dims>) array representing a stack of probes
    reexpressed_weight_matrix : array
        A the original weight matrix, re-expressed to work with the new probes 
    """

    return_np = False
    if isinstance(probes, np.ndarray):
        probes = t.as_tensor(probes)
        return_np = True
    if weight_matrix is not None and isinstance(weight_matrix, np.ndarray):
        weight_matrix = t.as_tensor(weight_matrix)
        return_np = True
    
    n_probe_pix = np.prod(np.array(probes.shape[-n_probe_dims:]))
    probes_mat = probes.reshape(probes.shape[:-n_probe_dims] +
                                (n_probe_pix,))
    
    if weight_matrix is None:
        # We just calculate a straight up SVD of the probes
        U, S, Vh = t.linalg.svd(probes_mat, full_matrices=False)
        
    else:
        U, S, Vh = product_svd(weight_matrix, probes_mat)

    output_shape = (-1,) + tuple(n for n in probes.shape[1:])
    orthogonalized_probes = (S[:,None] * Vh).reshape(output_shape)
    reexpressed_weight_matrix = U

    if return_np:
        orthogonalized_probes = orthogonalized_probes.numpy()
        reexpressed_weight_matrix = reexpressed_weight_matrix.numpy()

    if return_reexpressed_weights:
        return orthogonalized_probes, reexpressed_weight_matrix
    else:
        return orthogonalized_probes

        
def standardize(probe, obj, obj_slice=None, correct_ramp=False):
    """Standardizes a probe and object to prepare them for comparison

    There are a number of ambiguities in the definition of a ptychographic
    reconstruction. This function makes an explicit choice for each ambiguity
    to allow comparisons between independent reconstructions without confusing
    these ambiguities for real differences between the reconstructions.

    The ambiguities and standardizations are:

    1) a. Probe and object can be scaled inversely to one another
       b. So we set the probe intensity to an average per-pixel value of 1
    2) a. The probe and object can aquire equal and opposite phase ramps
       b. So we set the centroid of the FFT of the probe to zero frequency
    3) a. The probe and object can each acquire an arbitrary overall phase
       b. So we set the phase of the sum of all values of both the probe and object to 0

    When dealing with the properties of the object, a slice is used by
    default as the edges of the object often are dominated by unphysical
    noise. The default slice is from 3/8 to 5/8 of the way across. If the
    probe is actually a stack of incoherently mixing probes, then the
    dominant probe mode (assumed to be the first in the list) is used, but
    all the probes are updated with the same factors.

    Parameters
    ----------
    probe : array
        A complex array storing a retrieved probe or stack of incoherently mixed probes
    obj : array
        A complex array storing a retrieved object
    obj_slice : slice
        Optional, a slice to take from the object for calculating normalizations
    correct_ramp : bool
        Default False, whether to correct for the relative phase ramps

    Returns
    -------
    standardized_probe : array
        The standardized probe
    standardized_obj : array
        The standardized object

    """
    # First, we normalize the probe intensity to a fixed value.
    probe_np = False
    if isinstance(probe, np.ndarray):
        probe = t.as_tensor(probe, dtype=t.complex64)
        probe_np = True
    obj_np = False
    if isinstance(obj, np.ndarray):
        obj = t.as_tensor(obj,dtype=t.complex64)
        obj_np = True

    # If this is a single probe and not a stack of probes
    if len(probe.shape) == 2:
        probe = probe[None,...]
        single_probe = True
    else:
        single_probe = False

    normalization = t.sqrt(t.mean(t.abs(probe[0])**2))
    probe = probe / normalization
    obj = obj * normalization

    # Default slice of the object to use for alignment, etc.
    if obj_slice is None:
        obj_slice = np.s_[(obj.shape[0]//8)*3:(obj.shape[0]//8)*5,
                          (obj.shape[1]//8)*3:(obj.shape[1]//8)*5]


    if correct_ramp:
        # Need to check if this is actually working and, if not, why not
        center_freq = ip.centroid(t.abs(t.fft.fftshift(t.fft.fft2(probe[0]),
                                                       dim=(-1,-2)))**2)
        center_freq -= t.div(t.tensor(probe[0].shape,dtype=t.float32),2,rounding_mode='floor')
        center_freq /= t.as_tensor(probe[0].shape,dtype=t.float32)

        Is, Js = np.mgrid[:probe[0].shape[0],:probe[0].shape[1]]
        probe_phase_ramp = t.exp(2j * np.pi *
                                 (center_freq[0] * t.tensor(Is).to(t.float32) +
                                  center_freq[1] * t.tensor(Js).to(t.float32)))
        probe = probe *  t.conj(probe_phase_ramp)
        Is, Js = np.mgrid[:obj.shape[0],:obj.shape[1]]
        obj_phase_ramp = t.exp(2j*np.pi *
                               (center_freq[0] * t.tensor(Is).to(t.float32) +
                                center_freq[1] * t.tensor(Js).to(t.float32)))
        obj = obj * obj_phase_ramp

    # Then, we set them to consistent absolute phases

    obj_angle = t.angle(t.sum(obj[obj_slice]))
    obj = obj *  t.exp(-1j*obj_angle)

    for i in range(probe.shape[0]):
        probe_angle = t.angle(t.sum(probe[i]))
        probe[i] = probe[i] * t.exp(-1j*probe_angle)

    if single_probe:
        probe = probe[0]

    if probe_np:
        probe = probe.detach().cpu().numpy()
    if obj_np:
        obj = obj.detach().cpu().numpy()

    return probe, obj



def synthesize_reconstructions(probes, objects, use_probe=False, obj_slice=None, correct_ramp=False):
    """Takes a collection of reconstructions and outputs a single synthesized probe and object

    The function first standardizes the sets of probes and objects using the
    standardize function, passing through the relevant options. Then it
    calculates the closest overlap of subsequent frames to subpixel
    precision and uses a sinc interpolation to shift all the probes and objects
    to a common frame. Then the images are summed.

    Parameters
    ----------
    probes : list(array)
        A list of probes or stacks of probe modes
    objects : list(array)
        A list of objects
    use_probe : bool
        Default False, whether to use the probe or object for alignment
    obj_slice : slice
        Optional, A slice of the object to use for alignment and normalization
    correct_ramp : bool
        Default False, whether to correct for a relative phase ramp in the probe and object

    Returns
    -------
    synth_probe : array
        The synthesized probe
    synth_obj : array
        The synthesized object
    obj_stack : list(array)
        A list of standardized objects, for further processing
    """

    # This should be cleaned up so it accepts anything array_like
    probe_np = False
    if isinstance(probes[0], np.ndarray):
        probes = [t.as_tensor(probe,dtype=t.complex64) for probe in probes]
        probe_np = True
    obj_np = False
    if isinstance(objects[0], np.ndarray):
        objects = [t.as_tensor(obj,dtype=t.complex64) for obj in objects]
        obj_np = True

    obj_shape = np.min(np.array([obj.shape for obj in objects]),axis=0)
    objects = [obj[:obj_shape[0],:obj_shape[1]] for obj in objects]

    if obj_slice is None:
        obj_slice = np.s_[(objects[0].shape[0]//8)*3:(objects[0].shape[0]//8)*5,
                          (objects[0].shape[1]//8)*3:(objects[0].shape[1]//8)*5]



    synth_probe, synth_obj = standardize(probes[0].clone(), objects[0].clone(), obj_slice=obj_slice,correct_ramp=correct_ramp)

    obj_stack = [synth_obj]

    for i, (probe, obj) in enumerate(zip(probes[1:],objects[1:])):
        probe, obj = standardize(probe.clone(), obj.clone(), obj_slice=obj_slice,correct_ramp=correct_ramp)
        if use_probe:
            shift = ip.find_shift(synth_probe[0],probe[0], resolution=50)
        else:
            shift = ip.find_shift(synth_obj[obj_slice],obj[obj_slice], resolution=50)

        obj = ip.sinc_subpixel_shift(obj, shift)

        if len(probe.shape) == 3:
            probe = t.stack([ip.sinc_subpixel_shift(p,tuple(shift))
                             for p in probe],dim=0)
        else:
            probe = ip.sinc_subpixel_shift(probe,tuple(shift))

        synth_probe = synth_probe + probe
        synth_obj = synth_obj + obj
        obj_stack.append(obj)


    # If there only was one image
    try:
        i
    except:
        i = -1

    if probe_np:
        synth_probe = synth_probe.numpy()
    if obj_np:
        synth_obj = synth_obj.numpy()
        obj_stack = [obj.numpy() for obj in obj_stack]

    return synth_probe/(i+2), synth_obj/(i+2), obj_stack



def calc_consistency_prtf(synth_obj, objects, basis, obj_slice=None,nbins=None):
    """Calculates a PRTF between each the individual objects and a synthesized one

    The consistency PRTF at any given spatial frequency is defined as the ratio
    between the intensity of any given reconstruction and the intensity
    of a synthesized or averaged reconstruction at that spatial frequency.
    Typically, the PRTF is averaged over spatial frequencies with the same
    magnitude.

    Parameters
    ----------
    synth_obj : array
        The synthesized object in the numerator of the PRTF
    objects : list(array)
        A list of objects or diffraction patterns for the denomenator of the PRTF
    basis : array
        The basis for the reconstruction array to allow output in physical unit
    obj_slice : slice
        Optional, a slice of the objects to use for calculating the PRTF
    nbins : int
        Optional, number of bins to use in the histogram. Defaults to a sensible value

    Returns
    -------
    freqs : array
        The frequencies for the PRTF
    PRTF : array
        The values of the PRTF
    """

    obj_np = False
    if isinstance(objects[0], np.ndarray):
        objects = [t.as_tensor(obj, dtype=t.complex64) for obj in objects]
        obj_np = True
    if isinstance(synth_obj, np.ndarray):
        synth_obj = t.as_tensor(synth_obj, dtype=t.complex64)

    if isinstance(basis, t.Tensor):
        basis = basis.detach().cpu().numpy()

    if obj_slice is None:
        obj_slice = np.s_[(objects[0].shape[0]//8)*3:(objects[0].shape[0]//8)*5,
                          (objects[0].shape[1]//8)*3:(objects[0].shape[1]//8)*5]

    if nbins is None:
        nbins = np.max(synth_obj[obj_slice].shape) // 4

    synth_fft = (t.abs(t.fft.fftshift(t.fft.fft2(synth_obj[obj_slice]), dim=(-1,-2)))**2).numpy()


    di = np.linalg.norm(basis[:,0])
    dj = np.linalg.norm(basis[:,1])

    i_freqs = np.fft.fftshift(np.fft.fftfreq(synth_fft.shape[0],d=di))
    j_freqs = np.fft.fftshift(np.fft.fftfreq(synth_fft.shape[1],d=dj))

    Js,Is = np.meshgrid(j_freqs,i_freqs)
    Rs = np.sqrt(Is**2+Js**2)


    synth_ints, bins = np.histogram(Rs,bins=nbins,weights=synth_fft)

    prtfs = []
    for obj in objects:
        obj = obj[obj_slice]
        single_fft = (t.abs(t.fft.fftshift(t.fft.fft2(obj),
                                           dim=(-1,-2)))**2).numpy()
        single_ints, bins = np.histogram(Rs,bins=nbins,weights=single_fft)

        prtfs.append(synth_ints/single_ints)


    prtf = np.mean(prtfs,axis=0)

    if not obj_np:
        bins = t.Tensor(bins)
        prtf = t.Tensor(prtf)

    return bins[:-1], prtf



def calc_deconvolved_cross_correlation(im1, im2, im_slice=None):
    """Calculates a cross-correlation between two images with their autocorrelations deconvolved.

    This is formally defined as the inverse Fourier transform of the normalized
    product of the Fourier transforms of the two images. It results in a
    kernel, whose characteristic size is related to the exactness of the
    possible alignment between the two images, on top of a random background

    Parameters
    ----------
    im1 : array
        The first image, as a complex or real valued array
    im2 : array
        The first image, as a complex or real valued array
    im_slice : slice
        Default is from 3/8 to 5/8 across the image, a slice to use in the processing.

    Returns
    -------
    corr : array
        The complex-valued deconvolved cross-correlation, in real space

    """

    im_np = False
    if isinstance(im1, np.ndarray):
        im1 = t.as_tensor(im1)
        im_np = True
    if isinstance(im2, np.ndarray):
        im2 = t.as_tensor(im2)
        im_np = True

    if im_slice is None:
        im_slice = np.s_[(im1.shape[0]//8)*3:(im1.shape[0]//8)*5,
                          (im1.shape[1]//8)*3:(im1.shape[1]//8)*5]


    cor_fft = t.fft.fft2(im1[im_slice]) * \
        t.conj(t.fft.fft2(im2[im_slice]))

    # Not sure if this is more or less stable than just the correlation
    # maximum - requires some testing
    cor = t.fft.ifft2(cor_fft / t.abs(cor_fft))

    if im_np:
        cor = cor.numpy()

    return cor


def calc_frc(im1, im2, basis, im_slice=None, nbins=None, snr=1., limit='side'):
    """Calculates a Fourier ring correlation between two images

    This function requires an input of a basis to allow for FRC calculations
    to be related to physical units.

    Like other analysis functions, this can take input in numpy or pytorch,
    and will return output in the respective format.

    Parameters
    ----------
    im1 : array
        The first image, a complex or real valued array
    im2 : array
        The first image, a complex or real valued array
    basis : array
        The basis for the images, defined as is standard for datasets
    im_slice : slice
        Default is the full image
    nbins : int
        Number of bins to break the FRC up into
    snr : float
        The signal to noise ratio (for the combined information in both images) to return a threshold curve for.
    limit : str
        Default is 'side'. What is the highest frequency to calculate the FRC to? If 'side', it chooses the side of the Fourier transform, if 'corner' it goes fully to the corner.

    Returns
    -------
    freqs : array
        The frequencies associated with each FRC value
    FRC : array
        The FRC values
    threshold : array
        The threshold curve for comparison

    """

    im_np = False
    if isinstance(im1, np.ndarray):
        im1 = t.as_tensor(im1)
        im_np = True
    if isinstance(im2, np.ndarray):
        im2 = t.as_tensor(im2)
        im_np = True

    if isinstance(basis, np.ndarray):
        basis = t.tensor(basis)

    if im_slice is None:
        im_slice = np.s_[:,:]

    if nbins is None:
        nbins = np.max(im1[im_slice].shape) // 8

    f1 = t.fft.fftshift(t.fft.fft2(im1[im_slice]),dim=(-1,-2))
    f2 = t.fft.fftshift(t.fft.fft2(im2[im_slice]),dim=(-1,-2))
    cor_fft = f1 * t.conj(f2)
    #from cdtools.tools import plotting as p
    #from matplotlib import pyplot as plt
    #p.plot_phase(im1[im_slice], cmap='cividis')
    #p.plot_phase(im2[im_slice], cmap='cividis')    
    #p.plot_amplitude(t.log(t.abs(cor_fft)))
    #p.plot_amplitude(t.log(t.abs(f1)))
    #p.plot_phase(cor_fft)
    #plt.show()
    F1 = t.abs(f1)**2
    F2 = t.abs(f2)**2

    # TODO this is still incorrect if the two bases arent equal
    di = np.linalg.norm(basis[:,0])
    dj = np.linalg.norm(basis[:,1])

    i_freqs = np.fft.fftshift(np.fft.fftfreq(cor_fft.shape[0],d=di))
    j_freqs = np.fft.fftshift(np.fft.fftfreq(cor_fft.shape[1],d=dj))

    Js,Is = np.meshgrid(j_freqs,i_freqs)
    Rs = np.sqrt(Is**2+Js**2)

    if limit.lower().strip() == 'side':
        max_i = np.max(i_freqs)
        max_j = np.max(j_freqs)
        frc_range = [0, max(max_i,max_j)]

    elif limit.lower().strip() == 'corner':
        frc_range = [0, np.max(Rs)]
    else:
        raise ValueError('Invalid FRC limit: choose "side" or "corner"')

    numerator, bins = np.histogram(Rs, bins=nbins, range=frc_range,
                                   weights=cor_fft.numpy())
    denominator_F1, bins = np.histogram(Rs, bins=nbins, range=frc_range,
                                        weights=F1.detach().cpu().numpy())
    denominator_F2, bins = np.histogram(Rs, bins=nbins, range=frc_range,
                                        weights=F2.detach().cpu().numpy())
    n_pix, bins = np.histogram(Rs, bins=nbins, range=frc_range)

    
    #n_pix = n_pix / 4 # This is for an apodized image, apodized with a hann window
    
    frc = numerator / np.sqrt(denominator_F1*denominator_F2)
    #plt.figure()
    #plt.plot(np.real(frc))
    #plt.title('real')
    #plt.figure()
    #plt.plot(np.imag(frc))
    #plt.title('imag')
    #plt.show()
    # This moves from combined-image SNR to single-image SNR
    snr /= 2

    # NOTE: I should update this  to produce lots of different threshold curves
    # sigma, 2sigma, 3sigma, traditional FRC 1-bit, my better one, n_pix, etc.
    
    threshold = (snr + (2 * np.sqrt(snr) + 1) / np.sqrt(n_pix)) / \
        (1 + snr + (2 * np.sqrt(snr)) / np.sqrt(n_pix))

    my_threshold = np.sqrt(snr**2 + (2*snr**2 + 2*snr + 1)/n_pix) / \
        np.sqrt(snr**2 + 2 * snr + 1 + 2*snr**2 / n_pix)

    twosigma_threshold = 2/ np.sqrt(n_pix)
    
    if not im_np:
        bins = t.tensor(bins)
        frc = t.tensor(frc)
        threshold = t.tensor(threshold)

    return bins[:-1], frc, threshold


def calc_vn_entropy(matrix):
    """Calculates the Von Neumann entropy of a density matrix
    
    Will either accept a single matrix, or a stack of matrices. Matrices
    are assumed to be Hermetian and positive definite, to be well-formed
    density matrices
    
    Parameters
    ----------
    matrix : np.array
        The nxn matrix or lxnxn stack of matrices to calculate the entropy of

    
    Returns
    -------
    entropy: float or np.array
        The entropy or entropies of the arrays
    """

    if len(matrix.shape) == 3:
        # Get the eigenvalues
        eigs = [np.linalg.eigh(mat)[0] for mat in matrix]
        # Normalize them to match standard density matrix form
        eigs = [eig / np.sum(eig) for eig in eigs]
        # And calculate the VN entropy!
        entropies = [-np.sum(special.xlogy(eig,eig)) for eig in eigs]
        return np.array(entropies)
    else:
        eig = np.linalg.eigh(matrix)[0]
        entropy = -np.sum(special.xlogy(eig,eig))/np.sum(eig)
        return entropy


def calc_mode_power_fractions(
        probes,
        weight_matrix=None,
        n_probe_dims=2,
        assume_preorthogonalized=False,
):
    """Calculates the fraction of total power in each orthogonalized mode

    This code first orthogonalizes the probe modes, so the result of this
    function are independent of the particular way that the multi-mode
    breakdown is expressed.


    Parameters
    ----------
    probes : array
        An l x (<n_probe_pix>) array representing a stack of l probes
    weight_matrix : array
        Optional, an m x l weight matrix further elaborating on the state
    n_probe_dims : int
        Default is 2, the number of trailing dimensions for each probe state
    assume_preorthogonalized : bool
        Default is False. If True, will not orthogonalize the probes
    
    Returns
    -------
    power_fractions : array
        The fraction of the total power in each mode
    """

    if not assume_preorthogonalized:
        ortho_probes = orthogonalize_probes(
            probes,
            weight_matrix=weight_matrix,
            n_probe_dims=n_probe_dims,
            return_reexpressed_weights=False
        )
    else:
        weight_slice = np.s_[...,] + np.s_[None,] * n_probe_dims
        if weight_matrix is None:
            ortho_probes = probes
        else:
            ortho_probes = t.sum(weight_matrix[weight_slice] * probes,
                                 axis=-(n_probe_dims + 1))
        
    dims = [-d-1 for d in range(n_probe_dims)]
    power = t.sum(t.abs(ortho_probes)**2, dim=dims)
    power_fractions = power / t.sum(power)
    return power_fractions


def calc_rms_error(field_1, field_2, align_phases=True, normalize=False,
                   dims=2):
    """Calculates the root-mean-squared error between two complex wavefields

    The formal definition of this function is:

    output = norm * sqrt(mean(abs(field_1 - gamma * field_2)**2))
    
    Where norm is an optional normalization factor, and gamma is an
    optional phase factor which is appropriate when the wavefields suffer
    from a global phase degeneracy as is often the case in diffractive
    imaging.

    The normalization is defined as the square root of the total intensity
    contained in field_1, which is appropriate when field_1 represents a
    known ground truth:

    norm = sqrt(mean(abs(field_1)**2))
    
    The phase offset is an analytic expression for the phase offset which
    will minimize the RMS error between the two wavefields:

    gamma = exp(1j * angle(sum(field_1 * conj(field_2))))
    
    This implementation is stable even in cases where field_1 and field_2
    are completely orthogonal.

    In the definitions above, the field_n are n-dimensional wavefields. The
    dimensionality of the wavefields can be altered via the dims argument,
    but the default is 2 for a 2D wavefield.
    
    Parameters
    ----------
    field_1 : array
        The first complex-valued field
    field_2 : array
        The second complex-valued field
    align_phases : bool
        Default is True, whether to account for a global phase offset
    normalize : bool
        Default is False, whether to normalize to the intensity of field_1
    dims : (int or tuple of python:ints)
        Default is 2, the number of final dimensions to reduce over.

    
    Returns
    -------
    rms_error : float or t.Tensor
        The RMS error, or tensor of RMS errors, depending on the dim argument

    """
    field_1 = t.as_tensor(field_1)
    field_2 = t.as_tensor(field_2)

    sumdims = tuple(d - dims for d in range(dims))
        
    if align_phases:
        # Keepdim allows us to broadcast the result correctly when we
        # multiply by the fields
        gamma = t.exp(1j * t.angle(t.sum(field_1 * t.conj(field_2), dim=sumdims,
                                         keepdim=True)))
    else:
        gamma = 1

    if normalize:
        norm = 1 / t.mean(t.abs(field_1)**2, dim=sumdims)
    else:
        norm = 1

    difference = field_1 - gamma * field_2
    
    return t.sqrt(norm * t.mean(t.abs(difference)**2, dim=sumdims))


def calc_fidelity(fields_1, fields_2, dims=2):
    """Calculates the fidelity between two density matrices

    The fidelity is a comparison metric between two density matrices
    (i.e. mutual coherence functions) that extends the idea of the
    overlap to incoherent light. As a reminder, the overlap between two
    fields is:

    overlap = abs(sum(field_1 * field_2))**2
    
    Whereas the fidelity is defined as:
    
    fidelity = trace(sqrt(sqrt(dm_1) <dot> dm_2 <dot> sqrt(dm_1)))**2

    where dm_n refers to the density matrix encoded by fields_n such
    that dm_n = fields_n <dot> fields_<n>.conjtranspose(), sqrt
    refers to the matrix square root, and <dot> is the matrix product.
    
    This is not a practical implementation, however, as it is not feasible
    to explicitly construct the matrices dm_1 and dm_2 in memory. Therefore,
    we take advantage of the alternate definition based directly on the
    fields_<n> parameter:

    fidelity = sum(svdvals(fields_1 <dot> fields_2.conjtranspose()))**2
    
    In the definitions above, the fields_n are regarded as collections of
    wavefields, where each wavefield is by default 2-dimensional. The
    dimensionality of the wavefields can be altered via the dims argument,
    but the fields_n arguments must always have at least one more dimension
    than the dims argument. Any additional dimensions are treated as batch
    dimensions.
    
    Parameters
    ----------
    fields_1 : array
        The first set of complex-valued field modes
    fields_2 : array
        The second set of complex-valued field modes
    dims : int
        Default is 2, the number of final dimensions to reduce over.

    
    Returns
    -------
    fidelity : float or t.Tensor
        The fidelity, or tensor of fidelities, depending on the dim argument

    """

    fields_1 = t.as_tensor(fields_1)
    fields_2 = t.as_tensor(fields_2)
    mult = fields_1.unsqueeze(-dims-2) * fields_2.unsqueeze(-dims-1).conj()
    sumdims = tuple(d - dims for d in range(dims))
    mat = t.sum(mult,dim=sumdims)
    # Because I think this is the nuclear norm squared, I would like to swap
    # Out the definition for this, but I need to test it before swapping.
    # It also probably makes sense to implement sqrt_fidelity separately
    # because that's more important
    #return t.linalg.matrix_norm(mat, ord='nuc')**2
    
    # I think this is just the nuclear norm.
    svdvals = t.linalg.svdvals(mat)
    return t.sum(svdvals, dim=-1)**2


def calc_generalized_rms_error(fields_1, fields_2, normalize=False, dims=2):
    """Calculates a generalization of the root-mean-squared error between two complex wavefields

    This function calculates an generalization of the RMS error which uses the
    concept of fidelity to extend it to capture the error between
    incoherent wavefields, defined as a mode decomposition. The extension has
    several nice properties, in particular:

    1) For coherent wavefields, it precisely matches the RMS error including
       a correction for the global phase degeneracy (align_phases=True)
    2) All mode decompositions of either field that correspond to the same
       density matrix / mutual coherence function will produce the same 
       output
    3) The error will only be zero when comparing mode decompositions that
       correspond to the same density matrix.
    4) Due to (2), one need not worry about the ordering of the modes,
       properly orthogonalizing the modes, and it is even possible to
       compare mode decompositions with different numbers of modes.    
    
    The formal definition of this function is:

    output = norm * sqrt(mean(abs(fields_1)**2)
                         + mean(abs(fields_2)**2)
                         - 2 * sqrt(fidelity(fields_1,fields_2)))
    
    Where norm is an optional normalization factor, and the fidelity is
    defined based on the mean, rather than the sum, to match the convention
    for the root *mean* squared error.

    The normalization is defined as the square root of the total intensity
    contained in fields_1, which is appropriate when fields_1 represents a
    known ground truth:

    norm = sqrt(mean(abs(fields_1)**2))

    In the definitions above, the fields_n are regarded as collections of
    wavefields, where each wavefield is by default 2-dimensional. The
    dimensionality of the wavefields can be altered via the dims argument,
    but the fields_n arguments must always have at least one more dimension
    than the dims argument. Any additional dimensions are treated as batch
    dimensions.
    
    Parameters
    ----------
    fields_1 : array
        The first set of complex-valued field modes
    fields_2 : array
        The second set of complex-valued field modes
    normalize : bool
        Default is False, whether to normalize to the intensity of fields_1
    dims : (int or tuple of python:ints)
        Default is 2, the number of final dimensions to reduce over.

    Returns
    -------
    rms_error : float or t.Tensor
        The generalized RMS error, or tensor of generalized RMS errors, depending on the dim argument

    """
    # TODO either make this work correctly or just make everything work
    # only for tensors
    fields_1 = t.as_tensor(fields_1)
    fields_2 = t.as_tensor(fields_2)

    npix = t.prod(t.as_tensor(fields_1.shape[-dims:],dtype=t.int32))
    
    sumdims = tuple(d - dims - 1 for d in range(dims+1))
    fields_1_intensity = t.sum(t.abs(fields_1)**2,dim=sumdims) / npix
    fields_2_intensity = t.sum(t.abs(fields_2)**2,dim=sumdims) / npix
    fidelity = calc_fidelity(fields_1, fields_2, dims=dims) / npix**2

    result = fields_1_intensity + fields_2_intensity - 2 * t.sqrt(fidelity)
    
    if normalize:
        result /= fields_1_intensity

    return t.sqrt(result)
    

def calc_generalized_frc(fields_1, fields_2, basis, im_slice=None, nbins=None, snr=1., limit='side'):
    """Calculates a Fourier ring correlation between two images

    This function requires an input of a basis to allow for FRC calculations
    to be related to physical units.

    Like other analysis functions, this can take input in numpy or pytorch,
    and will return output in the respective format.

    Parameters
    ----------
    im1 : array
        The first image, a complex or real valued array
    im2 : array
        The first image, a complex or real valued array
    basis : array
        The basis for the images, defined as is standard for datasets
    im_slice : slice
        Default is from 3/8 to 5/8 across the image, a slice to use in the processing.
    nbins : int
        Number of bins to break the FRC up into
    snr : float
        The signal to noise ratio (for the combined information in both images) to return a threshold curve for.

    Returns
    -------
    freqs : array
        The frequencies associated with each FRC value
    FRC : array
        The FRC values
    threshold : array
        The threshold curve for comparison

    """

    im_np = False
    if isinstance(fields_1, np.ndarray):
        fields_1 = t.as_tensor(fields_1)
        im_np = True
    if isinstance(fields_2, np.ndarray):
        fields_2 = t.as_tensor(fields_2)
        im_np = True

    if isinstance(basis, np.ndarray):
        basis = t.tensor(basis)

    if im_slice is None:
        im_slice = np.s_[...,:,:]

    if nbins is None:
        nbins = np.max(fields_1[...,im_slice].shape[-2:]) // 4


    f1 = t.fft.fftshift(t.fft.fft2(fields_1[im_slice]),dim=(-1,-2))
    f2 = t.fft.fftshift(t.fft.fft2(fields_2[im_slice]),dim=(-1,-2))
    cor_fft = f1 * t.conj(f2)

    F1 = t.abs(f1)**2
    F2 = t.abs(f2)**2


    di = np.linalg.norm(basis[:,0])
    dj = np.linalg.norm(basis[:,1])

    i_freqs = np.fft.fftshift(np.fft.fftfreq(cor_fft.shape[-2],d=di))
    j_freqs = np.fft.fftshift(np.fft.fftfreq(cor_fft.shape[-1],d=dj))

    Js,Is = np.meshgrid(j_freqs,i_freqs)
    Rs = np.sqrt(Is**2+Js**2)

    if limit.lower().strip() == 'side':
        max_i = np.max(i_freqs)
        max_j = np.max(j_freqs)
        frc_range = [0, max(max_i,max_j)]

    elif limit.lower().strip() == 'corner':
        frc_range = [0, np.max(Rs)]
    else:
        raise ValueError('Invalid FRC limit: choose "side" or "corner"')

    # This line is used to get a set of bins that matches the logic
    # used by np.histogram, so that this function will match the choices
    # of bin edges that comes from the non-generalized version. This also
    # gets us the count on the number of pixels per bin so we can calculate
    # the threshold curve
    n_pix, bins = np.histogram(Rs,bins=nbins, range=frc_range)
    bins = t.as_tensor(bins)
    Rs = t.as_tensor(Rs)

    frc = []
    for i in range(len(bins)-1):
        mask = t.logical_and(Rs<bins[i+1], Rs>=bins[i])
        masked_f1 = f1 * mask[...,:,:]
        masked_f2 = f2 * mask[...,:,:]
        numerator = t.sqrt(calc_fidelity(masked_f1, masked_f2))
        denominator_f1 = t.sqrt(calc_fidelity(masked_f1, masked_f1))
        denominator_f2 = t.sqrt(calc_fidelity(masked_f2, masked_f2))
        frc.append(numerator / t.sqrt((denominator_f1 * denominator_f2)))

    frc = np.array(frc)

    # This moves from combined-image SNR to single-image SNR
    snr /= 2

    threshold = (snr + (2 * snr + 1) / np.sqrt(n_pix)) / \
        (1 + snr + (2 * np.sqrt(snr)) / np.sqrt(n_pix))

    if not im_np:
        bins = t.tensor(bins)
        frc = t.tensor(frc)
        threshold = t.tensor(threshold)

    return bins[:-1], frc, threshold


def remove_phase_ramp(im, window, probe=None):
    
    window = im[window]

    Is, Js = np.mgrid[:window.shape[0],:window.shape[1]]
    def zero_freq_component(freq):
        phase_ramp = np.exp(2j * np.pi * (freq[0] * Is + freq[1] * Js))
        return -np.abs(np.sum(phase_ramp * window))**2
    
    x0 = np.array([0,0])
    result = opt.minimize(zero_freq_component, x0)
    center_freq = result['x']

    Is, Js = np.mgrid[:im.shape[0],:im.shape[1]]
    phase_ramp = np.exp(2j * np.pi * (center_freq[0] * Is + center_freq[1] * Js))
    im = im * phase_ramp
    
    if probe is not None:
        Is, Js = np.mgrid[:probe.shape[-2],:probe.shape[-1]]
        phase_ramp = np.exp(-2j * np.pi * (center_freq[0] * Is + center_freq[1] * Js))
        probe = probe * phase_ramp
        return im, probe
    else:
        return im


def remove_amplitude_exponent(im, window, probe=None, weights=None, translations=None, basis=None):
    window = np.abs(im[window])
    
    Is, Js = np.mgrid[:window.shape[0],:window.shape[1]]
    def rms_error(x):
        constant = x[0]
        growth_rate = x[1:]
        exponential_decay = constant * np.exp((growth_rate[0] * Is + growth_rate[1] * Js))
        return np.sum((window - exponential_decay)**2)
    
    x0 = np.array([1,0,0])
    result = opt.minimize(rms_error, x0, method='Nelder-Mead')
    growth_rate = result['x'][1:]

    Is, Js = np.mgrid[:im.shape[0],:im.shape[1]]
    exponential_decay = np.exp(-(growth_rate[0] * Is + growth_rate[1] * Js))
    im = im * exponential_decay
    to_return = (im,)
    
    if probe is not None:
        Is, Js = np.mgrid[:probe.shape[-2],:probe.shape[-1]]
        exponential_decay = np.exp((growth_rate[0] * Is + growth_rate[1] * Js))
        probe = probe * exponential_decay
        to_return = to_return + (probe,)
        
    if weights is not None:
        pix_translations = cdtools.tools.interactions.translations_to_pixel(t.as_tensor(basis), t.as_tensor(translations)).numpy()
        pix_translations -= np.min(pix_translations,axis=0)
        weights = weights * np.exp(growth_rate[0] * pix_translations[:,0] + growth_rate[1] * pix_translations[:,1])
        to_return = to_return + (weights,)
    
    if len(to_return) == 1:
        return to_return[0]
    else:
        return to_return

def make_illumination_map(results, total_probe_intensity=None, padding=200):
    
    probe_intensity = np.sum(np.abs(results['probe'])**2, axis=0)
    if total_probe_intensity is not None:
        probe_intensity = probe_intensity / np.sum(probe_intensity) * total_probe_intensity
    
    illumination_map = np.zeros_like(results['obj'], dtype=np.float32)
    translations = t.as_tensor(results['translations'])
    pix_translations = cdtools.tools.interactions.translations_to_pixel(
        t.as_tensor(results['basis']), translations)
    
    # We use the min_translation stored in the model, which was calculated
    # from the uncorrected translations and therefore may be different from
    # what it is if we recalculate it on the corrected translations
    min_translation = t.as_tensor(results['state_dict']['min_translation'])
    pix_translations = np.round((pix_translations - min_translation).numpy()).astype(int)

    for translation in pix_translations:
        illumination_map[translation[0]:translation[0]+probe_intensity.shape[0],
                         translation[1]:translation[1]+probe_intensity.shape[1]] += probe_intensity
    
    return illumination_map


def standardize_reconstruction_set(
        half_1,
        half_2,
        full,
        correct_phase_offset=True,
        correct_phase_ramp=True,
        correct_amplitude_exponent=False,
        window=np.s_[:,:],
        nbins=50,
        frc_limit='side',
):
    """Standardizes and analyses a set of 50/50/100% reconstructions
    
    It's very common to split a ptychography dataset into two sub-datasets,
    each with 50% of the exposures, so that the difference between the two
    sub-datasets can be used to estimate the quality and resolution of the
    final, full reconstruction. But to do that analysis, first the
    reconstructions need to be aligned with respect to each other and
    normalized in a few ways.

    This function takes the results (as output by model.save_results) of
    a set of 50/50/100% reconstructions and:

    - Aligns the object reconstructions with one another
    - Corrects for the global phase offset (by default)
    - Sets a sensible value for the object/probe phase ramp (by default)
    - Sets a sensible value for the object/probe exponential decay (off by
        default)
    - Calculates the FRC and derived SSNR.

    Then, these results are packaged into an output dictionary. The output
    does not retain all the information from the inputs, so if full traceability
    is desired, do not delete the files containing the individual
    reconstructions

    Parameters
    ----------
    half_1 : dict
        The result of the first half dataset, as returned by model.save_results
    half_2 : dict
        The result of the second half dataset, as returned by model.save_results
    full : dict
        The result of the full dataset, as returned by model.save_results

    Returns
    -------
    results : dict
        A dictionary containing the synthesized results
    """
   # We get the two half-data reconstructions
    obj_1, probe_1, weights_1 = half_1['obj'],half_1['probe'],half_1['weights']
    obj_2, probe_2, weights_2 = half_2['obj'],half_2['probe'],half_2['weights']
    obj, probe, weights = full['obj'], full['probe'], full['weights']


    if correct_phase_ramp:
        obj_1, probe_1 = remove_phase_ramp(
            half_1['obj'], window, probe=half_1['probe'])
        obj_2, probe_2 = remove_phase_ramp(
            half_2['obj'], window, probe=half_2['probe'])
        obj, probe = remove_phase_ramp(
            full['obj'], window, probe=full['probe'])

    if correct_amplitude_exponent:
        obj_1, probe_1, weights_1 = remove_amplitude_exponent(
            obj_1, window, probe=probe_1,
            weights=half_1['weights'],
            basis=half_1['basis'],
            translations=half_1['translations'])
        obj_2, probe_2, weights_2 = remove_amplitude_exponent(
            obj_2, window, probe=probe_2,
            weights=half_2['weights'],
            basis=half_2['basis'],
            translations=half_2['translations'])
        obj, probe, weights = remove_amplitude_exponent(
            obj, window, probe=probe,
            weights=full['weights'],
            basis=full['basis'],
            translations=full['translations'])

    
    if correct_phase_offset:
        obj_1 = np.exp(-1j* np.angle(np.sum(obj_1[window]))) * obj_1
        obj_2 = np.exp(-1j* np.angle(np.sum(obj_2[window]))) * obj_2
        obj = np.exp(-1j* np.angle(np.sum(obj[window]))) * obj


    # Todo update the translations to account for the determined shift
    shift_1 = ip.find_shift(
        t.as_tensor(ip.hann_window(np.abs(obj[window]))),
        t.as_tensor(ip.hann_window(np.abs(obj_1[window]))))
    obj_1  = ip.sinc_subpixel_shift(
        t.as_tensor(obj_1), shift_1).numpy()

    shift_2 = ip.find_shift(
        t.as_tensor(ip.hann_window(np.abs(obj[window]))),
        t.as_tensor(ip.hann_window(np.abs(obj_2[window]))))
    obj_2  = ip.sinc_subpixel_shift(
        t.as_tensor(obj_2), shift_2).numpy()

    freqs, frc, threshold = calc_frc(
        ip.hann_window(obj_1[window]),
        ip.hann_window(obj_2[window]),
        full['obj_basis'], nbins=nbins, limit=frc_limit)

    
    
    # The correct formulation when the final output is the full reconstruction
    ssnr = 2 * np.abs(frc) / (1 - np.abs(frc))

    results = {
        'obj_half_1': obj_1,
        'probe_half_1': probe_1,
        'weights_half_1': weights_1,
        'translations_half_1': half_1['translations'],
        'background_1': half_1['background'],
        'obj_half_2': obj_2,
        'probe_half_2': probe_2,
        'weights_half_2': weights_2,
        'translations_half_2': half_2['translations'],
        'background_2': half_2['background'],
        'obj_full': obj,
        'probe_full': probe,
        'weights_full': weights,
        'translations_full': full['translations'],
        'background_full': full['background'],
        'wavelength': full['wavelength'],
        'obj_basis': full['obj_basis'],
        'probe_basis': full['probe_basis'],
        'frc_freqs': freqs,
        'frc': frc,
        'frc_threshold': threshold,
        'ssnr': ssnr}
    
    return results


def standardize_reconstruction_pair(
        half_1,
        half_2,
        correct_phase_offset=True,
        correct_phase_ramp=True,
        correct_amplitude_exponent=False,
        window=np.s_[:,:],
        nbins=50,
        probe_nbins=50,
        frc_limit='side',
):
    """Standardizes and analyses a set of two repeat
    
    It's very common to run two subsequent ptycho reconstructions, so that the
    effect of sample damage during the first reconstruction can be used in the
    estimate of thefinal quality. The difference between the two datasets
    datasets can be used to estimate the quality and resolution of each one.
    But to do that analysis, first the reconstructions need to be aligned
    with respect to each other and normalized in a few ways.

    This function takes the results (as output by model.save_results) of
    a pair of reconstructions and:

    - Aligns the object reconstructions with one another
    - Corrects for the global phase offset (by default)
    - Sets a sensible value for the object/probe phase ramp (by default)
    - Sets a sensible value for the object/probe exponential decay (off by
        default)
    - Calculates the FRC and derived SSNR.

    Then, these results are packaged into an output dictionary. The output
    does not retain all the information from the inputs, so if full traceability
    is desired, do not delete the files containing the individual
    reconstructions

    Parameters
    ----------
    half_1 : dict
        The result of the first half dataset, as returned by model.save_results
    half_2 : dict
        The result of the second half dataset, as returned by model.save_results

    Returns
    -------
    results : dict
        A dictionary containing the synthesized results
    """
   # We get the two half-data reconstructions
    obj_1, probe_1, weights_1 = half_1['obj'],half_1['probe'],half_1['weights']
    obj_2, probe_2, weights_2 = half_2['obj'],half_2['probe'],half_2['weights']


    if correct_phase_ramp:
        obj_1, probe_1 = remove_phase_ramp(
            half_1['obj'], window, probe=half_1['probe'])
        obj_2, probe_2 = remove_phase_ramp(
            half_2['obj'], window, probe=half_2['probe'])

    # TODO weights are not included
    if correct_amplitude_exponent:
        obj_1, probe_1 = remove_amplitude_exponent(
            obj_1, window, probe=probe_1,
            basis=half_1['obj_basis'],
            translations=half_1['translations'])
        obj_2, probe_2 = remove_amplitude_exponent(
            obj_2, window, probe=probe_2,
            basis=half_2['obj_basis'],
            translations=half_2['translations'])

    
    if correct_phase_offset:
        obj_1 = np.exp(-1j* np.angle(np.sum(obj_1[window]))) * obj_1
        obj_2 = np.exp(-1j* np.angle(np.sum(obj_2[window]))) * obj_2

    # Todo update the translations to account for the determined shift
    shift = ip.find_shift(
        t.as_tensor(ip.hann_window(obj_1[window])),
        t.as_tensor(ip.hann_window(obj_2[window])))
    obj_2  = ip.sinc_subpixel_shift(
        t.as_tensor(obj_2), shift).numpy()

    probe_shift = ip.find_shift(
        t.as_tensor(probe_1[0]),
        t.as_tensor(probe_2[0]),
    )
    
    for idx in range(probe_2.shape[0]):
        probe_2[idx] = ip.sinc_subpixel_shift(
            t.as_tensor(probe_2[idx]), probe_shift).numpy()

    # TODO I'm not sure if the default threshold treats this case right tbh
    freqs, frc, threshold = calc_frc(
        ip.hann_window(obj_1[window]),
        ip.hann_window(obj_2[window]),
        half_1['obj_basis'],
        nbins=nbins,
        limit=frc_limit,
    )

    probe_freqs, probe_frc, probe_frc_threshold = calc_generalized_frc(
        probe_1,
        probe_2,
        half_1['probe_basis'],
        nbins=probe_nbins,
        limit=frc_limit,
    )


    probe_1_intensity = np.sum(np.abs(probe_1)**2)
    probe_2_intensity = np.sum(np.abs(probe_2)**2)

    probe_nmse = 1 - (calc_fidelity(probe_1, probe_2)
                      / (probe_1_intensity * probe_2_intensity))
    
    probe_nrms_error = calc_generalized_rms_error(
        probe_1[0:],
        probe_2[0:],
        normalize=True
    )
    
    # The correct formulation when the final output is one of the two
    # reconstructions
    ssnr = np.abs(frc) / (1 - np.abs(frc))

    results = {
        'obj_1': obj_1,
        'probe_1': probe_1,
        'weights_1': weights_1,
        'translations_1': half_1['translations'],
        'background_1': half_1['background'],
        'obj_2': obj_2,
        'probe_2': probe_2,
        'weights_2': weights_2,
        'translations_2': half_2['translations'],
        'background_2': half_2['background'],
        'wavelength': half_1['wavelength'],
        'obj_basis': half_1['obj_basis'],
        'probe_basis': half_1['probe_basis'],
        'frc_freqs': freqs,
        'frc': frc,
        'frc_threshold': threshold,
        'ssnr': ssnr,
        'probe_freqs': probe_freqs,
        'probe_frc': probe_frc,
        'probe_frc_threshold': probe_frc_threshold,
        'probe_nrms_error': probe_nrms_error,
        'probe_nmse': probe_nmse,
    }
    
    return results


def calc_spectral_info(dataset, nbins=50):
    """Makes a properly normalized sum diffraction pattern

    This returns a scaled version of sum of all the diffraction patterns
    within the dataset. The scaling is defined so that the total intensity
    in the final image is equal to the intensity arising from a region of
    the scan pattern whose area matches one detector conjugate field of
    view.

    This estimation will start to deviate from the truth if the scan area
    is not significantly larger than the illumination function, because
    the nonzero size of the illumination function is not taken into account.
    Furthermore, in the edge case where all the scan points are colinear,
    the estimate will fail, and the mean diffraction pattern will be returned
    instead

    Parameters
    ----------
    dataset : Ptycho2DDataset
        A ptychography dataset to use
    nbins : int
        The number of bins to use for the SNR curve
    
    Returns
    -------
    spectrum : t.tensor
        An image of the spectral signal rate
    freqs : t.tensor
        The frequencies at which the SSNR is estimated
    SSNR : t.tensor
        The estimated SSNR
    
    """

    try:
        scan_hull = spatial.ConvexHull(dataset.translations[:,:2].cpu().numpy())
        scan_area = scan_hull.volume
    except spatial._qhull.QhullError as e:
        scan_area = None
        
    ewg = cdtools.tools.initializers.exit_wave_geometry
    obj_basis = ewg(
        dataset.detector_geometry['basis'],
        dataset[0][1].shape,
        dataset.wavelength,
        dataset.detector_geometry['distance'],
    )

    det_conj_fov_area = np.linalg.norm(
        np.cross(obj_basis[:,0]*dataset.patterns.shape[-2],
                 obj_basis[:,1]*dataset.patterns.shape[-1])
    )

    if scan_area is not None:
        scale_factor = det_conj_fov_area / scan_area
    else:
        warnings.warn("The scan points in this dataset are all colinear. The mean pattern will be calculated rather than a scaled mean based on the scanned area.")
        scale_factor = 1/len(dataset)

    mask = dataset.mask.cpu().numpy().astype(int)
    sum_pattern = dataset.mask * t.sum(dataset.patterns, dim=0) * scale_factor
    sum_pattern = sum_pattern.cpu().numpy()

    # TODO this assumes orthogonal axes
    pix_sizes = np.linalg.norm(obj_basis, axis=0)
    i_freqs = np.fft.fftshift(np.fft.fftfreq(
        sum_pattern.shape[0],d=pix_sizes[0]))
    j_freqs = np.fft.fftshift(np.fft.fftfreq(
        sum_pattern.shape[1],d=pix_sizes[1]))

    
    Js,Is = np.meshgrid(j_freqs,i_freqs)
    Rs = np.sqrt(Is**2+Js**2)
    max_i = np.max(i_freqs)
    max_j = np.max(j_freqs)
    frc_range = [0, max(max_i,max_j)]

    sum_spectrum, frc_bins = np.histogram(Rs, bins=nbins, range=frc_range,
                                          weights=sum_pattern)
    sum_spectrum_sq, frc_bins = np.histogram(Rs, bins=nbins, range=frc_range,
                                             weights=sum_pattern**2)

    
    n_pix, frc_bins = np.histogram(Rs, bins=nbins, range=frc_range,
                                   weights=mask)
    mean_spectrum = sum_spectrum / n_pix

    pattern_snr = sum_spectrum_sq / sum_spectrum

    return sum_pattern, frc_bins[:-1], mean_spectrum
        
