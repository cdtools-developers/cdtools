from __future__ import division, print_function

import torch as t
import numpy as np
from CDTools.tools import cmath
from CDTools.tools import image_processing as ip
from scipy import fftpack

__all__ = ['orthogonalize_probes','standardize', 'synthesize_reconstructions',
           'calc_consistency_prtf']


from matplotlib import pyplot as plt
def orthogonalize_probes(probes):
    """Orthogonalizes a set of incoherently mixing probes
    
    The strategy is to define a reduced orthogonal basis that spans
    all of the retrieved probes, and then build the density matrix
    defined by the probes in that basis. After diagonalization, the
    eigenvectors can be recast into the original basis and returned
    
    Args:
        probes (t.Tensor) : n x (image) size tensor, a stack of probes
    
    Returns:
        (t.Tensor) : n x (image) size tensor, a stack of probes
    """

    try:
        probes = cmath.torch_to_complex(probes.detach().cpu())
        send_to_torch = True
    except:
        send_to_torch = False

    bases = []
    coefficients = np.zeros((probes.shape[0],probes.shape[0]), dtype=np.complex64)
    for i, probe in enumerate(probes):
        ortho_probe = np.copy(probe)
        for j, basis in enumerate(bases):
            coefficients[j,i] = np.sum(basis.conj()*ortho_probe)
            ortho_probe -= basis * coefficients[j,i]
             

        coefficients[i,i] = np.sqrt(np.sum(np.abs(ortho_probe)**2))
        bases.append(ortho_probe / coefficients[i,i])


    density_mat = coefficients.dot(np.conj(coefficients).transpose())
    eigvals, eigvecs = np.linalg.eigh(density_mat)

    ortho_probes = []
    for i in range(len(eigvals)):
        coefficients = np.sqrt(eigvals[i]) * eigvecs[:,i]
        probe = np.zeros(bases[0].shape, dtype=np.complex64)
        for coefficient, basis in zip(coefficients, bases):
            probe += basis * coefficient
        ortho_probes.append(probe)

    
    if send_to_torch:
        return cmath.complex_to_torch(np.stack(ortho_probes[::-1]))
    else:
        return np.stack(ortho_probes[::-1])
        


def standardize(probe, obj, obj_slice=None, correct_ramp=False):
    """Standardizes a probe and object to prepare them for comparison

    There are a number of ambiguities in the definition of a ptychographic
    reconstruction. This function makes an explicit choice for each ambiguity
    to allow comparisons between independent reconstructions without confusing
    these ambiguities for real differences between the reconstructions.

    The ambiguities and standardizations are:
    * Probe and object can be scaled inversely to one another
        * So we set the probe intensity to an average per-pixel value of 1
    * The probe and object can aquire equal and opposite phase ramps
        * So we set the centroid of the FFT of the probe to zero frequency
    * The probe and object can each acquire an arbitrary overall phase
        * So we set the phase of the sum of all values of both the probe and object to 0

    When dealing with the properties of the object, a slice is used by
    default as the edges of the object often are dominated by unphysical
    noise. The default slice is from 3/8 to 5/8 of the way across. If the
    probe is actually a stack of incoherently mixing probes, then the
    dominant probe mode (assumed to be the first in the list) is used, but
    all the probes are updated with the same factors.

    Args:
        probe (t.tensor) : tensor or numpy array storing a retrieved probe or stack of incoherently mixed probes
        obj (t.tensor) : tensor or numpy array storing a retrieved probe
        obj_slice (slice) : optional, a slice to take from the object for calculating normalizations
        correct_ramp (bool) : Default False, whether to correct for the relative phase ramps

    Returns:
        (t.tensor) : The standardized probe
        (t.tensor) : The standardized object

    """
    # First, we normalize the probe intensity to a fixed value.
    probe_np = False
    if isinstance(probe, np.ndarray):
        probe = cmath.complex_to_torch(probe).to(t.float32)
        probe_np = True
    obj_np = False
    if isinstance(obj, np.ndarray):
        obj = cmath.complex_to_torch(obj).to(t.float32)
        obj_np = True

    # If this is a single probe and not a stack of probes
    if len(probe.shape) == 3:
        probe = probe[None,...]
        single_probe = True
    else:
        single_probe = False

        
    normalization = t.sqrt(t.sum(cmath.cabssq(probe[0])) / (len(probe[0].view(-1))/2))
    probe = probe / normalization
    obj = obj * normalization

    # Default slice of the object to use for alignment, etc.
    if obj_slice is None:
        obj_slice = np.s_[(obj.shape[0]//8)*3:(obj.shape[0]//8)*5,
                          (obj.shape[1]//8)*3:(obj.shape[1]//8)*5]

    
    if correct_ramp:
        # Need to check if this is actually working and, if noy, why not
        center_freq = ip.centroid_sq(cmath.fftshift(t.fft(probe[0],2)),comp=True)
        center_freq -= (t.tensor(probe[0].shape[:-1]) // 2).to(t.float32)
        center_freq /= t.tensor(probe[0].shape[:-1]).to(t.float32)


    
        Is, Js = np.mgrid[:probe[0].shape[0],:probe[0].shape[1]]
        probe_phase_ramp = cmath.expi(2*np.pi *
                                      (center_freq[0] * t.tensor(Is).to(t.float32) +
                                       center_freq[1] * t.tensor(Js).to(t.float32)))
        probe = cmath.cmult(probe, cmath.cconj(probe_phase_ramp))
        Is, Js = np.mgrid[:obj.shape[0],:obj.shape[1]]
        obj_phase_ramp = cmath.expi(2*np.pi *
                                    (center_freq[0] * t.tensor(Is).to(t.float32) +
                                     center_freq[1] * t.tensor(Js).to(t.float32)))
        obj = cmath.cmult(obj, obj_phase_ramp)

    
    # Then, we set them to consistent absolute phases
    
    obj_angle = cmath.cphase(t.sum(obj[obj_slice],dim=(0,1)))
    obj = cmath.cmult(obj, cmath.expi(-obj_angle))

    for i in range(probe.shape[0]):
        probe_angle = cmath.cphase(t.sum(probe[i],dim=(0,1)))
        probe[i] = cmath.cmult(probe[i], cmath.expi(-probe_angle))
        
    if single_probe:
        probe = probe[0]
        
    if probe_np:
        probe = cmath.torch_to_complex(probe.detach().cpu())
    if obj_np:
        obj = cmath.torch_to_complex(obj.detach().cpu())
    
    return probe, obj
    


def synthesize_reconstructions(probes, objects, use_probe=False, obj_slice=None, correct_ramp=False):
    """Takes a collection of reconstructions and outputs a single synthesized probe and object
    
    The function first standardizes the sets of probes and objects using the
    standardize function, passing through the relevant options. Then it
    calculates the closest overlap of subsequent frames to subpixel
    precision and uses a sinc interpolation to shift all the probes and objects
    to a common frame. Then the images are summed.
    
    Args:
        probes (list) : A list of probes or stacks of probe modes
        objects (list) : A list of objects
        use_probe (bool) : Default False, whether to use the probe or object for alignment
        obj_slice (slice) : Optional, A slice of the object to use for alignment and normalization
        correct_ramp (bool) : Default False, whether to correct for a relative phase ramp in the probe and object

    Returns:
        (array_like) : The synthesized probe
        (array_like) : The synthesized object
        (list) : a list of standardized objects, for further processing
    
    """
    
    probe_np = False
    if isinstance(probes[0], np.ndarray):
        probes = [cmath.complex_to_torch(probe).to(t.float32) for probe in probes]
        probe_np = True
    obj_np = False
    if isinstance(objects[0], np.ndarray):
        objects = [cmath.complex_to_torch(obj).to(t.float32) for obj in objects]
        obj_np = True

    
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
        

        obj = ip.sinc_subpixel_shift(obj,np.array(shift))

        if len(probe.shape) == 4:
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
        synth_probe = cmath.torch_to_complex(synth_probe)
    if obj_np:
        synth_obj = cmath.torch_to_complex(synth_obj)
        obj_stack = [cmath.torch_to_complex(obj) for obj in obj_stack]

    return synth_probe/(i+2), synth_obj/(i+2), obj_stack



def calc_consistency_prtf(synth_obj, objects, basis, obj_slice=None,nbins=None):
    """Calculates a PRTF between each the individual objects and an averaged one
    
    The consistency PRTF at any given spatial frequency is defined as the ratio
    between the intensity of any given reconstruction and the intensity
    of a synthesized or averaged reconstruction at that spatial frequency.
    Typically, the PRTF is averaged over spatial frequencies with the same
    magnitude.
    
    Args:
        synth_obj (t.Tensor) : The synthesized object in the numerator of the PRTF
        objects (list): A list of objects or diffraction patterns for the denomenator of the PRTF
        basis (array_like) : The basis for the reconstruction array to allow output in physical unit
        obj_slice : Optional, a slice of the objects to use for calculating the PRTF
        nbinbs (int) : Optional, number of bins to use in the histogram. Defaults to a sensible value

    Returns:
        (t.Tensor) : The frequencies for the PRTF
        (t.Tensor) : The values of the PRTF
    """

    obj_np = False
    if isinstance(objects[0], np.ndarray):
        objects = [cmath.complex_to_torch(obj).to(t.float32) for obj in objects]
        obj_np = True
    if isinstance(synth_obj, np.ndarray):
        synth_obj = cmath.complex_to_torch(synth_obj).to(t.float32)

        
    if obj_slice is None:
        obj_slice = np.s_[(objects[0].shape[0]//8)*3:(objects[0].shape[0]//8)*5,
                          (objects[0].shape[1]//8)*3:(objects[0].shape[1]//8)*5]

    if nbins is None:
        nbins = np.max(synth_obj[obj_slice].shape) // 4
    
    synth_fft = cmath.cabssq(cmath.fftshift(t.fft(synth_obj[obj_slice],2))).numpy()

    
    di = np.linalg.norm(basis[:,0]) 
    dj = np.linalg.norm(basis[:,1])
    
    i_freqs = fftpack.fftshift(fftpack.fftfreq(synth_fft.shape[0],d=di))
    j_freqs = fftpack.fftshift(fftpack.fftfreq(synth_fft.shape[1],d=dj))
    
    Js,Is = np.meshgrid(j_freqs,i_freqs)
    Rs = np.sqrt(Is**2+Js**2)
    
    
    synth_ints, bins = np.histogram(Rs,bins=nbins,weights=synth_fft)

    prtfs = []
    for obj in objects:
        obj = obj[obj_slice]
        single_fft = cmath.cabssq(cmath.fftshift(t.fft(obj,2))).numpy() 
        single_ints, bins = np.histogram(Rs,bins=nbins,weights=single_fft)

        prtfs.append(synth_ints/single_ints)


    if not obj_np:
        bins = t.Tensor(bins)
        prtfs = t.Tensor(prtfs)
        
    return bins[:-1], np.mean(prtfs,axis=0)
        
