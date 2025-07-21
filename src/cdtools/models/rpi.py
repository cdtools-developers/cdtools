import torch as t
from cdtools.models import CDIModel
from cdtools import tools
from cdtools.tools import plotting as p
from cdtools.tools.interactions import RPI_interaction
from cdtools.tools import initializers
from scipy.ndimage import binary_dilation
import numpy as np
from copy import copy
import time

__all__ = ['RPI']

#
# This model has works a bit differently from a ptychography model
# because a typical RPI dataset will have lots of images, each of which
# can be reconstructed on it's own.
#
# I made the decision that the model itself will only store one object guess,
# so it can't (for example) simultaneously reconstruct images from all the
# diffraction patterns within a large dataset. This could be possible to
# implement in the future, but is not implemented at present.
#
# To support the approach I chose, I have added a "subset" option to all of the
# optimization functions. To do an RPI reconstruction, the "subset" option
# should be set to a container (list, set, etc) which includes only the index
# of the diffraction pattern to be reconstructed. A few additional decisions:
#
# 1) Constructing a model from a dataser will automatically instantiate
#    the object guess from the first diffraction pattern in the dataset
# 5) A new function can be written to reconstruct the entire dataset by running through each pattern one at a time.
#
# Final note: It is worth seeing whether it is possible to include the
# probe propagation explicitly as a parameter which can be reconstructed
# via gradient descent.
#
#

__all__ = ['RPI']

class RPI(CDIModel):
    
    def __init__(
            self,
            wavelength,
            detector_geometry,
            probe_basis,
            probe,
            obj_guess, 
            background=None,
            mask=None,
            saturation=None,
            obj_support=None,
            oversampling=1,
            weight_matrix=False,
            exponentiate_obj=False,
            phase_only=False,
            high_NA=False,
            propagation_distance=0,
            units='um',
            dtype=t.float32,
    ):
        
        super(RPI, self).__init__()

        complex_dtype = (t.ones([1], dtype=dtype) +
                         1j * t.ones([1], dtype=dtype)).dtype
        
        self.register_buffer('wavelength',
                             t.as_tensor(wavelength, dtype=dtype))
        self.store_detector_geometry(detector_geometry,
                                     dtype=dtype)

        # NOTE: It is required that the probe basis match the exit wave
        # basis. If, for example, the probe reconstruction from ptychography
        # used a bandlimiting constraint and had a larger basis, the user is
        # expected to upsample it explicitly before doing RPI.
        self.register_buffer('probe_basis',
                             t.as_tensor(probe_basis, dtype=dtype))

        scale_factor = t.as_tensor([probe.shape[-1]/obj_guess.shape[-1],
                                 probe.shape[-2]/obj_guess.shape[-2]])
        self.register_buffer('obj_basis',
                             (self.probe_basis * scale_factor).to(dtype=dtype))

        if saturation is None:
            self.saturation = None
        else:
            self.register_buffer('saturation',
                                 t.as_tensor(saturation, dtype=dtype))

        # not sure how to make this a buffer, or if I have to
        self.units = units

        if mask is None:
            self.mask = None
        else:
            self.register_buffer('mask',
                                 t.as_tensor(mask, dtype=t.bool))

        self.register_buffer('probe', t.as_tensor(probe, dtype=complex_dtype))


        self.register_buffer('exponentiate_obj',
                             t.as_tensor(exponentiate_obj, dtype=bool))

        self.register_buffer('phase_only',
                             t.as_tensor(phase_only, dtype=bool))

        self.register_buffer('high_NA',
                             t.as_tensor(high_NA, dtype=bool))


        # We always use multi-modes to store the object, so we convert it
        # if we just get a single 2D array as an input
        if obj_guess.dim() == 2:
            obj_guess = obj_guess[None, :, :]
        
        self.obj = t.nn.Parameter(t.as_tensor(obj_guess, dtype=complex_dtype))
        
        self.weights = t.nn.Parameter(
            t.eye(probe.shape[0], dtype=complex_dtype))

        if not weight_matrix:
            self.weights.requires_grad=False
        
        if background is None:
            background = 1e-6 * t.ones(self.probe[0].shape,
                                       dtype=dtype)

        self.register_buffer('background',
                             t.as_tensor(background, dtype=t.float32))

        if obj_support is None:
            obj_support = t.ones_like(self.obj[0, ...], dtype=int)

        self.register_buffer('obj_support',
                             t.as_tensor(obj_support, dtype=int))
        
        self.obj.data = self.obj * self.obj_support[None, ...]
        
        self.register_buffer('oversampling',
                             t.as_tensor(oversampling, dtype=int))

        self.register_buffer('propagation_distance',
                             t.as_tensor(propagation_distance, dtype=dtype))

        # The propagation direction of the probe. For now it's fixed,
        # but perhaps it would need to be updated in the future
        self.register_buffer('prop_dir',
                             t.as_tensor([0, 0, 1], dtype=dtype))

        if high_NA:
            k_map, intensity_map = \
                tools.propagators.generate_high_NA_k_intensity_map(
                    self.probe_basis,
                    self.get_detector_geometry()['basis'] / oversampling,
                    [oversampling * d for d in self.background.shape],
                    self.get_detector_geometry()['distance'],
                    self.wavelength,
                    dtype=t.float32,
                    lens=False)

            self.register_buffer('k_map',
                                 t.as_tensor(k_map, dtype=dtype))
            self.register_buffer('intensity_map',
                                 t.as_tensor(intensity_map, dtype=dtype))

        else:
            self.k_map = None
            self.intensity_map = None



    @classmethod
    def from_dataset(
            cls,
            dataset,
            probe,
            obj_size=None,
            background=None,
            mask=None,
            n_modes=1,
            saturation=None,
            scattering_mode=None,
            oversampling=1,
            initialization='random',
            weight_matrix=False,
            exponentiate_obj=False,
            phase_only=False,
            high_NA=False,
            probe_threshold=0,
            dtype=t.float32,
    ):
        complex_dtype = (t.ones([1], dtype=dtype) +
                         1j * t.ones([1], dtype=dtype)).dtype
        
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # always do this on the cpu
        get_as_args = dataset.get_as_args
        dataset.get_as(device='cpu')
        
        # We only need the patterns here, not the inputs associated with them.
        _, patterns = dataset[:]
        dataset.get_as(*get_as_args[0],**get_as_args[1])
            
        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        ew_basis = ewg(det_basis,
                       det_shape,
                       wavelength,
                       distance,
                       oversampling=oversampling)

        probe = t.as_tensor(probe)
        
        # Potentially need all of this orientation stuff later
        
        #if hasattr(dataset, 'sample_info') and \
        #   dataset.sample_info is not None and \
        #   'orientation' in dataset.sample_info:
        #    surface_normal = dataset.sample_info['orientation'][2]
        #else:
        #    surface_normal = np.array([0.,0.,1.])

        # If this information is supplied when the function is called,
        # then we override the information in the .cxi file
        #if scattering_mode in {'t', 'transmission'}:
        #    surface_normal = np.array([0.,0.,1.])
        #elif scattering_mode in {'r', 'reflection'}:
        #    outgoing_dir = np.cross(det_basis[:,0], det_basis[:,1])
        #    outgoing_dir /= np.linalg.norm(outgoing_dir)
        #    surface_normal = outgoing_dir + np.array([0.,0.,1.])
        #    surface_normal /= np.linalg.norm(surface_normal)


        if background is None and hasattr(dataset, 'background') \
           and dataset.background is not None:
            background = t.sqrt(dataset.background)
        elif background is not None:
            background = t.sqrt(t.as_tensor(background).to(dtype=t.float32))

        det_geo = dataset.detector_geometry

        # If no mask is given, but one exists in the dataset, load it.
        if mask is None and hasattr(dataset, 'mask') \
           and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)

        # This will be superceded later by a call to init_obj, but it sets
        # the shape
        if obj_size is None:
            obj_size = (np.array(self.probe.shape[-2:]) // 2).astype(int)

        dummy_init_obj = t.ones([n_modes, obj_size[0], obj_size[1]],
                                dtype=complex_dtype)

        # This defines an object support in real space using the probe
        # intensities, if requested
        probe_intensity = t.sqrt(t.sum(t.abs(probe)**2,axis=0))
        probe_fft = tools.propagators.far_field(probe_intensity)
        pad0l = (probe.shape[-2] - obj_size[-2])//2
        pad0r = probe.shape[-2] - obj_size[-2] - pad0l
        pad1l = (probe.shape[-1] - obj_size[-1])//2
        pad1r = probe.shape[-1] - obj_size[-1] - pad1l
        probe_lr_fft = probe_fft[pad0l:-pad0r,pad1l:-pad1r]
        probe_lr = t.abs(tools.propagators.inverse_far_field(probe_lr_fft))

        obj_support = probe_lr > t.max(probe_lr) * probe_threshold
        obj_support = t.as_tensor(binary_dilation(obj_support))

        rpi_object = cls(wavelength, det_geo, ew_basis,
                         probe, dummy_init_obj, 
                         background=background, mask=mask,
                         saturation=saturation,
                         obj_support=obj_support,
                         oversampling=oversampling,
                         exponentiate_obj=exponentiate_obj,
                         phase_only=phase_only,
                         high_NA=high_NA,
                         weight_matrix=weight_matrix)

        # I don't love this pattern, where I do the "real" obj initialization
        # after creating the rpi object. But, I chose this so that I could
        # have a function to reinitialize the obj, for repeat reconstructions
        # using the same rpi object, without repeating code. There probably
        # is a better pattern for doing this.
        rpi_object.init_obj(initialization,
                            pattern=dataset.patterns[0])

        if exponentiate_obj:
            rpi_object.obj.data = -1j * t.log(rpi_object.obj.data)
        
        if phase_only:
            rpi_object.obj.data.imag[:] = 0

        
        return rpi_object

    @classmethod
    def from_calibration(
            cls,
            calibration,
            obj_size=None,
            n_modes=1,
            saturation=None, # TODO can we get this from the calibration?
            exponentiate_obj=False,
            phase_only=False,
            initialization='random',
            dtype=t.float32
    ):
        
        complex_dtype = (t.ones([1], dtype=dtype) +
                         1j * t.ones([1], dtype=dtype)).dtype

        
        wavelength = t.as_tensor(calibration['wavelength'], dtype=dtype)
        probe_basis = t.as_tensor(calibration['obj_basis'], dtype=dtype)
        # TODO this will fail if the probe from the calibration was restricted
        # in Fourier space
        probe = t.as_tensor(calibration['probe'], dtype=complex_dtype)
        if 'background' in calibration:
            background = t.sqrt(t.as_tensor(calibration['background'],
                                            dtype=dtype))
        else:
            background = t.zeros_like(probe.real)
        if 'mask' in calibration:
            mask = t.as_tensor(calibration['mask'], dtype=t.bool)
        else:
            mask = t.ones_like(probe.real, dtype=t.bool)

        # This will be superceded later by a call to init_obj, but it sets
        # the shape
        if obj_size is None:
            obj_size = (np.array(self.probe.shape[-2:]) // 2).astype(int)

        dummy_init_obj = t.ones([n_modes, obj_size[0], obj_size[1]],
                                dtype=complex_dtype)

        # Pretty sure that this will not work for reflection-mode
        det_geo = {'distance': 1,
                   'basis': wavelength / probe_basis} 
                
        rpi_object = cls(
            wavelength,
            det_geo,
            probe_basis,
            probe,
            dummy_init_obj,
            background=background,
            mask=mask,
            exponentiate_obj=exponentiate_obj,
            phase_only=phase_only,
        )

        rpi_object.init_obj(initialization)
        
        if exponentiate_obj:
            rpi_object.obj.data = -1j * t.log(rpi_object.obj.data)

        if phase_only:
            rpi_object.obj.data.imag[:] = 0

        return rpi_object

    
    def init_obj(
            self,
            initialization_type,
            obj_shape=None,
            n_modes=None,
            pattern=None,
    ):
        
        # I think something to do with the fact that the object is defined
        # on a coarser grid needs to be accounted for here that is not
        # accounted for yet

        if initialization_type.lower().strip() == 'random':
            self.random_init(obj_shape=obj_shape, n_modes=n_modes)
        elif initialization_type.lower().strip() == 'uniform':
            self.uniform_init(obj_shape=obj_shape, n_modes=n_modes)
        elif initialization_type.lower().strip() == 'spectral':
            if pattern is None:
                raise KeyError(
                    'A pattern must be supplied for spectral initialization')
            
            self.spectral_init(pattern=pattern,
                               obj_shape=obj_shape,
                               n_modes=n_modes)
        else:
            raise KeyError('Initialization "' + str(initialization) + \
                           '" invalid - use "spectral", "uniform", or "random"')

    
    def get_obj_shape_and_n_modes(self, obj_shape=None, n_modes=None):
        """Sets defaults for obj shape and n modes"""
        
        if obj_shape == None:
            if hasattr(self, 'obj'):
                obj_shape = self.obj.shape[-2:]
            else:
                obj_size = (np.array(self.probe.shape[-2:]) // 2).astype(int)
                obj_shape = [obj_size, obj_size]

        if n_modes == None:
            if hasattr(self, 'obj'):
                n_modes = self.obj.shape[0]
            else:
                n_modes = 1

        return n_modes, obj_shape
        

    def uniform_init(self, obj_shape=None,  n_modes=None):
        """Sets a uniform object initialization"""
        
        n_modes, obj_shape = self.get_obj_shape_and_n_modes(
            obj_shape=obj_shape, n_modes=n_modes)

        obj_guess = t.ones(
            [n_modes,]+list(obj_shape),
            dtype=self.probe.dtype,
            device=self.probe.device,
        )
        
        if hasattr(self, 'obj'):
            self.obj.data = obj_guess
        else:
            self.obj = t.nn.Parameter(obj_guess)

        
    def random_init(self,  obj_shape=None, n_modes=None):
        """Sets a uniform amplitude object initialization with random phase"""
        
        n_modes, obj_shape = self.get_obj_shape_and_n_modes(
            obj_shape=obj_shape, n_modes=n_modes)
        
        obj_guess = t.exp(
            2j * np.pi * t.rand([n_modes,]+list(obj_shape))).to(
                dtype=self.probe.dtype, device=self.probe.device)

        if hasattr(self, 'obj'):
            self.obj.data = obj_guess
        else:
            self.obj = t.nn.Parameter(obj_guess)

        
    def spectral_init(self, pattern, obj_shape=None, n_modes=None):
        """Initializes the object with a spectral method"""
        
        n_modes, obj_shape = self.get_obj_shape_and_n_modes(
            obj_shape=obj_shape, n_modes=n_modes)
        
        if self.background is not None:
            background = self.background**2
        else:
            background = None
        
        obj_guess = initializers.RPI_spectral_init(
            pattern,
            self.probe,
            obj_shape,
            mask=self.mask,
            background=background,
            n_modes=n_modes).to(
                dtype=self.obj.dtype, device=self.obj.device)
        
        if hasattr(self, 'obj'):
            self.obj.data = obj_guess
        else:
            self.obj = t.nn.Parameter(obj_guess)

    
    # Needs work
    def interaction(self, index, *args):
        # including *args allows this to work with all sorts of datasets
        # that might include other information in with the index in their
        # "input" parameters (such as translations for a ptychography dataset).
        # This makes it seamless to use such a dataset even though those
        # extra arguments will not be used.
        
        all_exit_waves = []

        # Mix the probes with the weight matrix
        prs = t.sum(self.weights[..., None, None] * self.probe, axis=-3)

        if self.exponentiate_obj:
            if self.phase_only:
                obj = t.exp(1j*self.obj.real)
            else:
                obj = t.exp(1j*self.obj)
        else:
            obj = self.obj

        
        for i in range(self.probe.shape[0]):
            pr = prs[i]
            # Here we have a 3D probe (one single mode)
            # and a 4D object (multiple modes mixing incoherently)
            exit_waves = RPI_interaction(pr,
                                         self.obj_support * obj)
                
            all_exit_waves.append(exit_waves)

        # This creates a bunch of modes generated from all possible combos
        # of the probe and object modes all strung out along the first index
        output = t.cat(all_exit_waves)
        # If we have multiple indexes input, we unsqueeze and repeat the stack
        # of wavefields enough times to simulate each requested index. This
        # seems silly, but it enables (for example) one to do a reconstruction
        # from a set of diffraction patterns that are all known to be from the
        # same object.
        try:
            # will fail if index has no length, for example when index
            # is just an int. In this case, we just do nothing instead
            output = output.unsqueeze(0).repeat(1,len(index),1,1,1)
        except TypeError:
            pass
        return output


    def forward_propagator(self, wavefields):
        if self.high_NA:
            wavefields = wavefields.reshape([wavefields.shape[0]] + [-1] + list(wavefields.shape[-2:]))
            return tools.propagators.high_NA_far_field(
                wavefields,self.k_map,intensity_map=self.intensity_map)
        else:
            return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)

    
    def measurement(self, wavefields):
        # Here I'm taking advantage of an undocumented feature in the
        # incoherent_sum measurement function where it will work with
        # a 4D wavefield array as well as a 5D array.
        m = tools.measurements.quadratic_background(
            wavefields,
            self.background,
            measurement=tools.measurements.incoherent_sum,
            saturation=self.saturation,
            oversampling=self.oversampling)
        return m
    
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)

    def regularizer(self, factors):
        if self.obj.shape[0] == 1:
            return factors[0] * t.sum(t.abs(self.obj[0,:,:])**2)
        else:
            return factors[0] * t.sum(t.abs(self.obj[0,:,:])**2) \
                + factors[1] * t.sum(t.abs(self.obj[1:,:,:])**2)
        

    def sim_to_dataset(self, args_list):
        raise NotImplementedError('No sim to dataset yet, sorry!')

    plot_list = [
        ('Root Sum Squared Amplitude of all Probes',
         lambda self, fig: p.plot_amplitude(
             np.sqrt(np.sum((t.abs(t.sum(self.weights[..., None, None].detach() * self.probe, axis=-3))**2).cpu().numpy(),axis=0)),
             fig=fig, basis=self.probe_basis)),
        ('Object Amplitude',
         lambda self, fig: p.plot_amplitude(
             self.obj,
             fig=fig,
             basis=self.obj_basis,
             units=self.units),
         lambda self: not self.exponentiate_obj),
        ('Object Phase',
         lambda self, fig: p.plot_phase(
             self.obj,
             fig=fig,
             basis=self.obj_basis,
             units=self.units),
         lambda self: not self.exponentiate_obj),
        ('Real Part of T', 
         lambda self, fig: p.plot_real(
             self.obj,
             fig=fig,
             basis=self.obj_basis,
             units=self.units,
             cmap='cividis'),
         lambda self: self.exponentiate_obj),
        ('Imaginary Part of T',
         lambda self, fig: p.plot_imag(
             self.obj,
             fig=fig,
             basis=self.obj_basis,
             units=self.units),
         lambda self: self.exponentiate_obj),
    ]


    def save_results(self, dataset=None):
        # dataset is set as a kwarg here because it isn't needed, but the
        # common pattern is to pass a dataset. This makes it okay if one
        # continues to use that standard pattern

        # This will save out everything needed to recreate the object
        # in the same state, but it's not the best formatted. For example,
        # "background" stores the square root of the background, etc.
        base_results = super().save_results()
        
        probe_basis = self.probe_basis.detach().cpu().numpy()
        obj_basis = self.obj_basis.detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        
        # We only save out the top object mode, which is the final result
        # The full object is still saved in the state_dict that is
        # included in the base_results
        obj = self.obj[0].detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        
        results = {
            'probe_basis': probe_basis,
            'obj_basis': obj_basis,
            'probe': probe,
            'obj': obj,
            'background': background
        }

        return {**base_results, **results}

