import torch as t
from cdtools.models import CDIModel
from cdtools.datasets import Ptycho2DDataset
from cdtools import tools
from cdtools.tools import plotting as p
from cdtools.tools.propagators import generate_generalized_angular_spectrum_propagator as ggasp
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from copy import copy

__all__ = ['Bragg2DPtycho']

#
# Key ideas:
# 1) To a first approximation, do the reconstruction on a parallelogram
#    shaped grid on the sample which is conjugate to the detector coordinates
# 2) Simulate a probe in those same coordinates, but propagate it back and
#    forth (along with the translations), using the angular spectrum method
# 3) Apply a correction to the simulated data to account for the tilt of the
#    sample with respect to the detector
# 4) Include a correction for the thickness of the sample
#

#
# How to do this properly?
# First thing to note is that the two corrections (probe propagation before
# interaction and high-NA correction for the final diffraction measurement)
# should be able to be turned on separately, since they show up in different
# situations. In fact, I would like to focus on the first aspect initially
# since I think that's the dominant issue we will contend with at CSX.
#

#
# It also should be possible to choose an "auto" setting for the two
# corrections, since the geometry information given should be enough to
# decide if the correction is needed.
# Probably the automatic check will have to be very conservative for the
# probe propagation side since the model has no information about the
# expected numerical aperture of the probe.
#

#
# I'm worried that the propagation doesn't happen along the correct
# direction, if a phase ramp is expected to be baked in to the
# retrieved focal spot. Unclear if this is the case though.
#
# Retrieved focal spot should have the implicit phase ramp subtracted
# (that is, it should be the focal spot along the sample plane, but with
# the e^ikz dependence removed). Therefore, the original e^ikz dependence
# should be easy to re-add jusy by using the propagate_along feature
# in ggasp. So I believe this should not be a problem
#

beam_basis = np.array([[0,-1],
                       [-1,0],
                       [0,0]])

class Bragg2DPtycho(CDIModel):

    def __init__(
            self,
            wavelength,
            detector_geometry,
            obj_basis,
            probe_guess,
            obj_guess,
            min_translation=t.tensor([0, 0],dtype=t.float32),
            probe_basis=None,
            median_propagation=t.tensor(0, dtype=t.float32),
            background=None,
            translation_offsets=None, mask=None,
            weights=None,
            translation_scale=1, saturation=None,
            probe_support=None,
            oversampling=1,
            propagate_probe=True,
            correct_tilt=True,
            lens=False,
            units='um',
            dtype=t.float32,
            obj_view_crop=0,
    ):

        # We need the detector geometry
        # We need the probe basis (but in this case, we don't need the surface
        # normal because it comes implied by the probe basis
        # we do need the detector slice I suppose
        # The min translation is also needed
        # The median propagation should be needed as well
        # translation_offsets can stay 2D for now
        # propagate_probe and correct_tilt are important!

        super(Bragg2DPtycho, self).__init__()
        self.register_buffer('wavelength',
                             t.as_tensor(wavelength, dtype=dtype))
        self.store_detector_geometry(detector_geometry,
                                     dtype=dtype)

        self.register_buffer('min_translation',
                             t.as_tensor(min_translation, dtype=dtype))
        self.register_buffer('median_propagation',
                             t.as_tensor(median_propagation, dtype=dtype))

        self.register_buffer('obj_basis',
                             t.as_tensor(obj_basis, dtype=dtype))
        if probe_basis is None:
            self.register_buffer('probe_basis',
                                 t.as_tensor(obj_basis, dtype=dtype))
        else:
            self.register_buffer('probe_basis',
                                 t.as_tensor(probe_basis, dtype=dtype))        

            self.units = units

        # calculate the surface normal from the object basis. Reminder
        # that for this model, the object basis should be defined in the
        # plane of the object.
        surface_normal =  np.cross(np.array(obj_basis)[:,1],
                           np.array(obj_basis)[:,0])
        surface_normal /= np.linalg.norm(surface_normal)
        self.register_buffer('surface_normal',
                             t.as_tensor(surface_normal, dtype=dtype))

        if saturation is None:
            self.saturation = None
        else:
            self.register_buffer('saturation',
                                 t.as_tensor(saturation, dtype=dtype))

        if mask is None:
            self.mask = None
        else:
            self.register_buffer('mask',
                                 t.as_tensor(mask, dtype=t.bool))

        probe_guess = t.as_tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.as_tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object
        if probe_guess.dim() > 2:
            probe_norm = 1 * t.max(t.abs(probe_guess[0]))
        else:
            probe_norm = 1 * t.max(t.abs(probe_guess))        
        self.register_buffer('probe_norm', probe_norm.to(dtype))
        
        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)

        # NOTE: I think it makes sense to protect against obj_view_crop
        # being zero or below, because there is nothing else to show outside
        # the object array. No reason to throw an error if, e.g., the user
        # asks for a big padding which goes outside of the actual object array.
        # Just show the full array.

        if obj_view_crop > 0:
            self.obj_view_slice = np.s_[obj_view_crop:-obj_view_crop,
                                        obj_view_crop:-obj_view_crop]
        else:
            self.obj_view_slice = np.s_[:,:]


        if probe_support is None:
            probe_support = t.ones_like(self.probe[0], dtype=t.bool)
        self.register_buffer('probe_support',
                             t.as_tensor(probe_support, dtype=t.bool))
        self.probe.data *= self.probe_support

        if background is None:
            raise NotImplementedError('Issues with this due to probe fourier padding')
            shape = [s//oversampling for s in self.probe[0]]
            background = 1e-6 * t.ones(shape, dtype=t.float32)

        self.background = t.nn.Parameter(background)

        if weights is None:
            self.weights = None
        else:
            # No incoherent + unstable here yet
            self.weights = t.nn.Parameter(t.as_tensor(weights,
                                                   dtype=t.float32))

        if translation_offsets is None:
            self.translation_offsets = None
        else:
            t_o = t.as_tensor(translation_offsets, dtype=t.float32)
            t_o = t_o / translation_scale
            self.translation_offsets = t.nn.Parameter(t_o)

        self.register_buffer('translation_scale',
                             t.as_tensor(translation_scale, dtype=dtype))

        self.register_buffer('oversampling',
                             t.as_tensor(oversampling, dtype=int))

        self.register_buffer('propagate_probe',
                             t.as_tensor(propagate_probe, dtype=bool))
        self.register_buffer('correct_tilt',
                             t.as_tensor(correct_tilt, dtype=bool))

        if correct_tilt:
            k_map, intensity_map = \
                tools.propagators.generate_high_NA_k_intensity_map(
                    self.obj_basis,
                    self.get_detector_geometry()['basis'] / oversampling,
                    [oversampling * d for d in self.background.shape],
                    self.get_detector_geometry()['distance'],
                    self.wavelength,dtype=t.float32,
                    lens=lens)

            self.register_buffer('k_map',
                                 t.as_tensor(k_map, dtype=dtype))
            self.register_buffer('intensity_map',
                                 t.as_tensor(intensity_map, dtype=dtype))

        else:
            self.k_map = None
            self.intensity_map = None

        # The propagation direction of the probe 
        self.register_buffer('prop_dir',
                             t.as_tensor([0, 0, 1], dtype=dtype))

        # This propagator should be able to be multiplied by the propagation
        # distance each time to get a propagator
        universal_propagator = t.angle(ggasp(
            self.probe.shape[-2:],
            self.probe_basis, self.wavelength,
            t.tensor([0, 0, self.wavelength/(2*np.pi)], dtype=t.float32),
            propagation_vector=self.prop_dir,
            dtype=t.complex64,
            propagate_along_offset=True))
        
        # TODO: probably doesn't support non-float-32 dtypes
        self.register_buffer('universal_propagator',
                             universal_propagator)
                            
        

    @classmethod
    def from_dataset(
            cls,
            dataset,
            randomize_ang=0,
            padding=0,
            n_modes=1,
            translation_scale = 1,
            saturation=None,
            probe_support_radius=None,
            propagation_distance=None,
            scattering_mode=None,
            oversampling=1,
            probe_fourier_crop=None,
            propagate_probe=True,
            correct_tilt=True,
            lens=False,
            obj_padding=200,
            obj_view_crop=None,
            units='um',
    ):
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # always do this on the cpu
        get_as_args = dataset.get_as_args
        dataset.get_as(device='cpu')
        (indices, translations), patterns = dataset[:]
        dataset.get_as(*get_as_args[0],**get_as_args[1])

        # Then, generate the exit wave geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        ew_basis =  ewg(det_basis,
                        det_shape,
                        wavelength,
                        distance,
                        oversampling=oversampling)
        
        # now we grab the sample surface normal
        if hasattr(dataset, 'sample_info') and \
           dataset.sample_info is not None and \
           'orientation' in dataset.sample_info:
            surface_normal = dataset.sample_info['orientation'][2]
        else:
            surface_normal = np.array([0.,0.,1.])
            
        # If this information is supplied when the function is called,
        # then we override the information in the .cxi file
        if scattering_mode in {'t', 'transmission'}:
            surface_normal = np.array([0.,0.,1.])
        elif scattering_mode in {'r', 'reflection'}:
            outgoing_dir = np.cross(det_basis[:,0], det_basis[:,1])
            outgoing_dir /= np.linalg.norm(outgoing_dir)
            surface_normal = outgoing_dir + np.array([0.,0.,1.])
            surface_normal /= np.linalg.norm(outgoing_dir)

        # and we use that to generate the probe basis

        ew_normal = np.cross(np.array(ew_basis)[:,1],
                           np.array(ew_basis)[:,0])
        ew_normal /= np.linalg.norm(ew_normal)

        # This is a bit of an odd way to do a projection but I think it's
        # the most compact and reliable. We set up two matrix-vector equations
        # to enforce the two conditions (on the plane normal to the surface
        # normal and in the ew_normal direction from the original point

        mat = np.vstack([np.eye(3) - np.outer(ew_normal,ew_normal),
                         surface_normal])

        # We know that this matrix multiplied by the final result must
        # equal the input vector with a trailing 0, so we can do the
        # projection with a pseudoinverse and removing the last column

        projector = np.linalg.pinv(mat)[:, :3]
        obj_basis = t.Tensor(np.dot(projector, ew_basis))
        
        # Now we need a much better way to handle the translations here
        # than translations_to_pixel
        
        # Next generate the object geometry from the probe geometry and
        # the translations
        p2s = tools.interactions.project_translations_to_sample
        pix_translations, propagations = p2s(obj_basis, translations)


        obj_size, min_translation = tools.initializers.calc_object_setup(
            [s * oversampling for s in det_shape],
            pix_translations,
            padding=obj_padding,
        )

        median_propagation = t.median(propagations)

        # Finally, initialize the probe and  object using this information
        # Because the grid we defined on the sample is projected from the
        # detector conjugate space, we can pretend that the grid is just in
        # that space and use the standard initializations anyway
        probe = tools.initializers.SHARP_style_probe(
            dataset,
            propagation_distance=propagation_distance,
            oversampling=oversampling
        )

            
        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = 1e-6 * t.ones(det_shape,
                                       dtype=t.float32)

        if probe_fourier_crop is not None:
            probe = tools.propagators.far_field(probe)
            probe = probe[...,
                          probe_fourier_crop[0]:-probe_fourier_crop[0],
                          probe_fourier_crop[1]:-probe_fourier_crop[1]]
            probe = tools.propagators.inverse_far_field(probe)
            
            scale_factor = np.array(det_shape) / np.array(probe.shape)
            probe_basis = obj_basis * scale_factor[None,:]
        else:
            probe_basis = obj_basis.clone()


        # Now we initialize all the subdominant probe modes
        probe_max = t.max(t.abs(probe))
        probe_stack = [0.01 * probe_max * t.rand(probe.shape,dtype=probe.dtype) for i in range(n_modes - 1)]
        probe = t.stack([probe,] + probe_stack)

        obj = t.exp(1j*(randomize_ang * (t.rand(obj_size)-0.5)))

        pfc = (probe_fourier_crop if probe_fourier_crop else [0,0])
        if obj_view_crop is None:
            obj_view_crop = min(
                probe.shape[-2] // 2 + pfc[0],
                probe.shape[-1] // 2 + pfc[1]
            )
        if obj_view_crop < 0:
            obj_view_crop += min(
                probe.shape[-2] // 2 + pfc[0],
                probe.shape[-1] // 2 + pfc[1]
            )

        obj_view_crop += obj_padding
        
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset),2)) - 0.5)

        weights = t.ones(len(dataset))
        
        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        if probe_support_radius is not None:
            probe_support = t.zeros(probe[0].shape, dtype=t.bool)
            xs, ys = np.mgrid[:probe.shape[-2], :probe.shape[-1]]
            xs = xs - np.mean(xs)
            ys = ys - np.mean(ys)
            Rs = np.sqrt(xs**2 + ys**2)
            
            probe_support[Rs < probe_support_radius] = 1
            probe = probe * probe_support[None, :, :]
            
        else:
            probe_support = None

        # Here we need to implement a simple condition to choose whether
        # to propagate the probe or not
        if not( propagate_probe is True or propagate_probe is False):
            raise NotImplementedError('No auto option implemented yet')

        if not(correct_tilt is True or correct_tilt is False):
            raise NotImplementedError('No auto option implemented yet')
        
        return cls(wavelength, det_geo, obj_basis, probe, obj,
                   min_translation=min_translation,
                   probe_basis=probe_basis,
                   median_propagation =median_propagation,
                   translation_offsets = translation_offsets,
                   weights=weights, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   probe_support=probe_support,
                   oversampling=oversampling,
                   propagate_probe=propagate_probe,
                   correct_tilt=correct_tilt,
                   lens=lens,
                   obj_view_crop=obj_view_crop,
                   units=units,
                   )
                   
    
    def interaction(self, index, translations):
        pix_trans, props = tools.interactions.project_translations_to_sample(
            self.obj_basis, translations)

        
        pix_trans -= self.min_translation
        props -= self.median_propagation
        
        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]

        Ws = self.weights[index]
        prs = Ws[...,None,None,None] * self.probe * self.probe_support[...,:,:]
        # Now we need to propagate each of the probes


        # TODO: This fails if there is no background set. In that case,
        # there's no way to know the "proper" size of the probe
        # We automatically rescale the probe to match the background size,
        # which allows us to do stuff like let the object be super-resolution,
        # while restricting the probe to the detector resolution but still
        # doing an explicit real-space limitation of the probe
        padding = [self.oversampling * self.background.shape[-2] - prs.shape[-2],
                   self.oversampling * self.background.shape[-1] - prs.shape[-1]]

        # probe propagation
        if self.propagate_probe:
            for j in range(prs.shape[0]):
                # I believe this -1 sign is in error, but I need a dataset with
                # well understood geometry to figure it out
                propagator = t.exp(
                    1j*(props[j]*(2*np.pi)/self.wavelength)
                    * self.universal_propagator)
                prs[j] = tools.propagators.near_field(prs[j], propagator)

        # we do the upscaling after the propagation because the propagator
        # is calculated based on the probe basis (before upsampling)
        if any([p != 0 for p in padding]): # For probe_fourier_crop != 0.
            padding = [padding[-1]//2, padding[-1]-padding[-1]//2,
                       padding[-2]//2, padding[-2]-padding[-2]//2]
            prs = tools.propagators.far_field(prs)
            prs = t.nn.functional.pad(prs, padding)
            prs = tools.propagators.inverse_far_field(prs)

        
        exit_waves = self.probe_norm * tools.interactions.ptycho_2D_sinc(
            prs, self.obj, pix_trans,
            shift_probe=True, multiple_modes=True)

        return exit_waves

        
    def forward_propagator(self, wavefields):
        if self.correct_tilt:
            return tools.propagators.high_NA_far_field(
                wavefields,self.k_map,intensity_map=self.intensity_map)
        else:
            return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        if self.correct_tilt:
            assert NotImplementedError('Backward propagator not defined with tilt correction')
        else:
            return tools.propagators.inverse_far_field(wavefields)

    
    def measurement(self, wavefields):
        return tools.measurements.quadratic_background(
            wavefields,
            self.background,
            measurement=tools.measurements.incoherent_sum,
            saturation=self.saturation,
            oversampling=self.oversampling,
        )

    
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)

    
    def sim_to_dataset(self, args_list):
        # In the future, potentially add more control
        # over what metadata is saved (names, etc.)
        
        # First, I need to gather all the relevant data
        # that needs to be added to the dataset
        entry_info = {'program_name': 'cdtools',
                      'instrument_n': 'Simulated Data',
                      'start_time': datetime.now()}

        surface_normal = self.surface_normal.detach().cpu().numpy()
        xsurfacevec = np.cross(np.array([0.,1.,0.]), surface_normal)
        xsurfacevec /= np.linalg.norm(xsurfacevec)
        ysurfacevec = np.cross(surface_normal, xsurfacevec)
        ysurfacevec /= np.linalg.norm(ysurfacevec)
        orientation = np.array([xsurfacevec, ysurfacevec, surface_normal])
        
        sample_info = {'description': 'A simulated sample',
                       'orientation': orientation}

        
        detector_geometry = self.detector_geometry
        mask = self.mask
        wavelength = self.wavelength
        indices, translations = args_list
        
        # Then we simulate the results
        data = self.forward(indices, translations)

        # And finally, we make the dataset
        return Ptycho2DDataset(translations, data,
                                 entry_info = entry_info,
                                 sample_info = sample_info,
                                 wavelength=wavelength,
                                 detector_geometry=self.get_detector_geometry(),
                                 mask=mask)

    
    def corrected_translations(self,dataset):
        translations = dataset.translations.to(dtype=self.probe.real.dtype,
                                               device=self.probe.device)
        t_offset = tools.interactions.pixel_to_translations(self.obj_basis,self.translation_offsets*self.translation_scale,surface_normal=self.surface_normal)
        return translations + t_offset


    plot_list = [
        ('Basis Probe Fourier Space Amplitudes',
         lambda self, fig: p.plot_amplitude(tools.propagators.inverse_far_field(self.probe), fig=fig)),
        ('Basis Probe Fourier Space Phases',
         lambda self, fig: p.plot_phase(tools.propagators.inverse_far_field(self.probe), fig=fig)),
        ('Basis Probe Real Space Amplitudes, Surface Normal View',
         lambda self, fig: p.plot_amplitude(
             self.probe,
             fig=fig,
             basis=self.probe_basis,
             units=self.units,
         )),
        ('Basis Probe Real Space Phases, Surface Normal View',
         lambda self, fig: p.plot_phase(
             self.probe,
             fig=fig,
             basis=self.probe_basis,
             units=self.units,
         )),
        ('Basis Probe Real Space Amplitudes, Beam View',
         lambda self, fig: p.plot_amplitude(
             self.probe,
             fig=fig,
             basis=self.probe_basis,
             view_basis=beam_basis,
             units=self.units,
         )),
        ('Basis Probe Real Space Phases, Beam View',
         lambda self, fig: p.plot_phase(
             self.probe,
             fig=fig,
             basis=self.probe_basis,
             view_basis=beam_basis,
             units=self.units,
         )),
        ('Object Amplitude, Surface Normal View', 
         lambda self, fig: p.plot_amplitude(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             units=self.units,
         )),
        ('Object Phase, Surface Normal View',
         lambda self, fig: p.plot_phase(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             units=self.units,
         )),
        ('Object Amplitude, Beam View', 
         lambda self, fig: p.plot_amplitude(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             view_basis=beam_basis,
             units=self.units,
         )),
        ('Object Phase, Beam View',
         lambda self, fig: p.plot_phase(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             view_basis=beam_basis,
             units=self.units,
         )),
        ('Object Amplitude, Detector View', 
         lambda self, fig: p.plot_amplitude(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             view_basis=self.det_basis,
             units=self.units,
         )),
        ('Object Phase, Detector View',
         lambda self, fig: p.plot_phase(
             self.obj[self.obj_view_slice],
             fig=fig,
             basis=self.obj_basis,
             view_basis=self.det_basis,
             units=self.units,
         )),
        ('Corrected Translations',
         lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig, units=self.units)),
        ('Background',
         lambda self, fig: plt.figure(fig.number) and plt.imshow(self.background.detach().cpu().numpy()**2))
    ]

    
    def save_results(self, dataset):
        # This will save out everything needed to recreate the object
        # in the same state, but it's not the best formatted. For example,
        # "background" stores the square root of the background, etc.
        base_results = super().save_results()

        # We also save out the main results in a more readable format
        obj_basis = self.obj_basis.detach().cpu().numpy()
        probe_basis = self.probe_basis.detach().cpu().numpy()
        translations=self.corrected_translations(dataset).detach().cpu().numpy()
        original_translations = dataset.translations.detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        oversampling = self.oversampling.cpu().numpy()
        wavelength = self.wavelength.cpu().numpy()

        results = {
            'obj_basis': obj_basis,
            'probe_basis': probe_basis,
            'translations': translations,
            'original_translations': original_translations,
            'probe': probe,
            'obj': obj,
            'background': background,
            'oversampling': oversampling,
            'weights': weights,
            'wavelength': wavelength,
        }

        return {**base_results, **results}
