import torch as t
from cdtools.models import CDIModel
from cdtools.datasets import Ptycho2DDataset
from cdtools import tools
from cdtools.tools import analysis, image_processing
from cdtools.tools import plotting as p
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from copy import copy
from functools import reduce

__all__ = ['Multislice2DPtycho']



class Multislice2DPtycho(CDIModel):

    @property
    def probe(self):
        return t.complex(self.probe_real, self.probe_imag)

    @property
    def obj(self):
        return t.complex(self.obj_real, self.obj_imag)

    def __init__(self,
                 wavelength,
                 detector_geometry,
                 probe_basis,
                 probe_guess, obj_guess, dz, nz,
                 detector_slice=None,
                 surface_normal=np.array([0., 0., 1.]),
                 min_translation=t.Tensor([0, 0]),
                 background=None,
                 translation_offsets=None,
                 mask=None,
                 weights=None,
                 translation_scale=1,
                 saturation=None,
                 probe_support=None,
                 oversampling=1,
                 bandlimit=None,
                 subpixel=True,
                 exponentiate_obj=True,
                 fourier_probe=False,
                 prevent_aliasing=True,
                 phase_only=False,
                 units='um',
                 panel_plot_mode=False,
                 plot_level=1,
                 ):

        super(Multislice2DPtycho, self).__init__(panel_plot_mode=panel_plot_mode,
                                                 plot_level=plot_level)
        self.wavelength = t.tensor(wavelength)
        self.detector_geometry = copy(detector_geometry)
        self.dz = dz
        self.nz = nz
        det_geo = self.detector_geometry
        if hasattr(det_geo, 'distance'):
            det_geo['distance'] = t.tensor(det_geo['distance'])
        if hasattr(det_geo, 'basis'):
            det_geo['basis'] = t.tensor(det_geo['basis'])
        if hasattr(det_geo, 'corner'):
            det_geo['corner'] = t.tensor(det_geo['corner'])

        self.min_translation = t.tensor(min_translation)

        self.probe_basis = t.tensor(probe_basis)
        self.detector_slice = copy(detector_slice)
        self.surface_normal = t.tensor(surface_normal)

        self.saturation = saturation
        self.subpixel = subpixel
        self.exponentiate_obj = exponentiate_obj
        self.fourier_probe = fourier_probe
        self.units = units
        self.phase_only = phase_only
        self.prevent_aliasing = prevent_aliasing

        if mask is None:
            self.mask = mask
        else:
            self.mask = t.tensor(mask, dtype=t.bool)

        probe_guess = t.tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object
        if probe_guess.dim() > 3:
            self.probe_norm = 1 * t.max(t.abs(probe_guess[0]))
        else:
            self.probe_norm = 1 * t.max(t.abs(probe_guess))

        pg = probe_guess / self.probe_norm
        self.probe_real = t.nn.Parameter(pg.real)
        self.probe_imag = t.nn.Parameter(pg.imag)

        self.obj_real = t.nn.Parameter(obj_guess.real)
        self.obj_imag = t.nn.Parameter(obj_guess.imag)

        #self.probe = t.nn.Parameter(probe_guess.to(t.complex64)
        #                            / self.probe_norm)
        #self.obj = t.nn.Parameter(obj_guess.to(t.complex64))

        if background is None:
            if detector_slice is not None:
                background = 1e-6 * t.ones(
                    self.probe[0][self.detector_slice].shape, dtype=t.float32)
            else:
                background = 1e-6 * t.ones(self.probe[0].shape,
                                           dtype=t.float32)

        self.background = t.nn.Parameter(background)

        if weights is None:
            self.weights = None
        else:
            # We now need to distinguish between real-valued per-image
            # weights and complex-valued per-mode weight matrices
            if len(weights.shape) == 1:
                # This is if it's just a list of numbers
                self.weights = t.nn.Parameter(t.tensor(weights,
                                                       dtype=t.float32))
            else:
                # Now this is a matrix of weights, so it needs to be complex
                self.weights = t.nn.Parameter(t.tensor(weights,
                                                       dtype=t.complex64))

        if translation_offsets is None:
            self.translation_offsets = None
        else:
            t_o = t.tensor(translation_offsets, dtype=t.float32)
            t_o = t_o / translation_scale
            self.translation_offsets = t.nn.Parameter(t_o)

        self.translation_scale = translation_scale

        if probe_support is not None:
            self.probe_support = t.tensor(probe_support, dtype=t.bool)
        else:
            self.probe_support = None

        self.oversampling = oversampling

        spacing = np.linalg.norm(self.probe_basis, axis=0)
        shape = np.array(self.probe.shape[1:])
        if prevent_aliasing:
            shape *= 2
            spacing /= 2

        self.bandlimit = bandlimit

        self.as_prop = tools.propagators.generate_angular_spectrum_propagator(shape, spacing, self.wavelength, self.dz, self.bandlimit)


    @classmethod
    def from_dataset(cls, dataset, dz, nz, probe_convergence_semiangle, padding=0, n_modes=1, dm_rank=None, translation_scale=1, saturation=None, propagation_distance=None, scattering_mode=None, oversampling=1, auto_center=True, bandlimit=None, replicate_slice=False, subpixel=True, exponentiate_obj=True, units='um', fourier_probe=False, phase_only=False, prevent_aliasing=True, probe_support_radius=None, panel_plot_mode=False, plot_level=1):

        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # always do this on the cpu
        get_as_args = dataset.get_as_args
        dataset.get_as(device='cpu')
        (indices, translations), patterns = dataset[:]
        dataset.get_as(*get_as_args[0], **get_as_args[1])

        # Set to none to avoid issues with things outside the detector
        if auto_center:
            center = tools.image_processing.centroid(t.sum(patterns, dim=0))
        else:
            center = None

        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, probe_shape, det_slice = ewg(det_basis,
                                                  det_shape,
                                                  wavelength,
                                                  distance,
                                                  center=center,
                                                  padding=padding,
                                                  oversampling=oversampling)


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

        # Next generate the object geometry from the probe geometry and
        # the translations
        pix_translations = tools.interactions.translations_to_pixel(probe_basis, translations, surface_normal=surface_normal)
        
        obj_size, min_translation = tools.initializers.calc_object_setup(probe_shape, pix_translations, padding=100)

        if hasattr(dataset, 'background') and dataset.background is not None:
            background = t.sqrt(dataset.background)
        else:
            background = None

        # Finally, initialize the probe and  object using this information
        probe = tools.initializers.STEM_style_probe(dataset, probe_shape, det_slice, probe_convergence_semiangle, propagation_distance=propagation_distance, oversampling=oversampling)
        #probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice, propagation_distance=propagation_distance, oversampling=oversampling)

        # Now we initialize all the subdominant probe modes
        probe_max = t.max(t.abs(probe))
        if n_modes >=2:
            probe_stack = list(0.01*tools.initializers.generate_subdominant_modes(probe,n_modes-1,circular=False))
            #probe_stack = [0.01 * probe_max * t.rand(probe.shape,dtype=probe.dtype) for i in range(n_modes - 1)]
            probe = t.stack([probe,] + probe_stack)
        else:
            probe = t.unsqueeze(probe,0)

        # For a Fourier space probe
        if fourier_probe:
            probe = tools.propagators.far_field(probe)

        # Consider a different start
        if exponentiate_obj:
            obj = t.zeros(obj_size, dtype=t.complex64)
        else:
            obj = t.exp(1j*t.zeros(obj_size))
        # If we will use a separate object per slice
        if not replicate_slice:
            obj = t.stack([obj]*nz)
            
        det_geo = dataset.detector_geometry

        translation_offsets = 0 * (t.rand((len(dataset),2)) - 0.5)

        if dm_rank is not None and dm_rank != 0:
            if dm_rank > n_modes:
                raise KeyError('Density matrix rank cannot be greater than the number of modes. Use dm_rank = -1 to use a full rank matrix.')
            elif dm_rank == -1:
                # dm_rank == -1 is defined to mean full-rank
                dm_rank = n_modes
            Ws = t.zeros(len(dataset),dm_rank,n_modes,2)
            # Start with as close to the identity matrix as possible,
            # cutting of when we hit the specified maximum rank
            for i in range(0,dm_rank):
                Ws[:,i,i,0] = 1
        else:
            # dm_rank == None or dm_rank = 0 triggers a special case where
            # a standard incoherent multi-mode model is used. This is the
            # default, because it is so common.
            # In this case, we define a set of weights which only has one index
            Ws = t.ones(len(dataset))
        
        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        if probe_support_radius is not None:
            probe_support = t.zeros(probe[0].shape, dtype=t.bool)
            xs, ys = np.mgrid[:probe.shape[-2],:probe.shape[-1]]
            xs = xs - np.mean(xs)
            ys = ys - np.mean(ys)
            Rs = np.sqrt(xs**2 + ys**2)
        
            probe_support[Rs<probe_support_radius] = 1
            probe = probe * probe_support[None,:,:]

        else:
            probe_support = None
 
        probe = probe
        
        return cls(wavelength, det_geo, probe_basis, probe, obj, dz, nz,
                   detector_slice=det_slice,
                   surface_normal=surface_normal,
                   probe_support=probe_support,
                   min_translation=min_translation,
                   translation_offsets=translation_offsets,
                   weights=Ws, mask=mask, background=background,
                   translation_scale=translation_scale,
                   saturation=saturation,
                   oversampling=oversampling,
                   bandlimit=bandlimit,
                   subpixel=subpixel,
                   exponentiate_obj=exponentiate_obj,
                   units=units, fourier_probe=fourier_probe,
                   phase_only=phase_only,
                   prevent_aliasing=prevent_aliasing,
                   panel_plot_mode=panel_plot_mode,
                   plot_level=plot_level)
                   
    
    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations,
                                                             surface_normal=self.surface_normal)
        pix_trans -= self.min_translation

        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]
        
            # This restricts the basis probes to stay within the probe support
        if self.probe_support is not None:
            basis_prs = self.probe * self.probe_support[...,:,:]
        else:
            basis_prs = self.probe 

        # For a Fourier-space probe, we take an IFT
        if self.fourier_probe:
            basis_prs = tools.propagators.inverse_far_field(basis_prs)

        if self.prevent_aliasing:
            #pix_trans = pix_trans * 2
            basis_prs = image_processing.fourier_upsample(basis_prs)

        # Now we construct the probes for each shot from the basis probes
        Ws = self.weights[index]

        if len(self.weights[0].shape) == 0:
            # If a purely stable coherent illumination is defined
            # No cmult because Ws is real in this case
            prs = Ws[...,None,None,None] * basis_prs
        else:
            # If a frame-by-frame weight matrix is defined
            # This takes the dot product of all the weight matrices with
            # the probes. The output has dimensions of translation, then
            # coherent mode index, then x,y, and then complex index
            prs = t.sum(Ws[...,None,None] * basis_prs, axis=-3)
            

        if self.exponentiate_obj:
            if self.phase_only:
                obj = t.exp(1j*self.obj.real)
            else:
                obj = t.exp(1j*self.obj)
        else:
            obj = self.obj

        #if self.prevent_aliasing:
        #    obj = image_processing.fourier_upsample(obj)

        exit_waves = self.probe_norm * prs
        for i in range(self.nz):
            # If only one object slice
            if self.obj.dim() == 2:
                if i == 0 and self.subpixel:
                    # We only need to apply the subpixel shift to the first
                    # slice, because it shifts the probe
                    exit_waves = tools.interactions.ptycho_2D_sinc(
                        exit_waves, obj, pix_trans,
                        shift_probe=True, multiple_modes=True)
                else:
                    exit_waves = tools.interactions.ptycho_2D_round(
                        exit_waves, obj, pix_trans,
                        multiple_modes=True,upsample_obj=self.prevent_aliasing)
                                        
                    
            elif self.obj.dim() == 3:
                # If separate slices
                if i == 0 and self.subpixel:
                    exit_waves = tools.interactions.ptycho_2D_sinc(
                        exit_waves, obj[i], pix_trans,
                        shift_probe=True, multiple_modes=True)
                else:
                    exit_waves = tools.interactions.ptycho_2D_round(
                        exit_waves, obj[i], pix_trans,
                        multiple_modes=True, upsample_obj=self.prevent_aliasing)

            #if self.iteration_count >= 1:
            #    plt.imshow(t.abs(tools.propagators.far_field(
            #        exit_waves[0,0].detach()).cpu()))
            #    plt.colorbar()
            #    plt.show()
            if i < self.nz-1: #on all but the last iteration
                exit_waves = tools.propagators.near_field(
                    exit_waves,self.as_prop)
            #plt.imshow(t.abs(exit_waves[0,0].detach().cpu()))
            #plt.show()

        return exit_waves
    
        
    def forward_propagator(self, wavefields):
        if self.prevent_aliasing:
            left = [self.probe.shape[-2]//2,self.probe.shape[-1]//2]
            right = [self.probe.shape[-2]//2+self.probe.shape[-2],
                     self.probe.shape[-1]//2+self.probe.shape[-1]]
            
            return tools.propagators.far_field(wavefields)[...,left[0]:right[0],
                                                           left[1]:right[1]]
        else:
            return tools.propagators.far_field(wavefields)

    
    def measurement(self, wavefields):
        return tools.measurements.quadratic_background(wavefields,
                            self.background,
                            detector_slice=self.detector_slice,
                            measurement=tools.measurements.incoherent_sum,
                            saturation=self.saturation,
                            oversampling=self.oversampling)

    
    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)
        #return tools.losses.poisson_nll(real_data, sim_data, mask=mask,eps=0.5)

    
    def to(self, *args, **kwargs):
        super(Multislice2DPtycho, self).to(*args, **kwargs)
        self.wavelength = self.wavelength.to(*args,**kwargs)
        # move the detector geometry too
        det_geo = self.detector_geometry
        if hasattr(det_geo, 'distance'):
            det_geo['distance'] = det_geo['distance'].to(*args,**kwargs)
        if hasattr(det_geo, 'basis'):
            det_geo['basis'] = det_geo['basis'].to(*args,**kwargs)
        if hasattr(det_geo, 'corner'):
            det_geo['corner'] = det_geo['corner'].to(*args,**kwargs)

        if self.mask is not None:
            self.mask = self.mask.to(*args, **kwargs)
            
        self.min_translation = self.min_translation.to(*args,**kwargs)
        self.probe_basis = self.probe_basis.to(*args,**kwargs)
        self.probe_norm = self.probe_norm.to(*args,**kwargs)
        if self.probe_support is not None:
            self.probe_support = self.probe_support.to(*args,**kwargs)
                
        
        self.surface_normal = self.surface_normal.to(*args, **kwargs)
        self.as_prop = self.as_prop.to(*args, **kwargs)
        
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
                                 detector_geometry=detector_geometry,
                                 mask=mask)

    
    def corrected_translations(self,dataset):
        translations = dataset.translations.to(dtype=self.probe.real.dtype,device=self.probe.device)
        t_offset = tools.interactions.pixel_to_translations(self.probe_basis,self.translation_offsets*self.translation_scale,surface_normal=self.surface_normal)
        return translations + t_offset

    
    def get_rhos(self):
        # If this is the general unified mode model
        if self.weights.dim() >= 2:
            Ws = self.weights.detach().cpu().numpy()
            rhos_out = np.matmul(np.swapaxes(Ws,1,2), Ws.conj())
            return rhos_out
        # This is the purely incoherent case
        else:
            return np.array([np.eye(self.probe.shape[0])]*self.weights.shape[0],
                            dtype=np.complex64)

    def tidy_probes(self):
        """Tidies up the probes
        
        What we want to do here is use all the information on all the probes
        to calculate a natural basis for the experiment, and update all the
        density matrices to operate in that updated basis

        As a first step, we calculate the state of the light field across the
        full experiment, using the weight matrices and basis probes. Then, we
        use an SVD to update the basis probes so they form an eigenbasis of
        the implied density matrix for the full experiment.

        Next, the weight matrices for each shot are recalculated so that the
        probes generated by weights * basis_probes for each shot are themselves
        an eigenbasis for that individual shot's density matrix.
        """
        
        # First we treat the incoherent but stable  case, where the weights are
        # just one per-shot overall weight
        if self.weights.dim() == 1:
            probe = self.probe.detach().cpu().numpy()
            ortho_probes = analysis.orthogonalize_probes(self.probe.detach())
            self.probe.data = ortho_probes
            return

        # What follows is for the unified OPRP and incoherent multi-mode model,
        # where each shot has it's own matrix of weights such that the probe
        # state for each shot is self.weights @ self.probe
        
        # We concatenate all the weight matrices, to come up with a state
        # corresponding to the summed light field across all the exposures.
        # This state will have a large number of modes, but all built from
        # the same small number of basis modes
        all_weights = t.cat(t.unbind(self.weights.detach(), dim=0), dim=0)

        # We generate the orthogonal probes based on this full-experiment
        # representation of the light field.
        ortho_probes, reexpressed_weights = \
            analysis.orthogonalize_probes(
                self.probe.detach(),
                weight_matrix=all_weights,
                return_reexpressed_weights=True
            )

        # We just orthogonalized the incoherent sum of all the exposures
        # across the full experiment, so the output probes are normalized so
        # that their intensity matches the summed intensity across the full
        # experiment. We divide their amplitudes by the square root of the
        # number of shots so that we now have a set of probes corresponding
        # to the mean shot
        ortho_probes /= np.sqrt(self.weights.shape[0])
        reexpressed_weights *= np.sqrt(self.weights.shape[0])
        
        # We now replace the shot-to-shot weights with the versions that have
        # been re-expressed in the new basis. 
        new_weights = t.stack(t.split(reexpressed_weights,
                                      self.weights.shape[1]), dim=0)

        # And we save it back to the model
        self.probe.data = ortho_probes.to(
            device=self.probe.device, dtype=self.probe.dtype)        
        self.weights.data = new_weights.to(
            device=self.weights.device, dtype=self.weights.dtype)

        # NOTE: I used to have this part as an option, with "tidy_each_frame",
        # because it took such a long time. Now that I've rewritten it properly,
        # it's quite fast and so I removed the kwarg because there's really
        # no situation where you woudn't want to do this.

        # Now, we seek to edit the shot-to-shot weight matrices such that
        # self.weights[i] @ self.probes will be properly orthogonalized for
        # all i.

        # All we need to know about the probes is that they are orthogonalized
        # and the intensity within each probe mode
        probe_sqrt_intensities = t.linalg.norm(self.probe.data, dim=(-2,-1))

        # This does a super fast batched computation
        U, S, Vh = t.linalg.svd(self.weights.data * probe_sqrt_intensities,
                                full_matrices=False)

        # We discard the U matrix and re-multiply S & Vh
        self.weights.data = S[:,:,None] * (Vh / probe_sqrt_intensities)
        

    # Needs to be updated to allow for plotting to an existing figure
    plot_list = [
        {'title': 'Probe Fourier Space Amplitude',
         'plot_func': lambda self, fig: p.plot_amplitude(self.probe if self.fourier_probe else tools.propagators.inverse_far_field(self.probe), fig=fig)},
        {'title': 'Probe Fourier Space Phase',
         'plot_func': lambda self, fig: p.plot_phase(self.probe if self.fourier_probe else tools.propagators.inverse_far_field(self.probe), fig=fig)},
        {'title': 'Probe Real Space Amplitude',
         'plot_func': lambda self, fig: p.plot_amplitude(self.probe if not self.fourier_probe else tools.propagators.inverse_far_field(self.probe), fig=fig, basis=self.probe_basis, units=self.units)},
        {'title': 'Probe Real Space Phase',
         'plot_func': lambda self, fig: p.plot_phase(self.probe if not self.fourier_probe else tools.propagators.inverse_far_field(self.probe), fig=fig, basis=self.probe_basis, units=self.units)},
        {'title': 'Average Weight Matrix Amplitudes',
         'plot_func': lambda self, fig: p.plot_amplitude(
             np.nanmean(np.abs(self.weights.data.cpu().numpy()), axis=0),
             fig=fig),
         'condition': lambda self: len(self.weights.shape) >= 2},
        {'title': '% of Power in Top Mode',
         'plot_func': lambda self, fig, dataset: p.plot_nanomap(
             self.corrected_translations(dataset),
             100 * t.stack([
                 analysis.calc_mode_power_fractions(
                     self.probe.data,
                     weight_matrix=self.weights.data[i])[0]
                 for i in range(self.weights.shape[0])
             ], dim=0),
             fig=fig,
             units=self.units),
         'condition': lambda self: len(self.weights.shape) >= 2},
        {'title': 'Slice by Slice Real Part of T',
         'plot_func': lambda self, fig: p.plot_real(self.obj.detach().cpu(), fig=fig, basis=self.probe_basis, units=self.units, cmap='cividis'),
         'condition': lambda self: self.exponentiate_obj},
        {'title': 'Slice by Slice Imaginary Part of T',
         'plot_func': lambda self, fig: p.plot_imag(self.obj.detach().cpu(), fig=fig, basis=self.probe_basis, units=self.units),
         'condition': lambda self: self.exponentiate_obj},
        {'title': 'Integrated Real Part of T',
         'plot_func': lambda self, fig: p.plot_real(t.sum(self.obj.detach().cpu(),dim=0), fig=fig, basis=self.probe_basis, units=self.units, cmap='cividis'),
         'condition': lambda self: self.exponentiate_obj and self.obj.dim() >= 3},
        {'title': 'Integrated Imaginary Part of T',
         'plot_func': lambda self, fig: p.plot_imag(t.sum(self.obj.detach().cpu(),dim=0), fig=fig, basis=self.probe_basis, units=self.units),
         'condition': lambda self: self.exponentiate_obj and self.obj.dim() >= 3},
        {'title': 'Slice by Slice Amplitude of Object Function',
         'plot_func': lambda self, fig: p.plot_amplitude(self.obj.detach().cpu(), fig=fig, basis=self.probe_basis, units=self.units),
         'condition': lambda self: not self.exponentiate_obj},
        {'title': 'Slice by Slice Phase of Object Function',
         'plot_func': lambda self, fig: p.plot_phase(self.obj.detach().cpu(), fig=fig, basis=self.probe_basis, units=self.units, cmap='cividis'),
         'condition': lambda self: not self.exponentiate_obj},
        {'title': 'Amplitude of Stacked Object Function',
         'plot_func': lambda self, fig: p.plot_amplitude(reduce(t.mul, self.obj.detach().cpu()), fig=fig, basis=self.probe_basis, units=self.units),
         'condition': lambda self: (not self.exponentiate_obj) and self.obj.dim() >= 3},
        {'title': 'Phase of Stacked Object Function',
         'plot_func': lambda self, fig: p.plot_phase(reduce(t.mul, self.obj.detach().cpu()), fig=fig, basis=self.probe_basis, units=self.units, cmap='cividis'),
         'condition': lambda self: (not self.exponentiate_obj) and self.obj.dim() >= 3},
        {'title': 'Corrected Translations',
         'plot_func': lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig, units=self.units)},
        {'title': 'Background',
         'plot_func': lambda self, fig: plt.figure(fig.number) and plt.imshow(self.background.detach().cpu().numpy()**2)},
    ]

    
    def save_results(self, dataset):
        basis = self.probe_basis.detach().cpu().numpy()
        translations = self.corrected_translations(dataset).detach().cpu().numpy()
        if self.fourier_probe:
            probe = tools.propagators.inverse_far_field(self.probe)
        else:
            probe = self.probe
            
        probe = probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        dz = self.dz
        nz = self.nz
        prop = self.as_prop
        
        return {'basis':basis, 'translation':translations,
                'probe':probe,'obj':obj,
                'background':background,
                'weights':weights, 'dz':dz, 'nz':nz,
                'interlayer propagator': prop}
