import torch as t
from cdtools.models import CDIModel
from cdtools import tools
from cdtools.tools import plotting as p

__all__ = ['SimplePtycho']

class SimplePtycho(CDIModel):
    """A simple ptychography model to demonstrate the structure of a model
    """
    def __init__(
            self,
            wavelength,
            probe_basis,
            probe_guess,
            obj_guess,
            min_translation = [0,0],
    ):

        # We initialize the superclass
        super().__init__()

        # We register all the constants, like wavelength, as buffers. This
        # lets the model hook into some nice pytorch features, like using
        # model.to, and broadcasting the model state across multiple GPUs
        self.register_buffer('wavelength', t.as_tensor(wavelength))
        self.register_buffer('min_translation', t.as_tensor(min_translation))
        self.register_buffer('probe_basis', t.as_tensor(probe_basis))

        # We cast the probe and object to 64-bit complex tensors
        probe_guess = t.as_tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.as_tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object when using optimizers, like Adam, which set the stepsize
        # to a fixed maximum
        self.register_buffer('probe_norm', t.max(t.abs(probe_guess)))

        # And we store the probe and object guesses as parameters, so
        # they can get optimized by pytorch
        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)


    @classmethod
    def from_dataset(cls, dataset):

        # We get the key geometry information from the dataset
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # Then, we generate the probe geometry
        ewg = tools.initializers.exit_wave_geometry
        probe_basis =  ewg(det_basis, det_shape, wavelength, distance)

        # Next generate the object geometry from the probe geometry and
        # the translations
        (indices, translations), patterns = dataset[:]
        pix_translations = tools.interactions.translations_to_pixel(
            probe_basis,
            translations,
        )
        obj_size, min_translation = tools.initializers.calc_object_setup(
            det_shape,
            pix_translations,
        )

        # Finally, initialize the probe and object using this information
        probe = tools.initializers.SHARP_style_probe(dataset)
        obj = t.ones(obj_size).to(dtype=t.complex64)

        return cls(
            wavelength,
            probe_basis,
            probe,
            obj,
            min_translation=min_translation
        )


    def interaction(self, index, translations):
        
        # We map from real-space to pixel-space units
        pix_trans = tools.interactions.translations_to_pixel(
            self.probe_basis,
            translations)
        pix_trans -= self.min_translation
        
        # This function extracts the appropriate window from the object and
        # multiplies the object and probe functions
        return tools.interactions.ptycho_2D_round(
            self.probe_norm * self.probe,
            self.obj,
            pix_trans)


    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)

    def measurement(self, wavefields):
        return tools.measurements.intensity(wavefields)

    def loss(self, real_data, sim_data):
        return tools.losses.amplitude_mse(real_data, sim_data)


    # This lists all the plots to display on a call to model.inspect()
    plot_list = [
        {
            'title': 'Probe Amplitude',
            'plot_func': lambda self, fig:
                p.plot_amplitude(self.probe, fig, basis=self.probe_basis),
        },
        {
            'title': 'Probe Phase',
            'plot_func': lambda self, fig:
                p.plot_phase(self.probe, fig, basis=self.probe_basis)
        },
        {
            'title': 'Object Amplitude',
            'plot_func': lambda self, fig:
                p.plot_amplitude(self.obj, fig, basis=self.probe_basis)
        },
        {
            'title': 'Object Phase',
            'plot_func': lambda self, fig:
                p.plot_phase(self.obj, fig, basis=self.probe_basis)
        },
    ]
    
    def save_results(self, dataset):
        # This will save out everything needed to recreate the object
        # in the same state, but it's not the best formatted. 
        base_results = super().save_results()

        # So we also save out the main results in a more useable format
        probe_basis = self.probe_basis.detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        wavelength = self.wavelength.cpu().numpy()

        results = {
            'probe_basis': probe_basis,
            'probe': probe,
            'obj': obj,
            'wavelength': wavelength,
        }

        return {**base_results, **results}
