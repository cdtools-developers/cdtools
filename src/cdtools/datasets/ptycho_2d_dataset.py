import numpy as np
import torch as t
from copy import copy
import h5py
import pathlib
from torch.nn.functional import interpolate as tinterpolate
from cdtools.datasets import CDataset
from cdtools.datasets.random_selection import random_selection
from cdtools.tools import data as cdtdata
from cdtools.tools import plotting
from matplotlib import pyplot as plt
from cdtools.tools import analysis
from copy import deepcopy

__all__ = ['Ptycho2DDataset']


class Ptycho2DDataset(CDataset):
    """The standard dataset for a 2D ptychography scan

    Subclasses datasets.CDataset

    This class loads and saves 2D ptychography scan data from .cxi files.
    It should save and load files compatible with most reconstruction
    programs, although it is only tested against SHARP.
    """
    def __init__(self, translations, patterns, intensities=None,
                 axes=None, *args, **kwargs):
        """The __init__ function allows construction from python objects.

        The detector_geometry dictionary is defined to have the
        entries defined by the outputs of data.get_detector_geometry.

        Note that the created dataset object will not copy the data in the
        patterns parameter in order to avoid doubling the memory requiement
        for large datasets.

        Parameters
        ----------
        translations : array
            An nx3 array containing the probe translations at each scan point
        patterns : array
            An nxmxl array containing the full stack of measured diffraction patterns
        axes : list(str)
            A list of names for the axes of the probe translations
        entry_info : dict
            A dictionary containing the entry_info metadata
        sample_info : dict
            A dictionary containing the sample_info metadata
        wavelength : float
            The wavelength of light used in the experiment
        detector_geometry : dict
            A dictionary containing the various detector geometry
            parameters
        mask : array
            A mask for the detector, defined as 1 for live pixels, 0
            for dead
        background : array
            An initial guess for the not-previously-subtracted
            detector background
        intensities : array
            A list of measured shot-to-shot intensities
        """
        

        super(Ptycho2DDataset,self).__init__(*args, **kwargs)
        self.axes = copy(axes)
        self.translations = t.tensor(translations)
        
        self.patterns = t.as_tensor(patterns)

        if self.mask is None:
            self.mask = t.ones(self.patterns.shape[-2:]).to(dtype=t.bool)
        self.mask.masked_fill_(t.isnan(t.sum(self.patterns,dim=(0,))),0)
        self.patterns.masked_fill_(t.isnan(self.patterns),0)

        if intensities is not None:
            self.intensities = t.as_tensor(intensities, dtype=t.float32)
        else:
            self.intensities = None
            
    def __len__(self):
        return self.patterns.shape[0]

    def _load(self, index):
        """ Internal function to load data

        This function is used internally by the global __getitem__ function
        defined in the base class, which handles moving data around when
        the dataset is (for example) storing the data on the CPU but
        getting data as GPU tensors.

        It loads data in the format (inputs, output)

        The inputs for a 2D ptychogaphy data set are:

        1) The indices of the patterns to use
        2) The recorded probe positions associated with those points

        Parameters
        ----------
        index : int or slice
             The index or indices of the scan points to use

        Returns
        -------
        inputs : tuple
             A tuple of the inputs to the related forward models
        outputs : tuple
             The output pattern or stack of output patterns
        """
        return (index, self.translations[index]), self.patterns[index]


    def to(self, *args, **kwargs):
        """Sends the relevant data to the given device and dtype

        This function sends the stored translations, patterns,
        mask and background to the specified device and dtype

        Accepts the same parameters as torch.Tensor.to
        """
        super(Ptycho2DDataset,self).to(*args,**kwargs)
        self.translations = self.translations.to(*args, **kwargs)
        self.patterns = self.patterns.to(*args, **kwargs)


    # It sucks that I can't reuse the base factory method here,
    # perhaps there is a way but I couldn't figure it out.
    @classmethod
    def from_cxi(cls, cxi_file, cut_zeros=True, load_patterns=True):
        """Generates a new Ptycho2DDataset from a .cxi file directly

        This generates a new Ptycho2DDataset from a .cxi file storing
        a 2D ptychography scan.

        Parameters
        ----------
        file : str, pathlib.Path, or h5py.File
            The .cxi file to load from
        cut_zeros : bool
            Default True, whether to set all negative data to zero

        Returns
        -------
        dataset : Ptycho2DDataset
            The constructed dataset object
        """
        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with h5py.File(cxi_file,'r') as f:
                return cls.from_cxi(f, cut_zeros=cut_zeros, load_patterns=load_patterns)

        # Generate a base dataset
        dataset = CDataset.from_cxi(cxi_file)
        # Mutate the class to this subclass (BasicPtychoDataset)
        dataset.__class__ = cls

        # Load the data that is only relevant for this class
        translations = cdtdata.get_ptycho_translations(cxi_file)
        # And now re-do the stuff from __init__
        dataset.translations = t.tensor(translations, dtype=t.float32)
        if load_patterns:
            patterns, axes = cdtdata.get_data(cxi_file, cut_zeros=cut_zeros)
            dataset.patterns = t.as_tensor(patterns)
            if dataset.patterns.dtype == t.float64:
                raise NotImplementedError('64-bit floats are not supported and precision will not be retained in reconstructions! Please explicitly convert your data to 32-bit or submit a pull request')
            
            dataset.axes = axes
            
            if dataset.mask is None:
                dataset.mask = t.ones(dataset.patterns.shape[-2:]).to(dtype=t.bool)

        try:
            intensities = cdtdata.get_shot_to_shot_info(cxi_file, 'intensities')
            dataset.intensities = t.as_tensor(intensities, dtype=t.float32)
        except KeyError:
            dataset.intensities = None
            
        return dataset


    def to_cxi(self, cxi_file):
        """Saves out a Ptycho2DDataset as a .cxi file

        This function saves all the compatible information in a
        Ptycho2DDataset object into a .cxi file. This saved .cxi file
        should be compatible with any standard .cxi file based
        reconstruction tool, such as SHARP.

        Parameters
        ----------
        cxi_file : str, pathlib.Path, or h5py.File
            The .cxi file to write to
        """

        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with cdtdata.create_cxi(cxi_file) as f:
                return self.to_cxi(f)

        super(Ptycho2DDataset,self).to_cxi(cxi_file)
        if hasattr(self, 'axes'):
            cdtdata.add_data(cxi_file, self.patterns, axes=self.axes)
        else:
            cdtdata.add_data(cxi_file, self.patterns)
        cdtdata.add_ptycho_translations(cxi_file, self.translations)

        if hasattr(self, 'intensities') and self.intensities is not None:
            cdtdata.add_shot_to_shot_info(cxi_file, self.intensities, 'intensities')


    def inspect(
            self,
            logarithmic=True,
            units='um',
            log_offset=1,
            plot_mean_pattern=True
    ):
        """Launches an interactive plot for perusing the data

        This launches an interactive plotting tool in matplotlib that
        shows the spatial map constructed from the integrated intensity
        at each position on the left, next to a panel on the right that
        can display a base-10 log plot of the detector readout at each
        position.
        """


        def get_images(idx):
            inputs, output = self[idx]
            meas_data = output.detach().cpu().numpy()
            if hasattr(self, 'mask') and self.mask is not None:
                mask = self.mask.detach().cpu().numpy()
            else:
                mask = 1
                
            if logarithmic:
                return np.log(meas_data + log_offset) / np.log(10) * mask
            else:
                return meas_data * mask

        translations = self.translations.detach().cpu().numpy()
        
        # This takes about twice as long as it would to just do it all at
        # once, but it avoids creating another self.patterns-sized array
        # as an intermediate step. This can be super important because
        # self.patterns can be more than half the available memory
        nanomap_values = np.ones(self.translations.shape[0])

        chunk_size = 10
        for i in range(0, self.translations.shape[0], chunk_size):
            nanomap_values[i:i+chunk_size] = \
                t.sum(self.mask * self.patterns[i:i+chunk_size],dim=(1,2))

        # This is the faster but more memory-intensive version
        # nanomap_values = (self.mask * self.patterns).sum(dim=(1,2)).detach().cpu().numpy()
    
        if logarithmic:
            cbar_title = f'Log Base 10 of Intensity + {log_offset}'
        else:
            cbar_title = 'Intensity'

        if plot_mean_pattern:
            self.plot_mean_pattern(log_offset=log_offset)
            
        return plotting.plot_nanomap_with_images(self.translations.detach().cpu(), get_images, values=nanomap_values, nanomap_units=units, image_title='Diffraction Pattern', image_colorbar_title=cbar_title)

    def plot_mean_pattern(self, log_offset=1):
        """Plots the mean diffraction pattern across the dataset

        The output is normalized so that the summed intensity on the
        detector is equal to the total intensity of light that passed
        through the sample within each detector conjugate field of view.

        The plot is plotted as log base 10 of the output plus log_offset.
        By default, log_offset is set equal to 1, which is a good level for
        shot-noise limited data captured in units of photons. More
        generally, log_offset should be set roughly at the background noise
        level.
        
        """
        mean_pattern, bins, ssnr = analysis.calc_spectral_info(self)
        cmap_label = f'Log Base 10 of Intensity + {log_offset}'
        title = 'Scaled mean diffraction pattern'
        return plotting.plot_real(
            t.log10(t.as_tensor(mean_pattern + log_offset)),
            cmap_label=cmap_label,
            title=title,
        )
        
        
    def split(self):
        """Splits a dataset into two pseudorandomly selected sub-datasets
        """

        # the selection is only 5,000 items long, so we repeat it to be long
        # enough for the dataset
        repeated_random_selection = (random_selection
                            * int(np.ceil(len(self) / len(random_selection))))

        repeated_random_selection = np.array(repeated_random_selection)
        # Here, I use a fixed random selection for reproducibility
        cut_random_selection =repeated_random_selection.astype(bool)[:len(self)]
        
        dataset_1 = deepcopy(self)
        dataset_1.translations = self.translations[cut_random_selection]
        dataset_1.patterns = self.patterns[cut_random_selection]
        if hasattr(self, 'intensities') and self.intensities is not None:
            dataset_1.intensities = self.intensities[cut_random_selection]
            
        dataset_2 = deepcopy(self)
        dataset_2.translations = self.translations[~cut_random_selection]
        dataset_2.patterns = self.patterns[~cut_random_selection]
        if hasattr(self, 'intensities') and self.intensities is not None:
            dataset_2.intensities = self.intensities[~cut_random_selection]

        return dataset_1, dataset_2


    def pad(self, to_pad, value=0, mask=True):
        """Pads all the diffraction patterns by a speficied amount

        This is useful for scenarios where the diffraction is strong, even
        near the edge of the detector. In this scenario, the discrete version
        of the ptychography model will alias. Padding the diffraction patterns
        to increase their size and masking off the outer region can account
        for this effect.

        If to_pad is an integer, the patterns will be padded on all sides by
        this value. If it is a tuple of length 2, then the patterns will be
        padded (left/right, top/bottom, left/right). If a tuple of length 4,
        the padding is done as (left, right, top, bottom), following the
        convention for torch.nn.functional.pad

        Any mask and background data which is stored with the dataset will be
        padded along with the diffraction patterns
        
        Parameters
        ----------
        to_pad : int or tuple(int)
            The number of pixels to pad by.
        value : float
            Optional, the fill value to pad with. Default is 0
        mask : bool
            Optional, whether to mask off the new pixels. Default is True
        """
        
        # Convert the padding to a common format
        if not hasattr(to_pad, "__len__"):
            to_pad = (to_pad,) * 4
        elif len(to_pad) == 2:
            to_pad = ((to_pad[0],) * 2) + ((to_pad[1],) * 2)

        self.patterns = t.nn.functional.pad(self.patterns, to_pad, value=value)
        if self.mask is not None:
            if mask:
                self.mask = t.nn.functional.pad(self.mask, to_pad, value=False)
            else:
                self.mask = t.nn.functional.pad(self.mask, to_pad, value=True)
        if self.background is not None:
            self.background = t.nn.functional.pad(self.background, to_pad)
        

    def downsample(self, scale):
        """ Downsamples the data by a specified factor

        Specifically, this will downsample...
            self.patterns to a [L x M/scale x N/scale] tensor
            self.mask to a [M/scale x N/scale] tensor
            self.background to a [M/scale x N/scale] tensor

        This will also increase self.detector_geometry['basis'] by a
        factor of scale

        NOTE: This method uses torch.nn.functional.interpolate. On
        the PyTorch documentation, this operation may result in
        nondeterministic gradients.

        Parameters
        ----------
        scale : int
            The factor to downsample the data by
        """

        # Grab the size of the image frames in dataset.patterns
        _, num_rows, num_cols = self.patterns.size()

        # Make sure that scale does not exceed the size of the patterns
        # No... You may not make the images smaller than 1x1 pixels..
        if scale >= num_rows or scale >= num_cols:
            raise ValueError(
                'Cannot downsample data by a factor larger than or equal to '
                'the size of the pattern frames.')

        # Change the detector_geometry basis by a factor of the
        # downsampling scale
        self.detector_geometry['basis'] *= scale

        # Downsample patterns
        # The "unsqueeze" is needed since tinterpolate is expecting an input of
        #   [mini-batch x channels x height x width]
        # Here, "mini-batch" is the translation while the unsqueeze is adding
        # the channel dimension.
        self.patterns = tinterpolate(self.patterns.unsqueeze(1),
                                     scale_factor=(1 / scale, 1 / scale),
                                     mode='bilinear',
                                     align_corners=False).squeeze(1)

        if self.mask is not None:
            # Next, downsample the mask. But first convert it to uint8
            # (can't interpolate bool). Also, the funky "[(None,)*2] does the
            # same job as "_.unsqueeze(0).unsqueeze(0)"
            temp_mask = tinterpolate(self.mask[(None,)*2].to(t.uint8),
                                     scale_factor=(1 / scale, 1 / scale),
                                     mode='bilinear',
                                     align_corners=False).squeeze()
            # Make the mask boolean again. The mask will be dilated such that
            # any live pixels surrounding dead pixels will be set as dead
            self.mask = temp_mask == 1

        if self.background is not None:
            # Downsample the background.
            self.background = tinterpolate(self.background[(None,) * 2],
                                           scale_factor=(1 / scale, 1 / scale),
                                           mode='bilinear',
                                           align_corners=False).squeeze()