import warnings
from copy import copy, deepcopy
import pathlib

import h5py
import numpy as np
import torch as t

from cdtools.datasets import CDataset
from cdtools.datasets.random_selection import random_selection
from cdtools.tools import data as cdtdata
from cdtools.tools import plotting
from cdtools.tools import analysis

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
            with h5py.File(cxi_file, 'r') as f:
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
                # If the data is 64-bit, we need to convert it to 32-bit
                # because 64-bit floats are not supported in reconstructions
                dataset.patterns = dataset.patterns.to(dtype=t.float32)
                warnings.warn(
                    "64-bit floats are not supported and precision will not be retained in reconstructions and were converted to t.float32! "
                    "If you would like to have 64-bit support, please open an issue or submit a pull request."
                )
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
        detector is roughly equal to the total intensity of light that passed
        through the sample within each detector conjugate field of view.

        If the scan points are colinear (which causes issues for this
        estimation), the mean pattern is displayed unscaled.

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
        

    def downsample(self, factor=2):
        """Downsamples all diffraction patterns by the specified factor

        This is an easy way to shrink the amount of data you need to work with
        if the speckle size is much larger than the detector pixel size.

        The downsampling factor must be an integer. The size of the output
        patterns are reduced by the specified factor, with each output pixel
        equal to the sum of a <factor> x <factor> region of pixels in the
        input pattern. This summation is done by pytorch.functional.avg_pool2d.
        
        Any mask, quantum efficiency, and background data which is stored with
        the dataset is downsampled with the data. The background is downsampled
        using the same method as the data.

        If there is no quantum efficiency mask, then the mask is downsampled so
        that any output pixel containing a masked pixel will be masked. If there
        is a quantum efficiency mask, then the quantum efficiency mask is
        downsampled using the same method as the data, and the mask is
        downsampled to include any pixels for which there is at least one valid
        pixel.

        To avoid leakage of data from masked pixels, the data is first
        multiplied by the mask before downsampling.
        
        Parameters
        ----------
        factor : int
            Default 2, the factor to downsample by

        """
        if hasattr(self, 'mask') and self.mask is not None:
            self.patterns = t.nn.functional.avg_pool2d(
                (self.mask * self.patterns).unsqueeze(0),
                factor, divisor_override=1)[0]
        else:
            self.patterns = t.nn.functional.avg_pool2d(
                self.patterns.unsqueeze(0),
                factor, divisor_override=1)[0]
            

        # If we have a QE mask, we want to include all pixels for which at
        # least one of the input pixels was unmasked, because we can account
        # for the masked pixels through quantum efficiency
        if hasattr(self, 'qe_mask') and self.qe_mask is not None:
            self.qe_mask = t.nn.functional.avg_pool2d(
                (self.mask * self.qe_mask).unsqueeze(0).unsqueeze(0),
                factor)[0,0]
            self.mask = t.nn.functional.max_pool2d(
                self.mask.to(dtype=t.uint8).unsqueeze(0).unsqueeze(0),
                factor)[0,0].to(dtype=t.bool)
            
        # But if there is no QE mask, we need to only preserve pixels for
        # which all input pixels were unmasked
        elif hasattr(self, 'mask') and self.mask is not None:
            self.mask = t.logical_not(t.nn.functional.max_pool2d(
                (1-self.mask.to(dtype=t.uint8)).unsqueeze(0).unsqueeze(0),
                factor
            )[0,0].to(dtype=t.bool))
        
        self.detector_geometry['basis'] = \
            self.detector_geometry['basis'] * factor


        
        if hasattr(self, 'background') and self.background is not None:
            self.background = t.nn.functional.avg_pool2d(
                self.background.unsqueeze(0).unsqueeze(0),
                factor,
                divisor_override=1)[0,0]


    def remove_translations_mask(self, mask_remove):
        """Removes one or more translation positions, and their associated
        properties, from the dataset using logical indexing.

        This takes a 1D mask (boolean torch tensor) with the length
        self.translations.shape[0] (i.e., the number of individual
        translated points). Patterns, translations, and intensities
        associated with indices that are "True" will be removed.

        Parameters:
        ----------
        mask_remove : 1D torch.tensor(dtype=torch.bool)
            The boolean mask indicating which elements are to be removed from
            the dataset. True indicates that the corresponding element will be
            removed.
        """

        # Check that the mask is the right size
        if mask_remove.shape != t.Size([self.translations.shape[0]]):
            raise ValueError(
                'The mask must have the same length as the number of translations in the dataset.'
            )

        # Update patterns, translations, and intensities
        self.patterns = self.patterns[~mask_remove]
        self.translations = self.translations[~mask_remove]

        if hasattr(self, 'intensities') and self.intensities is not None:
            self.intensities = self.intensities[~mask_remove]


    def crop_translations(self, roi):
        """Shrinks the range of translation positions that are analyzed

        This deletes all diffraction patterns associated with x- and 
        y-translations that lie outside of a specified rectangular
        region of interest. In essence, this operation crops the "relative 
        displacement map" (shown in self.inspect()) down to the region of 
        interest.

        Parameters:
        ----------
        roi : tuple(float, float, float, float)
            The translation-x and -y coordinates that define the rectangular 
            region of interest as (in units of meters)
            (left, right, bottom, top). The definition of these bounds are
            based on how an image is normally displayed with matplotlib's
            imshow. The order in which these elements are defined in roi
            do not matter as long as roi[:2] and roi[2:] correspond with 
            the x and y coordinates, respectively.
        """

        # Pull out the bounds of the ROI, ensuring that left < right and 
        #   top < bottom
        x_left, x_right = sorted(roi[:2])
        y_top, y_bottom = sorted(roi[2:])

        # Create pointers to the x- and y-translation positions in 
        #   self.translations
        x = self.translations[:, 0]
        y = self.translations[:, 1]

        # Go look for all translation values that lie inside of the roi
        #   and store their indices.
        inside_roi = (x >= x_left) & (x <= x_right) & (y <= y_bottom) & (y >= y_top)

        # Throw a value error if inside_roi is empty
        if not t.any(inside_roi):
            raise ValueError('The roi does not contain any positions from the dataset '
                             '(i.e., patterns and translations will be empty).'
                             ' Please redefine the bounds of the roi.')

        # Remove translations outside the ROI
        self.remove_translations_mask(~inside_roi)
