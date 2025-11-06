""" This module contains the base CDataset class for handling CDI data

Subclasses of CDataset are required to define their own implementations
of the following functions:

* __init__
* __len__
* _load
* to
* from_cxi
* to_cxi
* inspect

"""

import torch as t
from copy import copy
import h5py
import pathlib
from cdtools.tools import data as cdtdata
from torch.utils import data as torchdata

__all__ = ['CDataset']


class CDataset(torchdata.Dataset):
    """ The base dataset class which all other datasets subclass

    Subclasses torch.utils.data.Dataset

    This base dataset class defines the functionality which should be
    common to all subclassed datasets. This includes the loading and
    storage of the metadata portions of .cxi files, as well as the tools
    needed to allow for easy mixing of data on the CPU and GPU.
    """

    def __init__(
            self,
            entry_info=None,
            sample_info=None,
            wavelength=None,
            detector_geometry=None,
            mask=None,
            qe_mask=None,
            background=None,
    ):

        """The __init__ function allows construction from python objects.

        The detector_geometry dictionary is defined to have the
        entries defined by the outputs of data.get_detector_geometry.


        Parameters
        ----------
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
        """

        # Force pass-by-value-like behavior to stop strangeness
        self.entry_info = copy(entry_info)
        self.sample_info = copy(sample_info)
        self.wavelength = wavelength
        self.detector_geometry = copy(detector_geometry)
        if mask is not None:
            self.mask = t.tensor(mask, dtype=t.bool)
        else:
            self.mask = None

        if qe_mask is not None:
            self.qe_mask = t.as_tensor(qe_mask, dtype=t.float32)
        else:
            self.qe_mask = None

        if background is not None:
            self.background = t.tensor(background, dtype=t.float32)
        else:
            self.background = None

        self.get_as(device='cpu')


    def to(self, *args, **kwargs):
        """Sends the relevant data to the given device and dtype

        This function sends the stored mask and background to the
        specified device and dtype

        Accepts the same parameters as torch.Tensor.to
        """
        # The mask should always stay a uint8, but it should switch devices
        mask_kwargs = copy(kwargs)
        try:
            mask_kwargs.pop('dtype')
        except KeyError:
            pass

        if self.mask is not None:
            self.mask = self.mask.to(*args,**mask_kwargs)
        if self.qe_mask is not None:
            self.qe_mask = self.qe_mask.to(*args,**kwargs)
        if self.background is not None:
            self.background = self.background.to(*args,**kwargs)


    def get_as(self, *args, **kwargs):
        """Sets the dataset to return data on the given device and dtype

        Oftentimes there isn't room to store an entire dataset on a GPU,
        but it is still worth running the calculation on the GPU even with
        the overhead incurred by transferring data back and forth. In that
        case, get_as can be used instead of to, to declare a set of
        device and dtype that the data should be returned as, whenever it
        is accessed through the __getitem__ function (as it would be in
        any reconstructions).

        Parameters
        ----------
        Accepts the same parameters as torch.Tensor.to
        """
        self.get_as_args = (args, kwargs)

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        # Deals with loading to appropriate device/dtype, if
        # specified via a call to get_as
        inputs, outputs = self._load(index)
        if hasattr(self, 'get_as_args'):
            outputs = outputs.to(*self.get_as_args[0],**self.get_as_args[1])
            moved_inputs = []
            for inp in inputs:
                try:
                    moved_inputs.append(inp.to(*self.get_as_args[0],**self.get_as_args[1]) )
                except:
                    moved_inputs.append(inp)
        else:
            moved_inputs = inputs
        return moved_inputs, outputs

    
    def _load(self, index):
        """ Internal function to load data
        
        In all subclasses of CDataset, a _load function should be defined.
        This function is used internally by the global __getitem__ function
        defined in the base class, which handles moving data around when
        the dataset is (for example) storing the data on the CPU but 
        getting data as GPU tensors.

        It should accept an index or slice, and return output as a tuple.
        The first item of the tuple is a tuple containing the inputs to
        the forward model for the related ptychography model. The second
        item of the tuple should be the set of diffraction patterns
        associated with the returned inputs.

        Since there is no kind of data stored in a CDataset, this
        function is defined as returing a NotImplemented Error
        """
        raise NotImplementedError()
        
            
    @classmethod
    def from_cxi(cls, cxi_file):
        """Generates a new CDataset from a .cxi file directly

        This is the most commonly used constructor for CDatasets and
        subclasses thereof. It populates the dataset using the information
        in a .cxi file. It can either take an h5py.File object directly,
        or a filename or pathlib object pointing to the file

        Parameters
        ----------
        file : str, pathlib.Path, or h5py.File
            The .cxi file to load from

        Returns
        -------
        dataset : CDataset
            The constructed dataset object
        """
        
        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with h5py.File(cxi_file,'r') as f:
                return cls.from_cxi(f)
        
        entry_info = cdtdata.get_entry_info(cxi_file)
        sample_info = cdtdata.get_sample_info(cxi_file)
        wavelength = cdtdata.get_wavelength(cxi_file)
        distance, basis, corner = cdtdata.get_detector_geometry(cxi_file)
        detector_geometry = {'distance' : distance,
                             'basis'    : basis,
                             'corner'   : corner}
        mask = cdtdata.get_mask(cxi_file)
        qe_mask = cdtdata.get_qe_mask(cxi_file)
        dark = cdtdata.get_dark(cxi_file)
        return cls(
            entry_info=entry_info,
            sample_info=sample_info,
            wavelength=wavelength,
            detector_geometry=detector_geometry,
            mask=mask,
            qe_mask=qe_mask,
            background=dark,
        )
    
    
    def to_cxi(self, cxi_file):
        """Saves out a CDataset as a .cxi file 

        This function saves all the compatible information in a CDataset
        object into a .cxi file. This is useful for saving out modified
        or simulated datasets
        
        Parameters
        ----------
        cxi_file : str, pathlib.Path, or h5py.File
            The .cxi file to write to
        """

        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with h5py.File(cxi_file,'w') as f:
                return self.to_cxi(f)

        if self.entry_info is not None:
            cdtdata.add_entry_info(cxi_file, self.entry_info)
        if self.sample_info is not None:
            cdtdata.add_sample_info(cxi_file, self.sample_info)
        if self.wavelength is not None:
            cdtdata.add_source(cxi_file, self.wavelength)
        if self.detector_geometry is not None:
            if 'corner' in self.detector_geometry:
                corner = self.detector_geometry['corner']
            else:
                corner = None
            cdtdata.add_detector(cxi_file,
                               self.detector_geometry['distance'],
                               self.detector_geometry['basis'],
                               corner = corner)
        if self.mask is not None:
            cdtdata.add_mask(cxi_file, self.mask)
        if self.qe_mask is not None:
            cdtdata.add_qe_mask(cxi_file, self.qe_mask)
        if self.background is not None:
            cdtdata.add_dark(cxi_file, self.background)
        
    def inspect(self):
        """The prototype for the inspect function

        In all subclasses of CDataset, an inspect function should be
        defined which opens a tool that shows the data in a natural
        layout for that kind of experiment. In the base class, no actual
        data is stored, so this is defined to raise a NotImplementedError
        """
        raise NotImplementedError


