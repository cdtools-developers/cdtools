from torch.utils.data import Dataset as torchDataset
import torch as t
import numpy as np
import h5py
import pathlib
from PIL import Image


class DirectCDIDataset(torchDataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    @classmethod
    def from_h5(cls, h5_file, path_within_h5, axes_to_average=None,
                device=None, replace_dead_pixels=True):
        """Generates a new DirectCDIDataset from a .h5 file directly

        Parameters
        ----------
        h5_file : str, pathlib.Path, or h5py.File
            The .h5 file to load from
        path_within_h5 : str
            The path within the h5 file to load the data from
        axes_to_average : tuple, optional
            The axes to average the data over
        device : torch.device, optional
            The device to load the data onto

        Returns
        -------
        dataset : DirectCDIDataset
            The constructed dataset object
        """
        # If a bare string is passed
        if isinstance(h5_file, str) or isinstance(h5_file, pathlib.Path):
            with h5py.File(h5_file, 'r') as f:
                return cls.from_h5(f, path_within_h5, axes_to_average, device)

        data = t.tensor(h5_file[path_within_h5][:, :, :, :],
                        device=device,
                        dtype=t.float32)
        if axes_to_average is not None:
            data = t.mean(data, dim=axes_to_average)

        # Replace dead pixels with the mean of the surrounding pixels
        if replace_dead_pixels:
            dead_pixels = list(t.argwhere(data == 0))
            for n in dead_pixels:
                data[n[0], n[1]] = t.mean(t.tensor([data[n[0]+1, n[1]+1],
                                                    data[n[0]+1, n[1]],
                                                    data[n[0], n[1]+1],
                                                    data[n[0]-1, n[1]-1],
                                                    data[n[0]-1, n[1]],
                                                    data[n[0], n[1]-1]]))

        return cls(data)

    @classmethod
    def from_tif(cls, tif_file):
        """Generates a new DirectCDIDataset from a .tif file directly. It assumes
        that the .tif file is already stitched, edited and is a single pattern.

        Parameters
        ----------
        tif_file : str, pathlib.Path
            The .tif file to load from

        Returns
        -------
        dataset : DirectCDIDataset
            The constructed dataset object
        """
        data = t.tensor(np.array(Image.open(tif_file)), dtype=t.float32)
        return cls(data)

    def __len__(self):
        return 1  # Since we're only loading one pattern

    def __getitem__(self, idx):
        data = self.data
        if self.transform:
            data = self.transform(data)
        return data
