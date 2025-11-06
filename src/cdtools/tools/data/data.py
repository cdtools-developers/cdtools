"""Contains the base functions for loading and saving data from/to .cxi files

These functions are used when constructing a new dataset class to pull
specific desired information from a .cxi file. These functions should
handle all the needed conversions between standard formats (for example,
transposes of the basis arrays, shifting from object to probe motion, etc).
"""

import h5py
import numpy as np
import numbers
import datetime
import dateutil.parser
import torch as t
from contextlib import contextmanager
import numbers
from collections.abc import Mapping
import pathlib

__all__ = ['get_entry_info',
           'get_sample_info',
           'get_wavelength',
           'get_detector_geometry',
           'get_mask',
           'get_qe_mask',
           'get_dark',
           'get_data',
           'get_shot_to_shot_info',
           'get_ptycho_translations',
           'create_cxi',
           'add_entry_info',
           'add_sample_info',
           'add_source',
           'add_detector',
           'add_mask',
           'add_qe_mask',
           'add_dark',
           'add_data',
           'add_shot_to_shot_info',
           'add_ptycho_translations',
           'nested_dict_to_numpy',
           'nested_dict_to_torch',
           'nested_dict_to_h5',
           'h5_to_nested_dict',
           ]


#
# Functions to inspect the basic attributes of a cxi file represented as an
# h5 file object
#


def get_entry_info(cxi_file):
    """Returns a dictionary with the basic metadata from the cxi file's entry_1 attribute

    String type metadata is read out as a string, and datetime metadata
    is converted to python datetime objects if the string is properly
    formatted.

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read

    Returns
    -------
    entry_info : dict
        A dictionary with basic metadata defined in the cxi file

    """
    e1 = cxi_file['entry_1']
    metadata_attrs = ['title',
                      'experiment_identifier',
                      'experiment_description',
                      'program_name',
                      'start_time',
                      'end_time']
    metadata = {attr: str(e1[attr][()].decode()) for attr in metadata_attrs
                if attr in e1}
    

    for attr in metadata_attrs:
        if attr in e1:
            try:
                metadata[attr] = dateutil.parser.parse(str(e1[attr][()].decode()))
            except ValueError:
                metadata[attr] = str(e1[attr][()].decode())

    array_attrs = ['polarization_states']
    for attr in array_attrs:
        if attr in e1:
            metadata[attr] = np.array(e1[attr])

    return metadata


def get_sample_info(cxi_file):
    """Returns a dictionary with the basic metadata from the cxi file's entry_1/sample_1 attribute

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read

    Returns
    -------
    sample_info : dict
        A dictionary with basic metadata from the sample defined in the cxi file

    """
    if 'entry_1/sample_1' not in cxi_file:
        return None
    
    s1 = cxi_file['entry_1/sample_1']
    metadata_attrs = ['name','description','unit_cell_group']

    metadata = {}
    for attr in metadata_attrs:
        # Somehow different ways of saving can lead to different ways to
        # decode it here, so we try both
        if attr in s1:
            try:
                metadata[attr] = str(s1[attr][()].decode())
            except AttributeError as e:
                metadata[attr] = str(np.array(s1[attr][:])[0].decode())

    float_attrs = ['concentration',
                   'mass',
                   'temperature',
                   'thickness',
                   'unit_cell_volume']

    for attr in float_attrs:
        if attr in s1:
            metadata[attr] = np.float32(s1[attr][()])

    if 'unit_cell' in s1:
        metadata['unit_cell'] = s1['unit_cell'][()].astype(np.float32)


    if 'geometry_1/orientation' in s1:
        orient = s1['geometry_1/orientation'][()].astype(np.float32)
        xvec = orient[:3] / np.linalg.norm(orient[:3])
        yvec = orient[3:] / np.linalg.norm(orient[3:])
        metadata['orientation'] = np.array([xvec,yvec,
                                            np.cross(xvec,yvec)])

    if 'geometry_1/surface_normal' in s1:
        snorm = s1['geometry_1/surface_normal'][()].astype(np.float32)
        xvec = np.cross(np.array([0.,1.,0.]), snorm)
        xvec /= np.linalg.norm(xvec)
        yvec = np.cross(snorm, xvec)
        yvec /= np.linalg.norm(yvec)
        metadata['orientation'] = np.array([xvec, yvec, snorm])

    # Check if the metadata is empty
    if metadata == {}:
        metadata = None

    return metadata


def get_wavelength(cxi_file):
    """Returns the wavelength of the source defined in the cxi file object, in m

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read

    Returns
    -------
    wavelength: np.float32
        The wavelength of the source defined in the cxi file
    """
    i1 = cxi_file['entry_1/instrument_1']
    if 'source_1/wavelength' in i1:
        wavelength = np.float32(i1['source_1/wavelength'])
    elif 'source_1/energy' in i1:
        energy = np.float32(i1['source_1/energy'])
        wavelength = 1.9864459e-25 / energy
    else:
        raise KeyError('Neither Wavelength or Energy Defined in provided .cxi File')

    return wavelength


def get_detector_geometry(cxi_file):
    """Returns a standardized description of the detector geometry defined in the cxi file object

    It makes intelligent assumptions based on the definitions in the cxi
    file definition. The standardized description of the geometry that it
    outputs includes the sample to detector distance, the corner location
    of the detector, and the basis vectors defining the detector. It can
    only handle detectors defined as rectangular grids of pixels.

    The distance and corner_location values are technically overdetermining
    the detector location, but for many experiments (particularly
    transmission experiments), the distance is needed and the exact
    corner location is not. If the corner location is not reported in
    the cxi file, no attempt will be made to calculate it.

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read

    Returns
    -------
    distance : np.float32
        The sample to detector distance, in m
    basis_vectors : np.array
        The basis vectors for the detector
    corner_location : np.array
        The real-space location of the (0,0) pixel in the detector

    """
    i1 = cxi_file['entry_1/instrument_1']
    d1 = i1['detector_1']

    if 'detector_1/basis_vectors' in i1:
        basis_vectors = d1['basis_vectors'][()]
        if basis_vectors.shape == (2,3):
            basis_vectors = basis_vectors.T
    else:
        # This whole thing just to account for all the ways people can
        # implicitly define the x or y pixel size for a detector. I've
        # seen too many of these in the wild, unfortunately...
        try:
            x_pixel_size = np.float32(d1['x_pixel_size'])
        except:
            x_pixel_size = None
        try:
            y_pixel_size = np.float32(d1['y_pixel_size'])
        except:
            y_pixel_size = None

        if x_pixel_size is None and y_pixel_size is not None:
            x_pixel_size = y_pixel_size
        elif x_pixel_size is not None and y_pixel_size is None:
            y_pixel_size = x_pixel_size
        if x_pixel_size is None and y_pixel_size is None:
            raise KeyError('Detector pixel size not defined in file.')
        basis_vectors = np.array([[0,-y_pixel_size,0],
                                  [-x_pixel_size,0,0]]).transpose()

    try:
        distance = np.float32(d1['distance'][()])
    except:
        distance = None
    try:
        corner_position = d1['corner_position'][()]
    except:
        corner_position = None

    # Don't pretend to calculate corner position from distance if it's not
    # defined, but do calculate distance from corner position if distance is
    # not defined. If neither is defined, then raise an error.
    if distance is None and corner_position is not None:
        detector_normal = np.cross(basis_vectors[:,0],
                                   basis_vectors[:,1])
        detector_normal /= np.linalg.norm(detector_normal)
        distance = np.linalg.norm(np.dot(corner_position, detector_normal))

    if distance is None and corner_position is None:
        raise KeyError('Neither sample to detector distance nor corner position is defined in file.')

    return distance, basis_vectors, corner_position


def get_mask(cxi_file):
    """Returns the detector mask defined in the cxi file object

    This function converts from the format specified in the cxi file
    definition to a simple on/off mask, where a value of 1 defines a
    good pixel (on) and a value of 0 defines a bad pixel (off).

    If any bit is set in the mask at all, it will be defined as a bad
    pixel, with the exception of pixels marked exactly as 0x00001000,
    which is defined to mean that the pixel has signal above the
    background. These pixels are treated as on pixels.

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read

    Returns
    -------
    mask : np.array
        An array storing the mask from the cxi file
    """

    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1/mask' in i1:
        mask = i1['detector_1/mask'][()].astype(np.uint32)
        mask_on = np.equal(mask,np.uint32(0))
        mask_has_signal = np.equal(mask,np.uint32(0x00001000))
        return np.logical_or(mask_on,mask_has_signal).astype(bool)
    else:
        return None


def get_qe_mask(cxi_file):
    """Returns the quantum efficiency mask defined in the cxi file object

    There is no way to store a quantum efficiency mask (a.k.a. a flat-field
    image) in the .cxi file specification, but experience has indicated that
    this is often a valuable thing to store, because just correcting for a
    flatfield with e.g. a division will mess up the photon counting statistics.

    Because there is no specification, I have simply chosen to store the
    quantum efficiency mask as a float32 array in the same location as the
    mask is, i.e. `entry_1/instrument_1/detector_1/qe_mask`.
    
    The stored quantum efficiency mask should be defined as the mask that
    a simulated intensity pattern needs to be multiplied by to realize the
    measured image. In other words, it should be a flat-field image, not the
    inverse of a flat-field image.
    
    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read

    Returns
    -------
    qe_mask : np.array
        A float32 array storing the quantum efficiency mask from the cxi file
    """

    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1/qe_mask' in i1:
        qe_mask = i1['detector_1/qe_mask'][()].astype(np.float32)
        return qe_mask
    else:
        return None


def get_dark(cxi_file):
    """Returns an array with a dark image to use for initialization of a background model

    This looks for a set of dark images at
    entry_1/instrument_1/detector_1/data_dark. If the darks exist, it will
    return the mean of the array along all axes but the last two. That is,
    if the dark image is a single image, it will return that image. If it
    is a stack of images, it will return the mean along the stack axis.

    If the darks do not exist, it will return None.

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read

    Returns
    -------
    dark : np.array
        An array storing the dark image
    """

    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1/data_dark' in i1:
        darks = i1['detector_1/data_dark'][()]
        dims = tuple(range(len(darks.shape) - 2))
        darks = np.nanmean(darks,axis=dims)
    else:
        darks = None

    return darks


def get_data(cxi_file, cut_zeros = True):
    """Returns an array with the full stack of detector data defined in the cxi file object

    This function will make sure to check all the various places that it's
    okay to store the data in, to ensure that it can find the data regardless
    of whether the creator of the .cxi file has remembered to link the data
    to all the required locations.

    It will return the data array in whatever shape it's defined in.

    It will also read out the axes attribute of the data into a list of strings.

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read
    cut_zeros : bool
            Default True, whether to set all negative data to zero

    Returns
    -------
    data : np.array
        An array storing the data defined in the cxi file
    axes : list(str)
        A list of the axes defined in the axes attribute, if any
    """

    # Possible locations for the data
    if 'entry_1/data_1/data' in cxi_file:
        pull_from = 'entry_1/data_1/data'
    elif 'entry_1/instrument_1/detector_1/data' in cxi_file:
        pull_from = 'entry_1/instrument_1/detector_1/data'
    else:
        raise KeyError('Data is not defined within cxi file')

    data = cxi_file[pull_from][:]
    # Use maximum in-place to avoid allocating any more memory than is needed
    if cut_zeros:
        np.maximum(data,0,data)

    if 'axes' in cxi_file[pull_from].attrs:
        try:
            axes = str(cxi_file[pull_from].attrs['axes'].decode()).split(':')
        except AttributeError as e: # Weird string vs bytes thing, ehhh
            axes = str(cxi_file[pull_from].attrs['axes']).split(':')

        axes = [axis.strip().lower() for axis in axes]
    else:
        axes = None

    return data, axes


def get_shot_to_shot_info(cxi_file, field_name):
    """Gets a specified dataset of shot-to-shot information from the cxi file

    The data is assumed to be in the form of an array, with one dimension
    being the number of patterns being stored in the dataset. This is
    helpful for storing additional readback data on the shot-to-shot
    level that may be important but doens't have a clearly defined
    place to be stored in the .cxi file specification. Such data includes
    shot-to-shot probe intensity measurements, polarizer positions, etc.

    It will look for this data in 3 places (in the following order):

    1) entry_1/data_1/<field_name>
    2) entry_1/sample_1/geometry_1/<field_name>
    3) entry_1/instrument_1/detector_1/<field_name>

    This function is also used internally to read out the translations
    associated with a ptychography experiment

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read
    field_name : str
        The name of the field to be read from

    Returns
    -------
    data : np.array
        An array storing the translations defined in the cxi file
    """
    if 'entry_1/data_1/' + field_name in cxi_file:
        pull_from = 'entry_1/data_1/' + field_name
    elif 'entry_1/sample_1/geometry_1/' + field_name in cxi_file:
        pull_from = 'entry_1/sample_1/geometry_1/' + field_name
    elif 'entry_1/instrument_1/detector_1/' in cxi_file:
        pull_from = 'entry_1/instrument_1/detector_1/' + field_name
    else:
        raise KeyError('Data is not defined within cxi file')

    return cxi_file[pull_from][()].astype(np.float32)


def get_ptycho_translations(cxi_file):
    """Gets an array of x,y,z translations, if such an array has been defined in the file

    It negates the translations, because the CXI file format is designed
    to specify translations of the samples and the cdtools code specifies
    translations of the optics.

    Parameters
    ----------
    cxi_file : h5py.File
        A file object to be read

    Returns
    -------
    translations : np.array
        An array storing the translations defined in the cxi file
    """

    translations = get_shot_to_shot_info(cxi_file, 'translation')
    return -translations


#
# Now we move on to the helper functions for writing CXI files
#


def create_cxi(filename):
    """Creates a new cxi file with a single entry group

    Parameters
    ----------
    filename : str
        The path at which to create the file
    """
    file_obj = h5py.File(filename,'w')
    file_obj.create_dataset('cxi_version', data=160)
    file_obj.create_dataset('number_of_entries',data=1)
    e1f = file_obj.create_group('entry_1')
    return file_obj


def add_entry_info(cxi_file, metadata):
    """Adds a dictionary of entry metadata to the entry_1 group of a cxi file object

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the info to
    metadata : dict
        A dictionary containing all the metadata to be stored
    """
    # Just the string and datetime types should be relevant but all are
    # included in case the cxi spec becomes more permissive
    for key, value in metadata.items():
        if isinstance(value,(str,bytes)):
            cxi_file['entry_1'][key] = np.bytes_(value)
        elif isinstance(value, datetime.datetime):
            cxi_file['entry_1'][key] = np.bytes_(value.isoformat())
        elif isinstance(value, numbers.Number):
            cxi_file['entry_1'][key] = value
        elif isinstance(value, (np.ndarray,list,tuple)):
            cxi_file['entry_1'].create_dataset(key, data=np.asarray(value))
        elif isinstance(value, t.Tensor):
            asnumpy = value.detach().cpu().numpy()
            cxi_file['entry_1'].create_dataset(key, data=asnumpy)



def add_sample_info(cxi_file, metadata):
    """Adds a dictionary of entry metadata to the entry_1/sample_1 group of a cxi file object

    This function will create the sample_1 attribute if it doesn't already exist

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the info to
    metadata : dict
        A dictionary containing all the metadata to be stored
    """
    if 'entry_1/sample_1' not in cxi_file:
        cxi_file['entry_1'].create_group('sample_1')
    s1 = cxi_file['entry_1/sample_1']

    if 'orientation' in metadata:
        if 'geometry_1' not in s1:
            s1.create_group('geometry_1')
        # Only store the part of this matrix as defined in the CXI file spec
        s1['geometry_1'].create_dataset('orientation',
                                        data=metadata['orientation'].ravel()[:6])

    for key, value in metadata.items():
        if key == 'orientation':
            continue # this is a special case
        if isinstance(value,(str,bytes)):
            s1[key] = np.bytes_(value)
        elif isinstance(value, datetime.datetime):
            s1[key] = np.bytes_(value.isoformat())
        elif isinstance(value, numbers.Number):
            s1[key] = value
        elif isinstance(value, (np.ndarray,list,tuple)):
            s1.create_dataset(key, data=np.asarray(value))
        elif isinstance(value, t.Tensor):
            asnumpy = value.detach().cpu().numpy()
            s1.create_dataset(key, data=asnumpy)


def add_source(cxi_file, wavelength):
    """Adds the entry_1/source_1 group to a cxi file object

    It stores the energy and wavelength attributes in the source_1 group,
    given a wavelength to define them from.

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the source to
    wavelength : float
        The wavelength of light
    """
    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'source_1' not in i1:
        i1.create_group('source_1')
    s1 = i1['source_1']
    s1['wavelength'] = np.float32(wavelength)
    s1['energy'] = np.float32(1.9864459e-25 / wavelength)



def add_detector(cxi_file, distance, basis, corner=None):
    """Adds the entry_1/instrument_1/detector_1 group to a cxi file object

    It will define all the relevant parameters - distance, pixel size,
    detector basis, and corner position (if relevant) based on the provided
    information.

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the detector to
    distance : float
        The sample to detector distance
    basis : array
        The detector basis
    corner : array
        Optional, the corner position of the detector

    """
    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    d1 = i1['detector_1']

    d1['distance'] = np.float32(distance)

    if isinstance(basis, t.Tensor):
        basis = basis.detach().cpu().numpy()
    if basis.shape == (2,3):
        basis = basis.T
    d1['x_pixel_size'] = np.linalg.norm(basis[:,1])
    d1['y_pixel_size'] = np.linalg.norm(basis[:,0])
    d1.create_dataset('basis_vectors', data=basis)

    if corner is not None:
        if isinstance(corner, t.Tensor):
            corner = corner.detach().cpu().numpy()
        d1.create_dataset('corner_position',data=corner)


def add_mask(cxi_file, mask):
    """Adds the specified mask to the cxi file

    It places the mask into the mask dataset under
    entry_1/instrument_1/detector_1. The internal mask is defined
    simply as a 1 for an "on" pixel and a 0 for an "off" pixel, and
    the saved mask is exactly the opposite. This is simpler than the
    most general mask allowed by the cxi file format but it captures the
    distinction between pixels to be used and pixels not to be used.

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the mask to
    mask : array
        The mask to save out to the file
    """

    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    d1 = i1['detector_1']
    if isinstance(mask, t.Tensor):
        mask = mask.detach().cpu().numpy()

    mask_to_save = np.zeros(mask.shape).astype(np.uint32)
    mask_to_save[mask == 0] = 1
    d1.create_dataset('mask',data=mask_to_save)


def add_qe_mask(cxi_file, qe_mask):
    """Adds the specified quantum efficiency mask to the cxi file

    There is no way to store a quantum efficiency mask (a.k.a. a flat-field
    image) in the .cxi file specification, but experience has indicated that
    this is often a valuable thing to store, because just correcting for a
    flatfield with e.g. a division will mess up the photon counting statistics.

    Because there is no specification, I have simply chosen to store the
    quantum efficiency mask as an array in the same location as the
    mask is, i.e. `entry_1/instrument_1/detector_1/qe_mask`.
    
    The stored quantum efficiency mask should be defined as the mask that
    a simulated intensity pattern needs to be multiplied by to realize the
    measured image. In other words, it should be a flat-field image, not the
    inverse of a flat-field image.

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the mask to
    qe_mask : array
        The quantum efficiency mask to save out to the file
    """

    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    d1 = i1['detector_1']
    if isinstance(qe_mask, t.Tensor):
        qe_mask = qe_mask.detach().cpu().numpy()

    d1.create_dataset('qe_mask',data=qe_mask)


def add_dark(cxi_file, dark):
    """Adds the specified dark image to a cxi file

    It places the dark image data into the data_dark dataset under
    entry_1/instrument_1/detector_1.

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the mask to
    dark : array
        The dark image(s) to save out to the file
    """
    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    d1 = i1['detector_1']
    if isinstance(dark, t.Tensor):
        dark = dark.detach().cpu().numpy()

    d1.create_dataset('data_dark',data=dark)


def add_data(cxi_file, data, axes=None, compression='gzip',
             chunks=True):
    """Adds the specified data to the cxi file

    It will add the data unchanged to the file, placing it in two spots:

    1) The entry_1/instrument_1/detector_1/data path
    2) A softlink at entry_1/data_1/data

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the data to
    data : array
        The data to be saved
    axes : list(str)
        Optional, a list of axis names to be saved in the axes attribute
    """
    if 'entry_1/data_1' not in cxi_file:
        cxi_file['entry_1'].create_group('data_1')
    data1 = cxi_file['entry_1/data_1']

    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    det1 = i1['detector_1']

    if isinstance(data, t.Tensor):
        data = data.detach().cpu().numpy()

    det1.create_dataset('data', data=data, compression=compression,
                        chunks=chunks)
    data1['data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

    if axes is not None:
        if isinstance(axes, list):
            axes_str = ':'.join(axes)
        else:
            axes_str = str(axes)
        det1['data'].attrs['axes'] = np.bytes_(axes_str)

        
def add_shot_to_shot_info(cxi_file, data, field_name):
    """Adds a specified dataset of shot-to-shot information to the cxi file

    The data is assumed to be in the form of an array, with one dimension
    being the number of patterns being stored in the dataset. This is
    helpful for storing additional readback data on the shot-to-shot
    level that may be important but doens't have a clearly defined
    place to be stored in the .cxi file specification. Such data includes
    shot-to-shot probe intensity measurements, polarizer positions, etc.

    This function is also used internally to store the translations
    associated with a ptychography experiment
    
    It will store this data in 3 places:

    1) The entry_1/sample_1/geometry_1/<field_name> path
    2) A softlink at entry_1/data_1/<field_name>
    3) A softlink at entry_1/instrument_1/detector_1/<field_name>

    The geometry and detector paths may not always be relevant, but this
    ensures that the data is always available in any of the places that
    an eventual reader may go to look for, e.g., the translations.

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the translations to
    data : array
        The data to be saved
    field_name : str
        The field name to save the data under
    """
    
    if 'entry_1/sample_1' not in cxi_file:
        cxi_file['entry_1'].create_group('sample_1')
    s1 = cxi_file['entry_1/sample_1']

    if 'geometry_1' not in s1:
        s1.create_group('geometry_1')
    g1 = s1['geometry_1']

    if 'entry_1/data_1' not in cxi_file:
        cxi_file['entry_1'].create_group('data_1')
    data1 = cxi_file['entry_1/data_1']

    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    det1 = i1['detector_1']


    if isinstance(data, t.Tensor):
        data = data.detach().cpu().numpy()


    g1.create_dataset(field_name, data=data)
    data1[field_name] = h5py.SoftLink('/entry_1/sample_1/geometry_1/'
                                      + field_name)
    det1[field_name] = h5py.SoftLink('/entry_1/sample_1/geometry_1/'
                                      + field_name)


def add_ptycho_translations(cxi_file, translations):
    """Adds the specified translations to the cxi file

    It will add the translations to the file, negating them to conform to
    the standard in cxi files that the translations refer to the object's
    translation.

    It will store them in 3 places:

    1) The entry_1/sample_1/geometry_1/translation path
    2) A softlink at entry_1/data_1/translation
    3) A softlink at entry_1/instrument_1/detector_1/translation

    Parameters
    ----------
    cxi_file : h5py.File
        The file to add the translations to
    translations : array
        The translations to be saved
    """
    # accounting for the different definition between cxi files and
    # cdtools
    translations = -translations

    add_shot_to_shot_info(cxi_file, translations, 'translation')


def nested_dict_to_h5(h5_file, d):
    """Saves a nested dictionary to an h5 file object

    Parameters
    ----------
    h5_file : h5py.File
        A file object, or path to a file, to write the dictionary to
    d : dict
        A mapping whose keys are all strings and whose values are only numpy arrays, pytorch tensors, scalars, or other mappings meeting the same conditions
    """
    
    # If a bare string is passed
    if isinstance(h5_file, str) or isinstance(h5_file, pathlib.Path):
        with h5py.File(h5_file,'w') as f:
            return nested_dict_to_h5(f, d)
    
    for key in d.keys():
        value = d[key]
        if isinstance(value, (numbers.Number, np.bool_)):
            arr = np.array(value)
            h5_file.create_dataset(key, data=arr)
        elif isinstance(value, np.ndarray):
            h5_file.create_dataset(key, data=value)
        elif t.is_tensor(value):
            h5_file.create_dataset(key, data=value.detach().cpu().numpy())
        elif isinstance(value, str):
            h5_file.create_dataset(key, data=value, dtype=h5py.string_dtype())
        elif isinstance(value, Mapping):
            group = h5_file.create_group(key)
            nested_dict_to_h5(group, value)
        else:
            raise ValueError(f'{value} is not a number, numpy array or mapping')

    return


def h5_to_nested_dict(h5_file):
    """Loads a nested dictionary from an h5 file object

    Parameters
    ----------
    h5_file : h5py.File
        A file object, or path to a file, to load from

    Returns
    -------
    d : dict
        A dictionary whose keys are all strings and whose values are numpy arrays, scalars, or python strings. Will raise an error if the data cannot be loaded into this format
    """
    
    # If a bare string is passed
    if isinstance(h5_file, str) or isinstance(h5_file, pathlib.Path):
        with h5py.File(h5_file,'r') as f:
            return h5_to_nested_dict(f)

    d = {}
    for key in h5_file.keys():
        value = h5_file[key]
        if isinstance(value, h5py.Dataset):
            arr = value[()]
            # Strings stored via nested_dict_to_h5 will wind up as bytes objs
            if type(arr) == type(b''):
                d[key] = arr.decode('utf-8')
            # Some strings in h5 files seem to be stored this way
            elif hasattr(arr, 'dtype') and arr.dtype == object:
                d[key] = arr.ravel()[0].decode('utf-8')
            # This is the default case: it's an array of numbers
            else:
                d[key] = arr
            
        elif isinstance(value, h5py.Group):
            sub_d = h5_to_nested_dict(value)
            d[key] = sub_d            
        else:
            raise ValueError(f'{value} could not be interpreted sensibly')

    return d


def nested_dict_to_numpy(d):
    """Sends all array like objects in a nested dict to numpy arrays

    Parameters
    ----------
    d : dict
        A mapping whose keys are all strings and whose values are only numpy arrays, pytorch tensors, scalars, or other mappings meeting the same conditions

    Returns
    -------
    new_dict : dict
        A new dictionary with all array like objects sent to numpy 
    """
    
    new_dict = {}
    for key in d.keys():
        value = d[key]
        # bools are an instance of number, but not np.bool_...
        if isinstance(value, (numbers.Number, np.bool_, np.ndarray)):
            new_dict[key] = value
        elif t.is_tensor(value):
            new_dict[key] = value.cpu().numpy()
        elif isinstance(value, str):
            new_dict[key] = value
        elif isinstance(value, Mapping):
            new_dict[key] = nested_dict_to_numpy(value)
        else:
            raise ValueError(f'{value} is not a number, numpy array, torch tensor, string, or mapping')

    return new_dict

def nested_dict_to_torch(d, device=None):
    """Sends all array like objects in a nested dict to pytorch tensors

    This will also send all the tensors to a specific device, if specified.
    There is no option to send all tensors to a specific dtype, as tensors
    are often a mixture of integer, floating point, and complex types. In
    the future, this may support a "precision" option to send all tensors to
    a specified precision.
    
    Parameters
    ----------
    d : dict
        A mapping whose keys are all strings and whose values are only numpy arrays, pytorch tensors, scalars, or other mappings meeting the same conditions
    device : torch.device
        A valid device argument for torch.Tensor.to
    
    Returns
    -------
    new_dict : dict
        A new dictionary with all array like objects sent to torch tensors 
    """

    new_dict = {}
    for key in d.keys():
        value = d[key]
        # bools are an instance of number, but not np.bool_...
        if isinstance(value, (numbers.Number, np.bool_, np.ndarray)):
            new_dict[key] = t.as_tensor(value, device=device)
        elif t.is_tensor(value):
            new_dict[key] = value.to(device=device)
        elif isinstance(value, str):
            new_dict[key] = value
        elif isinstance(value, Mapping):
            new_dict[key] = nested_dict_to_torch(value, device=device)
        else:
            raise ValueError(f'{value} is not a number, numpy array, torch tensor, string, or mapping')

    return new_dict
