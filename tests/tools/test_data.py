from cdtools.tools import data
import numpy as np
import torch as t
import h5py
import pytest
import os
import datetime
import numbers
import pathlib



#
# We start with a bunch of tests of the data loading capabilities
#



def test_get_entry_info(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        entry_info = data.get_entry_info(cxi)
        for key in expected['entry metadata']:
            assert entry_info[key] == expected['entry metadata'][key]

        
def test_get_sample_info(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        sample_info = data.get_sample_info(cxi)
        if sample_info is None and  \
           ('sample info' not in expected or
            expected['sample info'] is None):
            # Valid if no sample info is defined at all
            continue
        for key in expected['sample info']:
            if isinstance(expected['sample info'][key],np.ndarray):
                assert np.allclose(sample_info[key],
                                   expected['sample info'][key])
            else:
                assert sample_info[key] == expected['sample info'][key]
                
    
def test_get_wavelength(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        assert np.isclose(expected['wavelength'],data.get_wavelength(cxi))


def test_get_detector_geometry(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        distance, basis, corner = data.get_detector_geometry(cxi)
        assert np.isclose(distance,expected['detector']['distance'])
        assert np.allclose(basis,expected['detector']['basis'])
        if isinstance(expected['detector']['corner'], np.ndarray):
            assert np.allclose(corner, expected['detector']['corner'])
        else:
            assert corner == expected['detector']['corner']

            
def test_get_mask(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        mask = data.get_mask(cxi)
        if expected['mask'] is None and mask is None:
            continue
        assert np.all(data.get_mask(cxi) == expected['mask'])


def test_get_dark(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        dark = data.get_dark(cxi)
        if dark is None:
            assert expected['dark'] is None
        else:
            assert np.allclose(dark, expected['dark'])
            
        
def test_get_data(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        patterns, axes = data.get_data(cxi)
        assert np.allclose(patterns, expected['data'])
        assert axes == expected['axes']


def test_get_shot_to_shot_info(polarized_ptycho_cxi):
    cxi, expected = polarized_ptycho_cxi
    for key in ('analyzer_angle', 'polarizer_angle'):
        assert np.allclose(data.get_shot_to_shot_info(cxi, key),
                           expected[key])


def test_get_ptycho_translations(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        assert np.allclose(data.get_ptycho_translations(cxi),
                           expected['translations'])



#
# Then, write a test for the data saving. It should create a .cxi file
# using the data seving tools, and then check that when read with the
# .cxi reading tools that it gets the same things that were written.
#


def test_create_cxi(tmp_path):
    data.create_cxi(tmp_path / 'test_create.cxi')
    with h5py.File(tmp_path / 'test_create.cxi','r') as f:
        assert f['cxi_version'][()] == 160
        assert 'entry_1' in f

        
def test_add_entry_info(tmp_path):
    entry_info = {'experiment_identifier':'test of cxi file writing tools',
                  'title': 'my cool experiment',
                  'start_time': datetime.datetime.now(),
                  'end_time': datetime.datetime.now()}

    with data.create_cxi(tmp_path / 'test_add_entry_info.cxi') as f:
        data.add_entry_info(f, entry_info)


    with h5py.File(tmp_path / 'test_add_entry_info.cxi','r') as f:
        read_entry_info = data.get_entry_info(f)

    print(read_entry_info)
        
    for key in entry_info:
        if isinstance(entry_info[key], np.ndarray):
            assert np.allclose(entry_info[key], read_entry_info[key])
        else:
            assert entry_info[key] == read_entry_info[key]


def test_add_sample_info(tmp_path):
    sample_info = {'name':'A nice fake sample',
                   'concentration': 10,
                   'mass': 5.3,
                   'temperature': 76,
                   'description': 'A very nice sample',
                   'unit_cell': np.array([1,1,1,90.,90.,90.])}

    with data.create_cxi(tmp_path / 'test_add_sample_info.cxi') as f:
        data.add_sample_info(f, sample_info)

    with h5py.File(tmp_path / 'test_add_sample_info.cxi','r') as f:
        read_sample_info = data.get_sample_info(f)    

    for key in sample_info:
        if isinstance(sample_info[key], np.ndarray):
            assert np.allclose(sample_info[key], read_sample_info[key])
        elif isinstance(sample_info[key], numbers.Number):
            assert np.isclose(sample_info[key], read_sample_info[key])
        else:
            assert sample_info[key] == read_sample_info[key]
                      

def test_add_source(tmp_path):
    wavelength = 1e-9
    energy = 1.9864459e-25 / wavelength

    with data.create_cxi(tmp_path / 'test_add_source.cxi') as f:
        data.add_source(f, wavelength)

    with h5py.File(tmp_path / 'test_add_source.cxi','r') as f:
        # Check this directly since we want to make sure it saved
        # the wavelength and energy
        read_wavelength = f['entry_1/instrument_1/source_1/wavelength'][()]
        read_energy = f['entry_1/instrument_1/source_1/energy'][()]

    assert np.isclose( wavelength, read_wavelength)
    assert np.isclose( energy, read_energy)

    
def test_add_detector(tmp_path):
    distance = 0.34
    basis = np.array([[0,-30e-6,0],
                      [-20e-6,0,0]]).astype(np.float32).transpose()
    corner = np.array((2550e-6,3825e-6,0.3)).astype(np.float32)
    
    with data.create_cxi(tmp_path / 'test_add_detector.cxi') as f:
        data.add_detector(f, distance, basis, corner=corner)

    with h5py.File(tmp_path / 'test_add_detector.cxi','r') as f:
        # Check this directly since we want to make sure it saved
        # the pixel sizes
        d1 = f['entry_1/instrument_1/detector_1']
        read_basis = d1['basis_vectors'][()]
        read_x_pix = d1['x_pixel_size'][()]
        read_y_pix = d1['y_pixel_size'][()]
        read_distance = d1['distance'][()]
        read_corner = d1['corner_position'][()]

    assert np.isclose(distance, read_distance)
    assert np.allclose(basis, read_basis)
    assert np.isclose(np.linalg.norm(basis[:,1]), read_x_pix)
    assert np.isclose(np.linalg.norm(basis[:,0]), read_y_pix)
    assert np.allclose(corner,read_corner)
   

def test_add_mask(tmp_path):
    mask = (np.random.rand(350,600) > 0.1).astype(np.uint8)

    with data.create_cxi(tmp_path / 'test_add_mask.cxi') as f:
        data.add_mask(f, mask)

    with h5py.File(tmp_path / 'test_add_mask.cxi','r') as f:
        read_mask = data.get_mask(f)

    assert np.all(mask == read_mask)

    
def test_add_dark(tmp_path):
    dark = np.random.rand(350,620)

    with data.create_cxi(tmp_path / 'test_add_dark.cxi') as f:
        data.add_dark(f, dark)

    with h5py.File(tmp_path / 'test_add_dark.cxi','r') as f:
        read_dark = data.get_dark(f)

    print(dark.shape)
    assert np.allclose(dark, read_dark)



def test_add_data(tmp_path):
    # First test from numpy, with axes
    fake_data = np.random.rand(100,256,256)
    axes = ['translation','y','x']

    with data.create_cxi(tmp_path / 'test_add_data.cxi') as f:
        data.add_data(f, fake_data, axes)

    with h5py.File(tmp_path / 'test_add_data.cxi','r') as f:
        # Check this directly since we want to make sure it saved
        # it in all the places it should have
        read_data_1 = f['entry_1/data_1/data'][()]
        read_data_2 = f['entry_1/instrument_1/detector_1/data'][()]
        read_axes = str(f['entry_1/instrument_1/detector_1/data'].attrs['axes'].decode())
    
    assert np.allclose(fake_data, read_data_1)
    assert np.allclose(fake_data, read_data_2)
    assert 'translation:y:x' == read_axes

    # Then test from torch, without axes
    fake_data = t.from_numpy(fake_data)
    
    with data.create_cxi(tmp_path / 'test_add_data_torch.cxi') as f:
        data.add_data(f, fake_data)

    with h5py.File(tmp_path / 'test_add_data_torch.cxi','r') as f:    
        read_data, axes = data.get_data(f)

    assert np.allclose(fake_data.numpy(),read_data)

def test_add_shot_to_shot_info(tmp_path):
    
    analyzer = np.random.rand(100)

    with data.create_cxi(tmp_path / 'test_add_shot_to_shot_info.cxi') as f:
        data.add_shot_to_shot_info(f, analyzer, 'analyzer_angle')
    
    with h5py.File(tmp_path / 'test_add_shot_to_shot_info.cxi') as f:
        # Check this directly since we want to make sure it saved
        # it in all the places it should have
        read_analyzer_1 = f['entry_1/data_1/analyzer_angle'][()]
        read_analyzer_2 = \
                       f['entry_1/instrument_1/detector_1/analyzer_angle'][()]
        read_analyzer_3 = f['entry_1/sample_1/geometry_1/analyzer_angle'][()]

    assert np.allclose(analyzer, read_analyzer_1)
    assert np.allclose(analyzer, read_analyzer_2)
    assert np.allclose(analyzer, read_analyzer_3)
    
        
def test_add_ptycho_translations(tmp_path):
    
    translations = np.random.rand(3,100)

    with data.create_cxi(tmp_path / 'test_add_ptycho_translations.cxi') as f:
        data.add_ptycho_translations(f, translations)
    
    with h5py.File(tmp_path / 'test_add_ptycho_translations.cxi','r') as f:
        # Check this directly since we want to make sure it saved
        # it in all the places it should have
        read_translations_1 = f['entry_1/data_1/translation'][()]
        read_translations_2 = \
                       f['entry_1/instrument_1/detector_1/translation'][()]
        read_translations_3 = f['entry_1/sample_1/geometry_1/translation'][()]

    assert np.allclose(-translations, read_translations_1)
    assert np.allclose(-translations, read_translations_2)
    assert np.allclose(-translations, read_translations_3)
