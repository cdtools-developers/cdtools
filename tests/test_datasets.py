from cdtools.datasets import CDataset, Ptycho2DDataset
from cdtools.tools import data as cdtdata
import numpy as np
import torch as t
import h5py
import datetime
from copy import deepcopy
import pytest
import itertools

#
# We start by testing the CDataset base class
#

def test_CDataset_init():
    entry_info = {'start_time': datetime.datetime.now(),
                  'title' : 'A simple test'}
    sample_info = {'name': 'A test sample',
                   'mass' : 3.4,
                   'unit_cell' : np.array([1,1,1,87,84.5,90])}
    wavelength = 1e-9
    detector_geometry = {'distance': 0.7,
                         'basis': np.array([[0,-30e-6,0],
                                            [-20e-6,0,0]]).transpose(),
                         'corner': np.array((2550e-6,3825e-6,0.3))}
    mask = np.ones((256,256))
    dataset = CDataset(entry_info, sample_info,
                       wavelength, detector_geometry, mask)

    assert t.all(t.eq(dataset.mask,t.tensor(mask.astype(bool))))
    assert dataset.entry_info == entry_info
    assert dataset.sample_info == sample_info
    assert dataset.wavelength == wavelength
    assert dataset.detector_geometry == detector_geometry


def test_CDataset_from_cxi(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        dataset = CDataset.from_cxi(cxi)

        # The entry metadata loaded
        for key in expected['entry metadata']:
            assert dataset.entry_info[key] == expected['entry metadata'][key]

        # Don't test for fidelity since this is tested in the data, just test
        # that it is loaded
        if expected['sample info'] is None:
            assert dataset.sample_info is None
        else:
            assert dataset.sample_info is not None

        assert np.isclose(dataset.wavelength,expected['wavelength'])

        # Just check one of the loaded attributes
        assert np.isclose(dataset.detector_geometry['distance'],
                          expected['detector']['distance'])
        # Check that the other ones are loaded but not for fidelity
        assert 'basis' in dataset.detector_geometry
        if expected['detector']['corner'] is not None:
            assert 'corner' in dataset.detector_geometry

        if expected['mask'] is not None:
            assert t.all(t.eq(t.tensor(expected['mask']),dataset.mask))

        if expected['dark'] is not None:
            assert t.all(t.eq(t.as_tensor(expected['dark'], dtype=t.float32),
                              dataset.background))

            

            
def test_CDataset_to_cxi(test_ptycho_cxis, tmp_path):
    for cxi, expected in test_ptycho_cxis:
        dataset = CDataset.from_cxi(cxi)
        with cdtdata.create_cxi(tmp_path / 'test_CDataset_to_cxi.cxi') as f:
            dataset.to_cxi(f)

        # Now we have to check that all the stuff was written
        with h5py.File(tmp_path / 'test_CDataset_to_cxi.cxi', 'r') as f:
            read_dataset = CDataset.from_cxi(f)

        assert dataset.entry_info == read_dataset.entry_info

        if dataset.sample_info is None:
            assert read_dataset.sample_info is None
        else:
            assert read_dataset.sample_info is not None

        assert np.isclose(dataset.wavelength, read_dataset.wavelength)

        
         # Just check one of the loaded attributes
        assert np.isclose(dataset.detector_geometry['distance'],
                          read_dataset.detector_geometry['distance'])
        # Check that the other ones are loaded but not for fidelity
        assert 'basis' in read_dataset.detector_geometry
        if dataset.detector_geometry['corner'] is not None:
            assert 'corner' in read_dataset.detector_geometry
            
        
        if dataset.mask is not None:
            assert t.all(t.eq(dataset.mask,read_dataset.mask))

        if dataset.background is not None:
            assert t.all(t.eq(dataset.background, read_dataset.background))



def test_CDataset_to(ptycho_cxi_1):
    dataset = CDataset.from_cxi(ptycho_cxi_1[0])

    dataset.to(dtype=t.float32)
    assert dataset.mask.dtype == t.bool
    # If cuda is available, check that moving the mask to CUDA works.
    if t.cuda.is_available():
        dataset.to(device='cuda:0')
        assert dataset.mask.device == t.device('cuda:0')
        assert dataset.background.device == t.device('cuda:0')


#
# And we then test the derived Ptychography class
#


def test_Ptycho2DDataset_init():
    entry_info = {'start_time': datetime.datetime.now(),
                  'title' : 'A simple test'}
    sample_info = {'name': 'A test sample',
                   'mass' : 3.4,
                   'unit_cell' : np.array([1,1,1,87,84.5,90])}
    wavelength = 1e-9
    detector_geometry = {'distance': 0.7,
                         'basis': np.array([[0,-30e-6,0],
                                            [-20e-6,0,0]]).transpose(),
                         'corner': np.array((2550e-6,3825e-6,0.3))}
    mask = np.ones((256,256))
    patterns = np.random.rand(20,256,256)
    translations = np.random.rand(20,3)
    
    dataset = Ptycho2DDataset(translations, patterns,
                                entry_info=entry_info,
                                sample_info=sample_info,
                                wavelength=wavelength,
                                detector_geometry=detector_geometry,
                                mask=mask)

    assert t.all(t.eq(dataset.mask,t.BoolTensor(mask)))
    assert dataset.entry_info == entry_info
    assert dataset.sample_info == sample_info
    assert dataset.wavelength == wavelength
    assert dataset.detector_geometry == detector_geometry
    assert t.allclose(dataset.patterns, t.as_tensor(patterns))
    assert t.allclose(dataset.translations, t.as_tensor(translations))


def test_Ptycho2DDataset_from_cxi(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        dataset = Ptycho2DDataset.from_cxi(cxi)

        # The entry metadata loaded
        for key in expected['entry metadata']:
            assert dataset.entry_info[key] == expected['entry metadata'][key]

        # Don't test for fidelity since this is tested in the data, just test
        # that it is loaded
        if expected['sample info'] is None:
            assert dataset.sample_info is None
        else:
            assert dataset.sample_info is not None

        assert np.isclose(dataset.wavelength,expected['wavelength'])

        # Just check one of the loaded attributes
        assert np.isclose(dataset.detector_geometry['distance'],
                          expected['detector']['distance'])
        # Check that the other ones are loaded but not for fidelity
        assert 'basis' in dataset.detector_geometry
        if expected['detector']['corner'] is not None:
            assert 'corner' in dataset.detector_geometry

        if expected['mask'] is not None:
            assert t.all(t.eq(t.tensor(expected['mask']),dataset.mask))

        if expected['dark'] is not None:
            assert t.all(t.eq(t.as_tensor(expected['dark'], dtype=t.float32),
                              dataset.background))

            
        assert t.allclose(t.tensor(expected['data']),dataset.patterns)
        assert t.allclose(t.tensor(expected['translations']),dataset.translations)


def test_Ptycho2DDataset_to_cxi(test_ptycho_cxis, tmp_path):
    for cxi, expected in test_ptycho_cxis:
        print('loading dataset')
        dataset = Ptycho2DDataset.from_cxi(cxi)
        print('dataset mask is type', dataset.mask.dtype)
        with cdtdata.create_cxi(tmp_path / 'test_Ptycho2DDataset_to_cxi.cxi') as f:
            dataset.to_cxi(f)

        # Now we have to check that all the stuff was written
        with h5py.File(tmp_path / 'test_Ptycho2DDataset_to_cxi.cxi', 'r') as f:
            read_dataset = Ptycho2DDataset.from_cxi(f)

        assert dataset.entry_info == read_dataset.entry_info

        if dataset.sample_info is None:
            assert read_dataset.sample_info is None
        else:
            assert read_dataset.sample_info is not None

        assert np.isclose(dataset.wavelength, read_dataset.wavelength)

        
         # Just check one of the loaded attributes
        assert np.isclose(dataset.detector_geometry['distance'],
                          read_dataset.detector_geometry['distance'])
        # Check that the other ones are loaded but not for fidelity
        assert 'basis' in read_dataset.detector_geometry
        if dataset.detector_geometry['corner'] is not None:
            assert 'corner' in read_dataset.detector_geometry
            

        
        if dataset.mask is not None:
            assert t.all(t.eq(dataset.mask,read_dataset.mask))

        if dataset.background is not None:
            assert t.all(t.eq(dataset.background, read_dataset.background))

        assert t.allclose(dataset.patterns, read_dataset.patterns)
        assert t.allclose(dataset.translations, read_dataset.translations)


def test_Ptycho2DDataset_to(ptycho_cxi_1):
    dataset = Ptycho2DDataset.from_cxi(ptycho_cxi_1[0])
    
    dataset.to(dtype=t.float64)
    assert dataset.mask.dtype == t.bool
    assert dataset.patterns.dtype == t.float64
    assert dataset.translations.dtype == t.float64
    # If cuda is available, check that moving the mask to CUDA works.
    if t.cuda.is_available():
        dataset.to(device='cuda:0')
        assert dataset.mask.device == t.device('cuda:0')
        assert dataset.background.device == t.device('cuda:0')
        assert dataset.patterns.device == t.device('cuda:0')
        assert dataset.translations.device == t.device('cuda:0')


def test_Ptycho2DDataset_ops(ptycho_cxi_1):
    cxi, expected = ptycho_cxi_1
    dataset = Ptycho2DDataset.from_cxi(cxi)
    dataset.get_as('cpu')

    assert len(dataset) == expected['data'].shape[0]
    (idx, translation), pattern = dataset[3]
    assert idx == 3
    assert t.allclose(translation, t.tensor(expected['translations'][3,:]))
    assert t.allclose(pattern, t.tensor(expected['data'][3,:,:]))


def test_Ptycho2DDataset_get_as(ptycho_cxi_1):
    cxi, expected = ptycho_cxi_1
    dataset = Ptycho2DDataset.from_cxi(cxi)
    if t.cuda.is_available():
        dataset.get_as('cuda:0')
        assert len(dataset) == expected['data'].shape[0]

        (idx, translation), pattern = dataset[3]
        assert str(translation.device) == 'cuda:0'
        assert str(pattern.device) == 'cuda:0'
        
        assert idx == 3
        assert t.allclose(translation.to(device='cpu'),
                          t.tensor(expected['translations'][3,:]))
        assert t.allclose(pattern.to(device='cpu'),
                          t.tensor(expected['data'][3,:,:]))


def test_Ptycho2DDataset_downsample(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        dataset = Ptycho2DDataset.from_cxi(cxi)

        # First we test the case of downsampling by 2 against some explicit
        # calculations
        copied_dataset = deepcopy(dataset)
        copied_dataset.downsample(2)

        # May start failing if the test datasets are changed to include
        # a dataset with any dimension not even. That's a problem with the
        # test, not the code. Sorry! -Abe
        assert t.allclose(
            copied_dataset.patterns,
            dataset.patterns[:,::2,::2] +
            dataset.patterns[:,1::2,::2] +
            dataset.patterns[:,::2,1::2] +
            dataset.patterns[:,1::2,1::2]
        )
        
        assert t.allclose(
            copied_dataset.mask,
            t.logical_and(
                t.logical_and(dataset.mask[::2,::2],
                              dataset.mask[1::2,::2]),
                t.logical_and(dataset.mask[::2,1::2],
                              dataset.mask[1::2,1::2]),
            )
        )


        
        if dataset.background is not None:
            assert t.allclose(
                copied_dataset.background,
                dataset.background[::2,::2] +
                dataset.background[1::2,::2] +
                dataset.background[::2,1::2] +
                dataset.background[1::2,1::2]
        )

        # And then we just test the shape for a few factors, and check that
        # it doesn't fail on edge cases (e.g. factor=1)
        for factor in [1, 2, 3]:
            copied_dataset = deepcopy(dataset)
            copied_dataset.downsample(factor=factor)

            expected_pattern_shape = np.concatenate(
                [[dataset.patterns.shape[0]],
                 np.array(dataset.patterns.shape[-2:]) // factor]
            )

            assert np.allclose(expected_pattern_shape,
                               np.array(copied_dataset.patterns.shape))
            
            assert np.allclose(np.array(dataset.mask.shape) // factor,
                               np.array(copied_dataset.mask.shape))
            
            if dataset.background is not None:
                assert np.allclose(np.array(dataset.background.shape) // factor,
                                   np.array(copied_dataset.background.shape))


def test_Ptycho2DDataset_remove_translations_mask(ptycho_cxi_1):
    # Grab dataset
    cxi, expected = ptycho_cxi_1
    dataset = Ptycho2DDataset.from_cxi(cxi)
    copied_dataset = deepcopy(dataset)

    # Test 1: Complain when the the mask is not the same shape as the pattern
    # length
    with pytest.raises(ValueError) as excinfo:
        copied_dataset.remove_translations_mask(mask=t.zeros(10))
    assert ('The mask must have the same length') in str(excinfo.value)

    # Test 2: Remove the mask from the dataset
    mask_success = t.zeros(len(copied_dataset.patterns))
    mask_success[1] = 1
    mask_success[10] = 1
    mask_success[-1] = 1
    mask_success = mask_success.bool()
    copied_dataset.remove_translations_mask(mask=mask_success)

    # test if the mask is removed and patterns length is correct
    assert len(copied_dataset.patterns) == len(mask_success) - 3


def test_Ptycho2DDataset_crop_translations(ptycho_cxi_1):
    # Grab dataset
    cxi, expected = ptycho_cxi_1
    dataset = Ptycho2DDataset.from_cxi(cxi)
    copied_dataset = deepcopy(dataset)

    # Test 1: Complain when the the bounds of an ROI are correctly defined,
    #   but it does not contain any sample positions inside of it. The 
    #   translations in ptycho_cxi_1 ranges from 0m to -300m in both x and y 
    #   (it looks like a line scan). We select an ROI at (0, -300) which should 
    #   not contain any translation positions.
    with pytest.raises(ValueError) as excinfo:
        copied_dataset.crop_translations(roi=(0,-5,-295,-300))
    assert('(i.e., patterns and translations will be empty)') in str(excinfo.value)

    # Test 2: Draw an ROI that's centered in the middle of the x/y translation range
    #   and make sure that the first and last x/y elements in dataset.translate
    #   contain x_left, x_right, y_top, and y_bottom

    #   Make tuples that will store the x and y positions. This will be used for
    #   making permutations of the x/y positions in the ROI later.
    x_left, y_top = dataset.translations[10,:2]
    x_right, y_bottom = dataset.translations[-11,:2]

    x_permutations = ((x_left, x_right), (x_right, x_left))
    y_permutations = ((y_top, y_bottom), (y_bottom, y_top))
    roi_permutations = tuple((x1, x2, y1, y2) for (x1, x2), (y1, y2) in
                              itertools.product(x_permutations, y_permutations))

    #   Get the dataset
    copied_dataset.crop_translations(roi=roi_permutations[0])

    #   Execute the actual test
    assert (copied_dataset.translations[0,0] in x_permutations[0]) and \
        (copied_dataset.translations[-1,0] in x_permutations[0])
    
    assert (copied_dataset.translations[0,1] in y_permutations[0]) and \
        (copied_dataset.translations[-1,1] in y_permutations[0])

    # Test 3: Check if the shape of dataset.patterns and dataset.translate is correct
    #   (designed to be 20 fewer rows here)
    #   In the future, this should include a check for dataset.intensities once
    #   an appropriate cxi file is set up for conftest.
    expected_patterns_shape = np.concatenate([[dataset.patterns.shape[0] - 20], 
                                            dataset.patterns.shape[-2:]])
    
    expected_translations_shape = np.concatenate([[dataset.translations.shape[0] - 20], 
                                                dataset.translations.shape[-1:]])

    assert np.allclose(np.array(copied_dataset.patterns.shape), expected_patterns_shape)
        
    assert np.allclose(np.array(copied_dataset.translations.shape), expected_translations_shape)

    # Test 4: Make sure that we always get the same result no matter what order we
    #   define the left/right and bottom/top values in roi, provided that roi[:2] 
    #   and roi[2:] correspond with the x and y coordinates, respectively.

    #   Check each permutation
    for roi in roi_permutations:
        # Copy the dataset again; dataset will be modified each time crop_translation
        # is successfully executed.
        copied_dataset = deepcopy(dataset)
        copied_dataset.crop_translations(roi=roi)

        # Check if the contents of dataset.patterns and dataset.translate is correct
        assert t.allclose(copied_dataset.patterns, dataset.patterns[10:-10,:])

        assert t.allclose(copied_dataset.translations, dataset.translations[10:-10,:])
