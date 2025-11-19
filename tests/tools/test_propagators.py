import numpy as np
import torch as t
import pytest
import scipy.datasets
from scipy import stats
from matplotlib import pyplot as plt

from cdtools.tools import initializers
from cdtools.tools import propagators
from cdtools.tools import image_processing


@pytest.fixture(scope='module')
def exit_waves_1():
    # Import scipy test image and add a random phase
    obj = scipy.datasets.ascent()[0:64, 0:64].astype(np.complex128)
    arr = np.random.random_sample((64, 64))
    obj *= (arr + (1 - arr**2)**.5 * 1j)
    obj = t.as_tensor(obj)

    # Construct wavefront from image
    probe = initializers.gaussian([64, 64], [5, 5], amplitude=1e3)
    return probe * obj


def test_far_field(exit_waves_1):
    # Far field diffraction patterns calculated by numpy with zero frequency in center
    np_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(exit_waves_1.numpy()), norm='ortho'))

    assert (np.allclose(np_result, propagators.far_field(exit_waves_1).numpy()))


def test_inverse_far_field(exit_waves_1):
    # We want the inverse far field to map back to the exit waves with no intensity corrections
    # Far field result for exit waves calculated with numpy
    far_field_np_result = t.as_tensor(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(exit_waves_1.numpy()), norm='ortho')))

    assert (np.allclose(exit_waves_1, propagators.inverse_far_field(far_field_np_result)))


def test_generate_high_NA_k_intensity_map():

    # We need to generate a plausible scenario. I will start
    # by using the initializer to generate a reasonable exit wave geometry
    # and detector pair
    basis = t.Tensor([[0, -30e-6, 0],
                      [-20e-6, 0, 0]]).transpose(0, 1)
    shape = t.Size([478, 573])
    wavelength = 1e-9
    distance = 1
    rs_basis = \
        initializers.exit_wave_geometry(basis, shape, wavelength, distance)

    k_map, intensity_map = propagators.generate_high_NA_k_intensity_map(
        rs_basis, basis, shape, distance, wavelength,
        dtype=t.float32)

    # generate a good test exit wave
    i = (np.arange(478) - 240)
    j = (np.arange(573) - 270)
    Is, Js = np.meshgrid(i, j, indexing='ij')
    wavefield = ((np.abs(Is) < 20) * (np.abs(Js) < 25)).astype(np.complex128)
    t_wavefield = t.as_tensor(wavefield, dtype=t.complex64)

    high_NA_propagated = propagators.high_NA_far_field(
        t_wavefield, k_map, intensity_map=intensity_map)
    low_NA_propagated = propagators.far_field(t_wavefield)

    low_NA = low_NA_propagated.numpy()
    high_NA = high_NA_propagated.numpy()

    # Checking first that for a low-NA propagation they give the same result
    # 1e-4 tolerance seems to be reasonable in this comparison given my
    # exploration with the code
    # assert np.max(np.abs(high_NA-low_NA))/np.max(np.abs(low_NA)) < 1e-4

    # Now I will explore some results with a tilted sample
    # print(rs_basis)
    # print(rs_basis_tilted)
    distance = 0.01  # 6e-3
    rs_basis = initializers.exit_wave_geometry(basis, shape, wavelength, distance)
    rs_basis_tilted = rs_basis.clone()
    rs_basis_tilted[2, 1] = rs_basis_tilted[0, 1]

    k_map, intensity_map = propagators.generate_high_NA_k_intensity_map(
        rs_basis_tilted, basis, shape, distance, wavelength,
        dtype=t.float32)

    high_NA_propagated = propagators.high_NA_far_field(
        t_wavefield, k_map, intensity_map=intensity_map)
    low_NA_propagated = propagators.far_field(t_wavefield)

    low_NA = low_NA_propagated.numpy()
    high_NA = high_NA_propagated.numpy()

    print(low_NA.shape, high_NA.shape)

    # plt.close('all')
    # plt.imshow(np.abs(low_NA))
    # plt.figure()
    # plt.imshow(np.abs(high_NA))
    # plt.colorbar()
    # plt.imshow(np.abs(wavefield))
    # plt.show()

    # Now I want to test that it doesn't crash for wavefields of various shapes
    propagators.high_NA_far_field(t_wavefield.unsqueeze(0),
                                  k_map, intensity_map=intensity_map)
    propagators.high_NA_far_field(t_wavefield.unsqueeze(0).unsqueeze(0),
                                  k_map, intensity_map=intensity_map)

    # I believe this works, but I still would like to get a second method for
    # simulating at least one diffraction pattern as an independent check

    # assert 0


def test_near_field_direction():
    #
    # The idea is to make sure that near field propagation is consistent
    # with far-field propagation. Because we consider the Fourier
    # transform to be the far-field propagator, we should find that
    # near-field propagation becomes far-field propagation in the limit
    # that z goes to positive infinity. This consistency requirement
    # forces a specific choice of sign for near-field propagation - choosing
    # it incorrectly will make the correct limit obtain at negative
    # infinity.
    #
    # We can check consistency by setting up a gaussian wavefield with an
    # offset in the far field. Then, we can do some near-field propagation,
    # and we will ensure that the centroid of the beam moves in the correct
    # direction. If the sign is incorrect, it will move in the wrong direction.

    x = (np.arange(901) - 400)
    y = (np.arange(1200) - 500)
    Ys, Xs = np.meshgrid(y, x)
    Rs = np.sqrt(Xs**2 + Ys**2)

    E0_fourier = t.as_tensor(np.exp(-Rs**2 / (2 * 40**2)), dtype=t.complex64)
    E0_real = propagators.inverse_far_field(E0_fourier)
    # This is in the top-left corner in Fourier space

    wavelength = 3e-9  # nm
    z = 1000e-9  # nm
    asp = propagators.generate_angular_spectrum_propagator(
        E0_real.shape, (1.5e-9, 1e-9), wavelength, z, dtype=t.complex64)

    Ez_real = propagators.near_field(E0_real, asp)

    centroid = image_processing.centroid(t.abs(Ez_real))

    # Assert it's in top half
    assert centroid[0] < Ez_real.shape[0] // 2
    # Assert it's in left half
    assert centroid[1] < Ez_real.shape[1] // 2

    # plt.imshow(t.abs(E0_fourier))
    # plt.figure()
    # plt.imshow(t.abs(E0_real))
    # plt.figure()
    # plt.imshow(t.abs(Ez_real))
    # plt.show()


def test_near_field():

    # The strategy is to compare the propagation of a gaussian beam to
    # the propagation in the paraxial approximation.

    x = (np.arange(901) - 450) * 1.5e-9
    y = (np.arange(1200) - 600) * 1e-9
    Ys, Xs = np.meshgrid(y, x)
    Rs = np.sqrt(Xs**2 + Ys**2)

    wavelength = 3e-9  # nm
    sigma = 20e-9  # nm
    z = 1000e-9  # nm

    k = 2 * np.pi / wavelength
    w0 = np.sqrt(2) * sigma
    zr = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + (z / zr)**2)
    Rz = z * (1 + (zr / z)**2)

    E0 = np.exp(-Rs**2 / w0**2)

    # The analytical expression for propagation of a gaussian beam in the
    # paraxial approx
    # The sign convention I'm using here is opposite from what is on
    # wikipedia (as of June 2021), but it is I believe the most sensible
    # choice. By choosing e^(ikx) to be the plane wave that's propagating
    # in the x direction, we preserve two nice properties:
    # 1) The time-dependence is e^(-iwt+ikx), which meshes with the choice
    #    we like to make as physicists of e^(-iwt) for time-dependence
    # 2) This makes it so that the Fourier transform does far-field
    #    propagation in direction of light propagation. The logic is as
    #    follows:
    #    a) We must assume that light only propagates one direction through
    #       the plane, in order to have well-defined propagation
    #    b) Fixing that direction to be the positive direciton of propagation,
    #       we can consider the in-plane image of a plane wave propagating
    #       through the plane of interest.
    #    c) We can ask whether the 2D Fourier transform of that in-plane image
    #       produces a spot on the same side of zero as the in-plane component
    #       of k.
    #    If we choose e^(-ikx) to represent light propagating along K, the
    #    answer is no, and we find we have to use the inverse FT instead.
    #    Thus, e^(ikx) is the right choice here.

    Ez = w0 / wz * np.exp(-Rs**2 / wz**2) * np.exp(1j * k * (z + Rs**2 / (2 * Rz)) - 1j * np.arctan(z / zr))
    Ez_nozphase = Ez * np.exp(-1j * k * z)

    # First we check it normally
    asp = propagators.generate_angular_spectrum_propagator(
        E0.shape, (1.5e-9, 1e-9), wavelength, z, dtype=t.complex128)

    Ez_t = propagators.near_field(t.as_tensor(E0), asp).numpy()

    # Check for at least 10^-3 relative accuracy in this scenario
    assert np.max(np.abs(Ez_nozphase - Ez_t)) < 1e-3 * np.max(np.abs(Ez_nozphase))

    Emz = np.conj(Ez_nozphase)

    Emz_t = propagators.inverse_near_field(t.as_tensor(E0), asp).numpy()

    # Again, 10^-3 is about all the accuracy we can expect
    assert np.max(np.abs(Emz - Emz_t)) < 1e-3 * np.max(np.abs(Emz))

    # Then, we check that the bandlimiting at least does something
    asp = propagators.generate_angular_spectrum_propagator(
        E0.shape, (1.5e-9, 1e-9), wavelength, z,
        dtype=t.complex128, bandlimit=0.3)

    assert asp[140, 0] == 0
    assert asp[0, 180] == 0
    assert asp[130, 0] != 0
    assert asp[0, 175] != 0

    # Then, we check that automatic differentiation works
    z = t.tensor([z], requires_grad=True)
    spacing = t.tensor((1.5e-9, 1e-9), requires_grad=True)
    wavelength = t.tensor([wavelength], requires_grad=True)

    asp = propagators.generate_angular_spectrum_propagator(
        E0.shape, spacing, wavelength, z)

    t.real(asp[10, 10]).backward()
    assert z.grad != 0
    assert spacing.grad[0] != 0
    assert wavelength.grad != 0


def test_generalized_near_field():

    # The strategy is to compare the propagation of a gaussian beam to
    # the propagation in the paraxial approximation.

    # For this one, we want to test it on a rotated coordinate system
    # First, we should do a test with the phase ramp along the z direction
    # explicitly included

    basis = np.array([[0, -1.5e-9], [-1e-9, 0], [0, 0]])
    i_vec, j_vec = np.arange(901) - 450, np.arange(1200) - 600
    Is, Js = np.meshgrid(i_vec, j_vec, indexing='ij')
    Xs_0, Ys_0, Zs_0 = np.tensordot(basis, np.stack([Is, Js]), axes=1)
    # x = (np.arange(901) - 450) * 1.5e-9
    # y = (np.arange(1200) - 600) * 1e-9
    # Xs_0,Ys_0 = np.meshgrid(x,y)
    # Zs_0 = np.zeros(Xs_0.shape)

    Positions = np.stack([Xs_0, Ys_0, Zs_0])

    # assert 0
    wavelength = 3e-9  # nm
    sigma = 20e-9  # nm
    z = 1000e-9  # nm

    k = 2 * np.pi / wavelength
    w0 = np.sqrt(2) * sigma
    zr = np.pi * w0**2 / wavelength

    # The analytical expression for propagation of a gaussian beam in the
    # paraxial approx
    def get_w(Zs):
        return w0 * np.sqrt(1 + (Zs / zr)**2)

    def get_inv_R(Zs):
        return Zs / (Zs**2 + zr**2)

    def get_E(Xs, Ys, Zs, correct=True):
        # Again, this follows the convention opposite from Wikipedia. See
        # the note in test_angular_spectrum_propagator.

        # if correct is True, remove the e^(ikz) dependence

        Rs_sq = Xs**2 + Ys**2
        Wzs = get_w(Zs)
        E = w0 / Wzs * np.exp(-Rs_sq / Wzs**2) *\
            np.exp(1j * k * (Zs + Rs_sq * get_inv_R(Zs) / 2) - 1j * np.arctan(Zs / zr))

        # This removes the z-dependence of the phase
        if correct:
            E = E * np.exp(-1j * k * Zs)
        return E

    def check_equiv(analytical, numerical):
        phase = np.angle(np.mean(numerical.conj() * analytical))
        comp = np.exp(1j * phase) * numerical
        return (np.max(np.abs(analytical - comp)) < 1e-3 * np.max(np.abs(analytical)))

    # We make some rotation matrices to test

    # This tests the straight ahead case
    IdentityMatrix = np.eye(3)

    # This tests a rotation about the y axis
    th = np.deg2rad(5)
    Ry = np.array([[np.cos(th), 0, np.sin(th)],
                   [0, 1, 0],
                   [-np.sin(th), 0, np.cos(th)]])

    # This tests a rotation about two axes
    phi = np.deg2rad(2)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    Rboth = np.matmul(Rx, Ry)

    # This tests a shearing
    shear = 0.23
    Rshear = np.array([[1, shear, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

    # This tests an inversion of the axes
    Rinv = np.array([[-1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]])

    # This tests a reflection about the y-z plane
    Rrefl = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])

    # This tests a shearing and a rotation together
    Rall = np.matmul(Rrefl, np.matmul(Rboth, Rshear))

    # And we make some propagation vectors to test:

    # This is along the z direction
    z_dir = np.array([0, 0, 1])

    # This checks that it's not sensitive to the magnitude
    z_dir_large = np.array([0, 0, 10])

    # And finally some offset vectors

    # This checks straight ahead
    z_offset = np.array([0, 0, z])

    # This checks with an offset in x and y
    shear_offset = np.array([0.1 * z, -0.03 * z, z])

    # This checks with an offset in x and y, with negative z
    shear_back_offset = np.array([0.1 * z, -0.03 * z, -z])

    rot_mats = [Rrefl, IdentityMatrix, Rinv, Rboth, Rboth, Rboth, Rall, Rall, Rall, IdentityMatrix, Rall]
    offset_vecs = [z_offset] * 8 + [shear_offset] + [shear_back_offset] * 2
    propagation_vecs = ['perp', 'offset', z_dir,
                        'perp', 'offset', z_dir_large,
                        'perp', 'offset', z_dir_large,
                        z_dir, z_dir_large]
    purposes = ['standard'] * 3 + ['both-rot'] * 3 + ['shear-rot'] * 3 + ['backward'] * 2

    # rot_mats = [Ry.transpose()]
    # offset = np.cross(np.dot(Ry.transpose(),basis)[:,0],
    #                  np.dot(Ry.transpose(),basis)[:,1])
    # offset /= np.linalg.norm(offset) / 3e-6
    # offset_vecs = [-offset]#[shear_offset]
    # propagation_vecs = [z_dir]
    # purposes=['meh']

    for purpose, rot_mat, offset_vec, propagation_vec in zip(purposes, rot_mats, offset_vecs, propagation_vecs):
        print('Testing', purpose)
        Xs, Ys, Zs_0 = np.tensordot(rot_mat, Positions, axes=1)
        new_basis = np.dot(rot_mat, basis)
        Xs_prop, Ys_prop, Zs_prop = np.stack([Xs, Ys, Zs_0]) \
            + offset_vec[:, None, None]

        print('Propagate Along', propagation_vec)

        if str(propagation_vec) == 'perp':
            E0 = get_E(Xs, Ys, Zs_0, correct=False)
            Ez = get_E(Xs_prop, Ys_prop, Zs_prop, correct=False)
            asp = propagators.generate_generalized_angular_spectrum_propagator(
                E0.shape, new_basis, wavelength, offset_vec, dtype=t.complex128)
        elif str(propagation_vec) == 'offset':
            E0 = get_E(Xs, Ys, Zs_0, correct=True)
            Ez = get_E(Xs_prop, Ys_prop, Zs_prop, correct=True)
            asp = propagators.generate_generalized_angular_spectrum_propagator(
                E0.shape, new_basis, wavelength, offset_vec,
                dtype=t.complex128, propagate_along_offset=True)
        else:
            E0 = get_E(Xs, Ys, Zs_0, correct=True)
            Ez = get_E(Xs_prop, Ys_prop, Zs_prop, correct=True)
            asp = propagators.generate_generalized_angular_spectrum_propagator(
                E0.shape, new_basis, wavelength, offset_vec, dtype=t.complex128,
                propagation_vector=propagation_vec)

        Ez_t = propagators.near_field(t.as_tensor(E0), asp).numpy()

        # Check for at least 10^-3 relative accuracy in this scenario
        if not check_equiv(Ez, Ez_t):
            # if True:
            plt.close('all')
            plt.figure()
            plt.imshow(np.angle(E0))
            plt.title('Angle of E0')
            plt.figure()
            plt.imshow(np.angle(Ez))
            plt.title('Angle of analytically calculated Ez')
            plt.figure()
            plt.imshow(np.abs(Ez))
            plt.title('Magnitude of analytically calculated Ez')
            plt.figure()
            plt.imshow(np.abs(Ez_t))
            plt.title('Magnitude of numerically calculated Ez')
            plt.figure()
            plt.imshow(np.angle(Ez_t))
            plt.title('Angle of numerically calculated Ez')
            plt.figure()
            plt.imshow(np.abs(Ez - Ez_t))
            plt.title('Magnitude of difference')
            plt.show()

        assert check_equiv(Ez, Ez_t)

        Em0_t = propagators.inverse_near_field(t.as_tensor(Ez), asp).numpy()

        assert check_equiv(E0, Em0_t)

        print('Test Successful')

    # One final test, to see if any of a few arbitrary rotations will
    # change the predicted propagation if everything else is kept
    # constant
    Rrands = [stats.ortho_group.rvs(3) for i in range(3)]
    Xs, Ys, Zs_0 = np.tensordot(Rboth, Positions, axes=1)
    new_basis = np.dot(rot_mat, basis)
    offset_vec = shear_back_offset
    propagation_vec = z_dir

    E0 = get_E(Xs, Ys, Zs_0, correct=True)
    asp = propagators.generate_generalized_angular_spectrum_propagator(
        E0.shape, new_basis, wavelength, offset_vec,
        dtype=t.complex128, propagation_vector=propagation_vec)
    Ez_t = propagators.near_field(t.as_tensor(E0), asp).numpy()

    for Rrand in Rrands:
        Xs, Ys, Zs_0 = np.tensordot(Rrand, np.tensordot(Rboth, Positions, axes=1), axes=1)
        rot_offset = np.dot(Rrand, offset_vec)
        rot_basis = np.dot(Rrand, new_basis)
        rot_prop = np.dot(Rrand, propagation_vec)
        asp = propagators.generate_generalized_angular_spectrum_propagator(
            E0.shape, rot_basis, wavelength, rot_offset,
            dtype=t.complex128, propagation_vector=rot_prop)
        Ez_rot_t = propagators.near_field(t.as_tensor(E0), asp).numpy()

        assert np.max(np.abs(Ez_t - Ez_rot_t)) < 1e-3 * np.max(np.abs(Ez_t))


def test_inverse_near_field():

    x = (np.arange(800) - 400) * 1.5e-9
    y = (np.arange(1200) - 600) * 1e-9
    Ys, Xs = np.meshgrid(y, x)
    Rs = np.sqrt(Xs**2 + Ys**2)

    wavelength = 3e-9  # nm
    sigma = 20e-9  # nm
    z = 1000e-9  # nm

    w0 = np.sqrt(2) * sigma
    E0 = np.exp(-Rs**2 / w0**2)

    asp = propagators.generate_angular_spectrum_propagator(
        E0.shape, (1.5e-9, 1e-9), wavelength, z, dtype=t.complex128)

    E0 = t.as_tensor(E0, dtype=t.complex128)
    E_prop = propagators.near_field(E0, asp)

    E_backprop = propagators.inverse_near_field(E_prop, asp)

    # We just want to check that it actually is the inverse
    assert t.all(t.isclose(E0, E_backprop))
