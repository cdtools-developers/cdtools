import numpy as np
import torch as t
from scipy import linalg as la
from scipy.sparse import linalg as spla

from cdtools.tools import analysis, initializers


def test_product_svd():

    rank = 4
    shape_A = (12, rank)
    shape_B = (rank, 9)
    A = np.random.rand(*shape_A) + 1j * np.random.rand(*shape_A)
    B = np.random.rand(*shape_B) + 1j * np.random.rand(*shape_B)

    AB = np.matmul(A, B)
    U_1, S_1, Vh_1 = t.linalg.svd(t.as_tensor(AB), full_matrices=False)

    U_2, S_2, Vh_2 = analysis.product_svd(t.as_tensor(A), t.as_tensor(B))
    check_AB = U_2 @ t.diag_embed(S_2).to(dtype=Vh_2.dtype) @ Vh_2

    # So, at a minimum, U S Vh = AB
    assert np.allclose(AB, check_AB.numpy())

    # SVD is only defined up to an arbitrary complex valued phase per
    # singular vector, so all we can ask for in the comparison is that the
    # magnitudes here are
    assert np.allclose(S_1[:rank].numpy(), S_2.numpy())
    prod_U = U_1[:, :rank].transpose(0, 1).conj() @ U_2
    prod_Vh = Vh_1[:rank, :] @ Vh_2.transpose(0, 1).conj()
    assert np.allclose(t.abs(prod_U).numpy(), np.eye(rank))
    assert np.allclose(t.abs(prod_Vh).numpy(), np.eye(rank))
    # Confirms that the phases are consistent between the two, I think
    # it's redundant with the first check but I'm not sure
    assert np.allclose(prod_Vh.numpy(), prod_U.numpy())

    # test with numpy
    U_3, S_3, Vh_3 = analysis.product_svd(A, B)
    assert isinstance(U_3, np.ndarray)
    assert isinstance(S_3, np.ndarray)
    assert isinstance(Vh_3, np.ndarray)

    assert np.allclose(S_1[:rank].numpy(), S_3)
    prod_U = U_1[:, :rank].transpose(0, 1).numpy().conj() @ U_3
    prod_Vh = Vh_1[:rank, :].numpy() @ Vh_3.transpose().conj()
    assert np.allclose(np.abs(prod_U), np.eye(rank))
    assert np.allclose(np.abs(prod_Vh), np.eye(rank))
    # Confirms that the phases are consistent between the two, I think
    # it's redundant with the first check but I'm not sure
    assert np.allclose(prod_Vh, prod_U)


def test_orthogonalize_probes():

    op = analysis.orthogonalize_probes

    probe_xs = np.arange(64) - 32
    probe_ys = np.arange(76) - 38
    probe_Ys, probe_Xs = np.meshgrid(probe_ys, probe_xs)
    probe_Rs = np.sqrt(probe_Xs**2 + probe_Ys**2)

    probes = np.array(
        [
            10 * np.exp(-(probe_Rs**2) / (2 * 10**2 + 1j)),
            3 * np.exp(-(probe_Rs**2) / (2 * 12**2 - 3j)),
            1 * np.exp(-(probe_Rs**2) / (2 * 15**2)),
        ]
    )

    weight_matrix_none = None
    weight_matrix_single = np.random.randn(1, 3) + 1j * np.random.randn(1, 3)
    weight_matrix_small = np.random.randn(2, 3) + 1j * np.random.randn(2, 3)
    weight_matrix_medium = np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
    weight_matrix_large = np.random.randn(7, 3) + 1j * np.random.randn(7, 3)

    weight_matrices = [
        weight_matrix_none,
        weight_matrix_single,
        weight_matrix_small,
        weight_matrix_medium,
        weight_matrix_large,
    ]

    for weight_matrix in weight_matrices:
        ortho_probes_np, rwm_np = op(
            probes, weight_matrix=weight_matrix, return_reexpressed_weights=True
        )
        assert isinstance(ortho_probes_np, np.ndarray)
        assert isinstance(rwm_np, np.ndarray)

        probes_t = t.as_tensor(probes)
        wm_t = (
            t.as_tensor(weight_matrix) if weight_matrix is not None else weight_matrix
        )

        ortho_probes_t, rwm_t = op(
            probes_t, weight_matrix=wm_t, return_reexpressed_weights=True
        )
        assert t.is_tensor(ortho_probes_t)
        assert t.is_tensor(rwm_t)

        assert np.allclose(ortho_probes_np, ortho_probes_t.numpy())
        assert np.allclose(rwm_np, rwm_t.numpy())

        # Now we test a if wm @ ortho_probes is actually the original
        # input
        if weight_matrix is not None:
            realized_probes = np.tensordot(weight_matrix, probes, axes=1)
        else:
            realized_probes = probes

        calculated_probes = np.tensordot(rwm_np, ortho_probes_np, axes=1)

        assert np.allclose(realized_probes, calculated_probes)

        # Now we test if the orthogonalized probes are orthogonalized
        reshaped_probes = ortho_probes_np.reshape(
            (
                ortho_probes_np.shape[0],
                ortho_probes_np.shape[1] * ortho_probes_np.shape[2],
            )
        )
        products = np.matmul(reshaped_probes, reshaped_probes.conj().transpose())

        if weight_matrix is not None:
            output_nmodes = min(weight_matrix.shape[0], probes.shape[0])
        else:
            output_nmodes = probes.shape[0]

        for i in range(output_nmodes):
            for j in range(i + 1, output_nmodes):
                assert np.isclose(products[i, j], 0)

        # And now we test if they multiply to the same density matrix as
        # the original probes + weight matrix
        reshaped_realized_probes = realized_probes.reshape(
            (
                realized_probes.shape[0],
                realized_probes.shape[1] * realized_probes.shape[2],
            )
        )

        dm_original = np.matmul(
            reshaped_realized_probes.conj().transpose(), reshaped_realized_probes
        )
        dm_output = np.matmul(reshaped_probes.conj().transpose(), reshaped_probes)
        assert np.allclose(dm_original, dm_output)

        # And finally, we confirm that what we have are the eigenvectors/values
        # of that density matrix
        w, v = spla.eigsh(dm_original, k=output_nmodes)
        assert np.allclose(w, np.diag(products))

        cross_products = np.matmul(reshaped_probes, np.sqrt(w) * v)
        # The abs accounts for the fact that the phase of the eigenvectors
        # is undefined
        assert np.allclose(np.abs(cross_products), np.abs(products))


def test_standardize():

    # Start by making a probe and object that should meet the standardization
    # conditions
    probe = initializers.gaussian((230, 240), (20, 20), curvature=(0.01, 0.01)).numpy()
    probe = probe * np.sqrt(len(probe.ravel()) / np.sum(np.abs(probe) ** 2))
    probe = probe * np.exp(-1j * np.angle(np.sum(probe)))

    assert np.isclose(1, np.sum(np.abs(probe) ** 2) / len(probe.ravel()))
    assert np.angle(np.sum(probe)) < 2e-7

    obj = 30 * np.random.rand(230, 240) * np.exp(1j * (np.random.rand(230, 240) - 0.5))
    obj_slice = np.s_[
        (obj.shape[0] // 8) * 3:(obj.shape[0] // 8) * 5,
        (obj.shape[1] // 8) * 3:(obj.shape[1] // 8) * 5,
    ]

    obj = obj * np.exp(-1j * np.angle(np.sum(obj[obj_slice])))
    assert np.isclose(0, np.angle(np.sum(obj[obj_slice])))

    # Then make a nonstandard version of them and standardize it
    # First, don't add a phase ramp and test
    test_probe = probe * 37.6 * np.exp(1j * 0.35)
    test_obj = obj / 37.6 * np.exp(1j * 1.43)
    s_probe, s_obj = analysis.standardize(test_probe, test_obj)
    assert np.allclose(probe, s_probe)
    assert np.allclose(obj, s_obj)

    # Test that it works on torch tensors
    s_probe, s_obj = analysis.standardize(
        t.as_tensor(test_probe, dtype=t.complex64),
        t.as_tensor(test_obj, dtype=t.complex64),
    )
    s_probe = s_probe.numpy()
    s_obj = s_obj.numpy()
    assert np.allclose(probe, s_probe)
    assert np.allclose(obj, s_obj)

    # Then do one with a phase ramp
    phase_ramp_dir = np.random.rand(2) - 0.5

    probe_Xs, probe_Ys = np.mgrid[: probe.shape[0], : probe.shape[1]]
    phase_ramp = np.exp(
        1j * probe_Ys * phase_ramp_dir[1] + 1j * probe_Xs * phase_ramp_dir[0]
    )
    test_probe = test_probe * phase_ramp

    obj_Xs, obj_Ys = np.mgrid[: obj.shape[0], : obj.shape[1]]
    obj_phase_ramp = np.exp(
        -1j * obj_Ys * phase_ramp_dir[1] + -1j * obj_Xs * phase_ramp_dir[0]
    )
    test_obj = test_obj * obj_phase_ramp

    s_probe, s_obj = analysis.standardize(test_probe, test_obj, correct_ramp=True)

    assert np.max(s_probe - probe) / np.max(np.abs(probe)) < 1e-4
    assert np.max(s_obj - obj) / np.max(np.abs(obj)) < 1e-4

    # Finally a test with the phase ramp and multiple probes
    subdominant_probe = (0.1 * np.random.rand(230, 240) * np.exp(1j * (np.random.rand(230, 240) - 0.5)))
    subdominant_probe = subdominant_probe * np.exp(-1j * np.angle(np.sum(subdominant_probe)))
    test_subdominant_probe = subdominant_probe * 37.6
    test_subdominant_probe = test_subdominant_probe * phase_ramp

    incoh_probe = np.array([test_probe, test_subdominant_probe])

    s_probe, s_obj = analysis.standardize(incoh_probe, test_obj, correct_ramp=True)

    assert np.max(s_probe[0] - probe) / np.max(np.abs(probe)) < 1e-4
    assert np.max(s_obj - obj) / np.max(np.abs(obj)) < 1e-4
    assert np.max(s_probe[1] - subdominant_probe) / np.max(np.abs(subdominant_probe)) < 1e-4


def test_synthesize_reconstructions():
    # I can only really test for a lack of failures, so I think my plan
    # will be to create a dataset that just needs to be added and see that
    # it successfully doesn't mess it up.

    # Start by making a probe and object that should meet the standardization
    # conditions
    probe = initializers.gaussian((230, 240), (20, 20), curvature=(0.01, 0.01)).numpy()
    probe = probe * np.sqrt(len(probe.ravel()) / np.sum(np.abs(probe) ** 2))
    probe = probe * np.exp(-1j * np.angle(np.sum(probe)))

    assert np.isclose(1, np.sum(np.abs(probe) ** 2) / len(probe.ravel()))
    assert np.abs(np.angle(np.sum(probe))) < 2e-7

    obj = 30 * np.random.rand(230, 240) * np.exp(1j * (np.random.rand(230, 240) - 0.5))
    obj_slice = np.s_[
        (obj.shape[0] // 8) * 3:(obj.shape[0] // 8) * 5,
        (obj.shape[1] // 8) * 3:(obj.shape[1] // 8) * 5,
    ]

    obj = obj * np.exp(-1j * np.angle(np.sum(obj[obj_slice])))
    assert np.isclose(0, np.angle(np.sum(obj[obj_slice])))

    # Now I make stacks of identical probes and objects
    probes = [probe, probe, probe, probe]
    probe = np.copy(probe)
    objects = [obj, obj, obj, obj]
    obj = np.copy(obj)

    s_probe, s_obj, obj_stack = analysis.synthesize_reconstructions(probes, objects)
    assert np.max(s_probe - probe) < 2e-5
    assert np.max(s_obj - obj) < 2e-5
    for t_obj in obj_stack:
        assert np.max(t_obj - obj) < 5e-5


def test_calc_consistency_prtf():

    # Create an object with a specific structure
    obj = (30 * np.random.rand(1030, 1040) * np.exp(1j * (np.random.rand(1030, 1040) - 0.5)))

    #
    synth_obj = np.sqrt(0.7) * obj

    obj_stack = [obj, obj, obj, obj]

    basis = np.array([[0, 2, 0], [3, 0, 0]])

    freqs, prtf = analysis.calc_consistency_prtf(synth_obj, obj_stack, basis)
    assert np.allclose(prtf, 0.7)

    freqs, prtf = analysis.calc_consistency_prtf(synth_obj, obj_stack, basis, nbins=30)
    assert np.allclose(prtf, 0.7)

    # Check that it also works with torch input
    t_synth_obj = t.as_tensor(synth_obj)
    t_obj_stack = [t.as_tensor(obj) for obj in obj_stack]
    freqs, prtf = analysis.calc_consistency_prtf(
        t_synth_obj, t_obj_stack, basis, nbins=30
    )
    assert np.allclose(prtf.numpy(), 0.7)

    # And also when the basis is in torch
    t_synth_obj = t.as_tensor(synth_obj)
    t_obj_stack = [t.as_tensor(obj) for obj in obj_stack]
    freqs, prtf = analysis.calc_consistency_prtf(
        t_synth_obj, t_obj_stack, t.Tensor(basis), nbins=30
    )
    assert np.allclose(prtf.numpy(), 0.7)

    # Check that is uses the right number of bins
    assert len(prtf) == 30
    assert len(freqs) == 30

    # Check that the maximum frequency is correct for the basis
    assert np.isclose(freqs[-1] - freqs[-2] + freqs[-1], np.sqrt(1 / 4**2 + 1 / 6**2))


def test_calc_deconvolved_cross_correlation():

    obj1 = np.random.rand(200, 300) + 1j * np.random.rand(200, 300)
    obj2 = np.random.rand(200, 300) + 1j * np.random.rand(200, 300)

    cor_fft = np.fft.fft2(obj1) * np.conj(np.fft.fft2(obj2))

    # Not sure if this is more or less stable than just the correlation
    # maximum - requires some testing
    np_cor = np.fft.ifft2(cor_fft / np.abs(cor_fft))
    # test with numpy inputs
    test_cor = analysis.calc_deconvolved_cross_correlation(
        obj1, obj2, im_slice=np.s_[:, :]
    )
    assert np.allclose(test_cor, np_cor)

    # test with pytorch inputs
    obj1_t = t.as_tensor(obj1)
    obj2_t = t.as_tensor(obj2)
    test_cor_t = analysis.calc_deconvolved_cross_correlation(
        obj1_t, obj2_t, im_slice=np.s_[:, :]
    )

    assert np.allclose(test_cor_t.numpy(), np_cor)


def test_calc_frc():

    obj1 = np.random.rand(270, 230) + 1j * np.random.rand(270, 230)
    obj2 = np.random.rand(270, 230) + 1j * np.random.rand(270, 230)

    basis = np.array([[0, 2, 0], [3, 0, 0]])

    nbins = 100
    snr = 2

    cor_fft = np.fft.fftshift(np.fft.fft2(obj1[10:-10, 20:-20])) * np.fft.fftshift(
        np.conj(np.fft.fft2(obj2[10:-10, 20:-20]))
    )

    F1 = np.abs(np.fft.fftshift(np.fft.fft2(obj1[10:-10, 20:-20]))) ** 2
    F2 = np.abs(np.fft.fftshift(np.fft.fft2(obj2[10:-10, 20:-20]))) ** 2

    di = np.linalg.norm(basis[:, 0])
    dj = np.linalg.norm(basis[:, 1])

    i_freqs = np.fft.fftshift(np.fft.fftfreq(cor_fft.shape[0], d=di))
    j_freqs = np.fft.fftshift(np.fft.fftfreq(cor_fft.shape[1], d=dj))

    Js, Is = np.meshgrid(j_freqs, i_freqs)
    Rs = np.sqrt(Is**2 + Js**2)

    numerator, bins = np.histogram(Rs, bins=nbins, weights=cor_fft)
    denominator_F1, bins = np.histogram(Rs, bins=nbins, weights=F1)
    denominator_F2, bins = np.histogram(Rs, bins=nbins, weights=F2)
    n_pix, bins = np.histogram(Rs, bins=nbins)
    bins = bins[:-1]

    frc = numerator / np.sqrt(denominator_F1 * denominator_F2)
    # This moves from combined-image SNR to single-image SNR
    snr /= 2

    threshold = (snr + (2 * snr + 1) / np.sqrt(n_pix)) / (
        1 + snr + (2 * np.sqrt(snr)) / np.sqrt(n_pix)
    )

    test_bins, test_frc, test_threshold = analysis.calc_frc(
        obj1,
        obj2,
        basis,
        im_slice=np.s_[10:-10, 20:-20],
        nbins=100,
        snr=2,
        limit="corner",
    )

    assert np.allclose(bins, test_bins)
    assert np.allclose(frc, test_frc)
    assert np.allclose(threshold, test_threshold)

    # try again with complex
    obj1_torch = t.as_tensor(obj1)
    obj2_torch = t.as_tensor(obj2)
    basis_torch = t.tensor(basis)

    test_bins_t, test_frc_t, test_threshold_t = analysis.calc_frc(
        obj1_torch,
        obj2_torch,
        basis_torch,
        im_slice=np.s_[10:-10, 20:-20],
        nbins=100,
        snr=2,
        limit="corner",
    )

    assert np.allclose(bins, test_bins_t.numpy())
    assert np.allclose(frc, test_frc_t.numpy())
    assert np.allclose(threshold, test_threshold_t.numpy())


def test_calc_rms_error():
    field_1 = t.rand(14, 19, dtype=t.complex64)
    field_2 = t.rand(14, 19, dtype=t.complex64)

    # Check that the calculation is insensitive to phase
    assert t.allclose(
        analysis.calc_rms_error(field_1, field_2),
        analysis.calc_rms_error(field_1, np.exp(0.7j) * field_2),
    )

    # And that it is sensitive to phase if we turn off the
    assert not t.allclose(
        analysis.calc_rms_error(field_1, field_2, align_phases=False),
        analysis.calc_rms_error(field_1, np.exp(0.7j) * field_2, align_phases=False),
    )

    # Check that the result is positive
    assert analysis.calc_rms_error(field_1, field_2) > 0

    # And that it is a smaller number with align_phases on
    assert analysis.calc_rms_error(field_1, field_2) <= analysis.calc_rms_error(
        field_1, field_2, align_phases=False
    )

    # Now we check against an explicit implementation:
    gamma = field_1 * t.conj(field_2)
    gamma /= t.abs(gamma)

    # This is an alternate way of doing the calculation. Actually, would this
    # be a better implementation anyway? Probably no difference tbh.
    rms_error_nophase = t.sqrt(
        (
            t.mean(t.abs(field_1) ** 2)
            + t.mean(t.abs(field_2) ** 2)
            - 2 * t.abs(t.mean(field_1 * t.conj(field_2)))
        )
    )
    assert t.allclose(rms_error_nophase, analysis.calc_rms_error(field_1, field_2))

    rms_error_phase = t.sqrt(
        (
            t.mean(t.abs(field_1) ** 2)
            + t.mean(t.abs(field_2) ** 2)
            - 2 * t.real(t.mean(field_1 * t.conj(field_2)))
        )
    )

    assert t.allclose(
        rms_error_phase, analysis.calc_rms_error(field_1, field_2, align_phases=False)
    )

    # Now let's test that it works along a dimension:

    field_1 = t.rand(3, 14, 19, dtype=t.complex64)
    field_2 = t.rand(3, 14, 19, dtype=t.complex64)
    result = analysis.calc_rms_error(field_1, field_2, normalize=True)
    assert result.shape == t.Size([3])

    for i in range(3):
        assert t.allclose(
            analysis.calc_rms_error(field_1[i], field_2[i], normalize=True), result[i]
        )


def test_calc_fidelity():

    fields_1 = t.rand(2, 30, 17, dtype=t.complex128)
    fields_2 = t.rand(3, 30, 17, dtype=t.complex128)

    dm_1 = t.reshape(fields_1, (2, -1))
    dm_1 = t.tensordot(dm_1.transpose(0, 1), dm_1.conj(), dims=1).numpy()
    dm_2 = t.reshape(fields_2, (3, -1))
    dm_2 = t.tensordot(dm_2.transpose(0, 1), dm_2.conj(), dims=1).numpy()

    sqrt_dm_1 = la.sqrtm(dm_1).astype(dm_1.dtype)
    inner_mat = la.sqrtm(np.dot(np.dot(sqrt_dm_1, dm_2), sqrt_dm_1))
    inner_mat = inner_mat.astype(dm_1.dtype)  # la.sqrtm doubles the precision
    fidelity = t.as_tensor(np.abs(np.trace(inner_mat)) ** 2)

    assert t.isclose(fidelity, analysis.calc_fidelity(fields_1, fields_2))

    # Check that it reduces to the overlap for coherent fields
    fields_1 = t.rand(1, 30, 17, dtype=t.complex128)
    fields_2 = t.rand(1, 30, 17, dtype=t.complex128)

    assert t.isclose(
        t.abs(t.sum(fields_1 * fields_2.conj())) ** 2,
        analysis.calc_fidelity(fields_1, fields_2),
    )
    # Checking that it works with extra dimensions
    fields_1 = t.rand(3, 3, 30, 17, dtype=t.complex128)
    fields_2 = t.rand(3, 1, 30, 17, dtype=t.complex128)
    field_3 = t.rand(1, 30, 17, dtype=t.complex128)

    fidelities = analysis.calc_fidelity(fields_1, fields_2)
    fidelities_2 = analysis.calc_fidelity(fields_1, field_3)
    for i in range(3):
        assert t.isclose(
            analysis.calc_fidelity(fields_1[i], fields_2[i]), fidelities[i]
        )
        assert t.isclose(analysis.calc_fidelity(fields_1[i], field_3), fidelities_2[i])

    # Check that the diensionality argument works
    fields_1 = t.rand(3, 2, 12, dtype=t.complex128)
    fields_2 = t.rand(3, 2, 12, dtype=t.complex128)

    assert analysis.calc_fidelity(fields_1, fields_2, dims=1).shape == t.Size([3])

    fields_1 = t.rand(3, 2, 12, 4, 5, dtype=t.complex128)
    fields_2 = t.rand(3, 2, 12, 4, 5, dtype=t.complex128)

    assert analysis.calc_fidelity(fields_1, fields_2, dims=3).shape == t.Size([3])


def test_calc_generalized_rms_error():

    # Test that it matches the rms error for coherent fields

    fields_1 = t.rand(1, 30, 17, dtype=t.complex128)
    fields_2 = t.rand(1, 30, 17, dtype=t.complex128)

    assert t.isclose(
        analysis.calc_generalized_rms_error(fields_1, fields_2),
        analysis.calc_rms_error(fields_1[0], fields_2[0], align_phases=True),
    )

    # Test that it is independent of field order
    fields_1 = t.rand(5, 30, 17, dtype=t.complex128)
    fields_2 = t.rand(3, 30, 17, dtype=t.complex128)
    fields_3 = fields_2.flip(0)

    assert t.isclose(
        analysis.calc_generalized_rms_error(fields_1, fields_2),
        analysis.calc_generalized_rms_error(fields_1, fields_3),
    )

    # Test with leading dimensions
    fields_1 = t.rand(3, 4, 2, 10, 17, dtype=t.complex128)
    fields_2 = t.rand(3, 4, 3, 10, 17, dtype=t.complex128)

    assert analysis.calc_generalized_rms_error(fields_1, fields_2).shape == t.Size(
        [3, 4]
    )

    # And test with different number of dimensions dims
    # Test that it is independent of field order
    fields_1 = t.rand(3, 6, 17, dtype=t.complex128)
    fields_2 = t.rand(3, 1, 17, dtype=t.complex128)
    fields_3 = fields_2.flip(0)

    assert analysis.calc_generalized_rms_error(
        fields_1, fields_2, dims=1
    ).shape == t.Size([3])
