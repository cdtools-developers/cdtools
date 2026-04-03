import cdtools
from cdtools.tools import plotting as p
from matplotlib import pyplot as plt
import numpy as np

# We load all three reconstructions
half_1 = cdtools.tools.data.h5_to_nested_dict(
    'example_reconstructions/gold_balls_half_1.h5')
half_2 = cdtools.tools.data.h5_to_nested_dict(
    'example_reconstructions/gold_balls_half_2.h5')
full = cdtools.tools.data.h5_to_nested_dict(
    'example_reconstructions/gold_balls_full.h5')

# This defines the region of recovered object to use for the analysis.
pad = 260
window = np.s_[pad:-pad, pad:-pad]

# This brings all three reconstructions to a common basis, correcting for
# possible global phase offsets, position shifts, and phase ramps. It also
# calculates a Fourier ring correlation and spectral signal-to-noise ratio
# estimate from the two half reconstructions.
results = cdtools.tools.analysis.standardize_reconstruction_set(
    half_1,
    half_2,
    full,
    window=window,
    nbins=40, # The number of bins to use for the FRC calculation
)

# We plot the normalized object images
p.plot_amplitude(results['obj_half_1'][window], basis=results['obj_basis'])
p.plot_phase(results['obj_half_1'][window], basis=results['obj_basis'])
p.plot_amplitude(results['obj_half_2'][window], basis=results['obj_basis'])
p.plot_phase(results['obj_half_2'][window], basis=results['obj_basis'])
p.plot_amplitude(results['obj_full'][window], basis=results['obj_basis'])
p.plot_phase(results['obj_full'][window], basis=results['obj_basis'])

# We plot the calculated Fourier ring correlation
plt.figure()
plt.plot(1e-6*results['frc_freqs'], results['frc'])
plt.plot(1e-6*results['frc_freqs'], results['frc_threshold'], 'k--')
plt.xlabel('Frequency (cycles / um)')
plt.ylabel('Fourier Ring Correlation')
plt.legend(['FRC', 'threshold'])

# We plot the calculated spectral signal-to-noise ratio
plt.figure()
plt.semilogy(1e-6*results['frc_freqs'], results['ssnr'])
plt.xlabel('Frequency (cycles / um)')
plt.ylabel('Spectral signal-to-noise ratio')
plt.grid('on')

plt.show()
