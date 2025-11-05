import numpy as np
import matplotlib.pyplot as plt

from cdtools.tools.analysis import line_based_frc


def create_stripe_test_pattern(shape, stripe_spacing, noise_level=0.1):
    """Create test pattern with anisotropic stripes"""
    h, w = shape

    # Create vertical stripes (varying along x-axis)
    x = np.arange(w)
    pattern = np.sin(2 * np.pi * x / stripe_spacing)

    # Broadcast to full image
    image = np.tile(pattern, (h, 1))

    # Add some modulation along y-axis for realism
    y = np.arange(h)
    y_mod = 1 + 0.3 * np.sin(2 * np.pi * y / (h // 3))
    image = image * y_mod[:, np.newaxis]

    # make the lower part 0
    image[7 * h // 8:] = 0

    # Add noise
    image += noise_level * np.random.randn(h, w)

    return image


# Create test images
shape = (200, 300)
stripe_spacing = 8  # pixels

unit = "nm"
pixel_size = 19.0  # nm, for example

# Create two slightly different versions to simulate repeated measurements
np.random.seed(42)
image1 = create_stripe_test_pattern(
    shape,
    stripe_spacing,
    noise_level=0.05,
)

np.random.seed(43)  # Different seed for second image
image2 = create_stripe_test_pattern(
    shape,
    stripe_spacing,
    noise_level=0.05,
)

# Display test images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(image1, cmap="gray")
ax1.set_title("Test Image 1")
ax1.axis("off")

ax2.imshow(image2, cmap="gray")
ax2.set_title("Test Image 2")
ax2.axis("off")
plt.tight_layout()


frc_thresholds = {"1/2": 0.5, "1/4": 0.25, "1/7": 1 / 7}

# Compute binned FRC
frequencies, frc_curve, resolution_dict = line_based_frc(
    image1, image2, axis=1, n_bins=100, thresholds=frc_thresholds,
    pixel_size=pixel_size, unit=unit
)

###########################
# Plot FRC curve with threshold crossings
###########################

# Color map for different thresholds
colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

# FRC curve with all thresholds
ax1.plot(frequencies, frc_curve, "k-", linewidth=3, label="FRC", zorder=10)

for i, (method, threshold_val) in enumerate(frc_thresholds.items()):
    color = colors[i % len(colors)]
    ax1.axhline(
        y=threshold_val,
        color=color,
        linestyle="--",
        alpha=0.8,
        label=f"{method} ({resolution_dict[method]:.3f} {unit})",
    )

    # Mark threshold frequency using the resolution value
    resolution = resolution_dict[method]
    if resolution is not None and resolution > 0:
        freq = pixel_size / resolution
        # Find the closest frequency in the array
        ax1.scatter(
            # threshold_val * np.ones_like(freq),
            freq,
            threshold_val,
            color=color,
            s=80,
            edgecolor="black",
            zorder=20,
        )

ax1.set_xlabel(f"Spatial Frequency (1/{unit})")
ax1.set_ylabel("FRC")
ax1.set_title("Line-based FRC Curve")
ax1.legend()
plt.tight_layout()
plt.show()
