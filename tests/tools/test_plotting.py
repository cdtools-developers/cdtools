import numpy as np
import torch as t
import scipy.datasets
import matplotlib.pyplot as plt

from cdtools.tools import plotting
from cdtools.tools import initializers


def test_plot_amplitude(show_plot):
    # Test with tensor
    im = t.as_tensor(scipy.datasets.ascent(), dtype=t.complex128)
    plotting.plot_amplitude(im, basis=np.array([[0, -1], [-1, 0], [0, 0]]), title='Test Amplitude')
    if show_plot:
        plt.show()

    # Test with numpy array
    im = scipy.datasets.ascent().astype(np.complex128)
    plotting.plot_amplitude(im, title='Test Amplitude')
    if show_plot:
        plt.show()


def test_plot_phase(show_plot):
    # Test with tensor
    im = initializers.gaussian([512, 512], [200, 200], amplitude=100, curvature=[.1, .1])
    plotting.plot_phase(im, title='Test Phase')
    if show_plot:
        plt.show()

    # Test with numpy array
    im = initializers.gaussian([512, 512], [200, 200], amplitude=100, curvature=[.1, .1]).numpy()
    plotting.plot_phase(im, title='Test Phase', basis=np.array([[0, -1], [-1, 0], [0, 0]]))
    if show_plot:
        plt.show()


def test_plot_colorized(show_plot):
    # Test with tensor
    gaussian = initializers.gaussian([512, 512], [200, 200], amplitude=100, curvature=[.1, .1])
    im = gaussian * t.as_tensor(scipy.datasets.ascent(), dtype=t.complex64)
    plotting.plot_colorized(im, title='Test Colorize', basis=np.array([[0, -1], [-1, 0], [0, 0]]))
    if show_plot:
        plt.show()

    # Test with numpy array
    im = im.numpy()
    plotting.plot_colorized(im, title='Test Colorize')
    if show_plot:
        plt.show()
