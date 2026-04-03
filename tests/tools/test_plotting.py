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
    
    # Test with numpy array and an extra dimension
    im = np.stack([scipy.datasets.ascent().astype(np.complex128)]*3, axis=0)
    plotting.plot_amplitude(im, title='Test Amplitude')
    
    # Test with pytorch tensor and two extra dimensions
    im = t.as_tensor(np.stack([im]*5, axis=0))
    plotting.plot_amplitude(im, title='Test Amplitude',
                            additional_axis_labels=['Hi','There'])
    
    if show_plot:
        plt.show()
    plt.close('all')

def test_plot_phase(show_plot):
    # Test with tensor
    im = initializers.gaussian([512, 512], [200, 200], amplitude=100, curvature=[.1, .1])
    plotting.plot_phase(im, title='Test Phase')
    
    # Test with numpy array
    im = initializers.gaussian([512, 512], [200, 200], amplitude=100, curvature=[.1, .1]).numpy()
    plotting.plot_phase(im, title='Test Phase', basis=np.array([[0, -1], [-1, 0], [0, 0]]))
    
    if show_plot:
        plt.show()
    plt.close('all')


def test_plot_colorized(show_plot):
    # Test with tensor
    gaussian = initializers.gaussian([512, 512], [200, 200], amplitude=100, curvature=[.1, .1])
    im = gaussian * t.as_tensor(scipy.datasets.ascent(), dtype=t.complex64)
    plotting.plot_colorized(im, title='Test Colorize', basis=np.array([[0, -1], [-1, 0], [0, 0]]))

    # Test with numpy array and hsv
    im = im.numpy()
    plotting.plot_colorized(im, title='Test Colorize', use_cmocean=False)
    
    if show_plot:
        plt.show()
    plt.close('all')


def test_plot_translations(show_plot):
    rng = np.random.default_rng(0)
    trans_np = rng.uniform(-5e-6, 5e-6, (20, 2))
    trans_t = t.as_tensor(trans_np)

    # numpy, defaults
    plotting.plot_translations(trans_np)

    # torch tensor and reuse figure
    fig = plotting.plot_translations(trans_t)
    plotting.plot_translations(trans_np, lines=False, color='red', label='scan', fig=fig, clear_fig=False)
    
    if show_plot:
        plt.show()
    plt.close('all')

def test_plot_nanomap(show_plot):
    rng = np.random.default_rng(0)
    trans_np = rng.uniform(-5e-6, 5e-6, (20, 2))
    values_np = np.random.default_rng(1).uniform(0, 1, 20)
    trans_t = t.as_tensor(trans_np)
    values_t = t.as_tensor(values_np)

    # numpy, defaults
    plotting.plot_nanomap(trans_np, values_np)

    # torch tensors
    plotting.plot_nanomap(trans_t, values_t, units='nm', cmap_label='Intensity', convention='sample')
    
    if show_plot:
        plt.show()
    plt.close('all')
    

def test_plot_nanomap_with_images(show_plot):
    rng = np.random.default_rng(0)
    trans_np = rng.uniform(-5e-6, 5e-6, (20, 2))
    values_np = np.random.default_rng(1).uniform(0, 1, 20)
    # plot_nanomap_with_images requires tensor translations
    trans_t = t.as_tensor(trans_np)
    values_t = t.as_tensor(values_np)

    def get_image_2d(i):
        return np.random.default_rng(i).uniform(0, 1, (32, 32))

    def get_image_3d(i):
        return np.random.default_rng(i).uniform(0, 1, (4, 32, 32))

    # basic call, no values
    plotting.plot_nanomap_with_images(trans_np, get_image_2d)

    # with explicit values
    plotting.plot_nanomap_with_images(trans_t, get_image_2d, values=values_np)

    # 3D image stack
    fig = plt.figure(figsize=(11,7))
    plotting.plot_nanomap_with_images(trans_np, get_image_3d, values=values_t, fig=fig)
    
    if show_plot:
        plt.show()
    plt.close('all')
