import setuptools
import os
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

# read version from src/cdtools/_version.py
version_file = os.path.join("src/cdtools", "_version.py")
with open(version_file) as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
if not version_match:
    raise RuntimeError("Unable to find version string.")
version = version_match.group(1)
    
setuptools.setup(
    name="cdtools-py",
    version=version,
    python_requires='>3.8', # recommended minimum version for pytorch 2.3.0
    author="Abe Levitan",
    author_email="abraham.levitan@psi.ch",
    description="Tools for coherent diffractive imaging and ptychography",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cdtools-developers/cdtools",
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "matplotlib>=2.0", # 2.0 has better colormaps which are used by default
        "python-dateutil",
        "torch>=2.3.0", #2.3.0 is the earliest release for which L-BFGS works directly on complex-valued leaf tensors
        "h5py>=2.1"],
    extras_require={
        'tests': [
            "pytest",
            "pooch",
        ],
        'docs': [
            "sphinx>=4.3.0",
            "sphinx-argparse",
            "sphinx_rtd_theme>=0.5.1"
        ]
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

