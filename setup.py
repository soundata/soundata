""" Setup script for sound_datasets. """
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages

version_sfl = SourceFileLoader("soundata.version", "soundata/version.py")
version_module = version_sfl.load_module()

with open("README.md", "r") as fdesc:
    long_description = fdesc.read()

if __name__ == "__main__":
    setup(
        name="soundata",
        version=version_module.version,
        description="Python library for loading and working with sound datasets.",
        url="https://github.com/soundata/soundata",
        packages=find_packages(exclude=["test", "*.test", "*.test.*"]),
        download_url="http://github.com/soundata/soundata/releases",
        package_data={"soundata": ["datasets/indexes/*"]},
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python",
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Multimedia :: Sound/Audio :: Analysis",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8"
        ],
        keywords="sound dataset loader audio",
        license="BSD-3-Clause",
        install_requires=[
            "librosa>=0.10.0",
            "numpy>=1.21.6",
            "pandas>=1.3.5",
            "requests>=2.31.0",
            "tqdm>=4.65.0",
            "jams>=0.3.4",
            "flake8>=5.0.4",
            "mypy>=0.982",
            "py7zr>=0.16.0",
            "pydub>=0.25.1",
            "simpleaudio>=1.0.4",
            "seaborn>=0.11.2",
            "ipywidgets>=8.1.1",
        ],
        extras_require={
            "tests": [
                "pytest>=7.2.0",
                "pytest-cov>=4.1.0",
                "pytest-pep8>=1.0.6",
                "pytest-mock>=3.10.0",
                "pytest-localserver>=0.7.1",
                "pytest-xdist>=2.3.0",
                "future>=0.18.3",
                "coveralls>=3.3.1",
                "black>=23.3.0",
            ],
            "docs": [
                "docutils==0.16",
                "numpydoc",
                "recommonmark",
                "sphinx>=3.4.0",
                "sphinxcontrib-napoleon",
                "sphinx_rtd_theme",
                "sphinx-togglebutton"
            ],
        },
    )
