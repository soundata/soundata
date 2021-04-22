# soundata

<img src="docs/img/soundata.png" height="100px">

Common loaders for sound datasets. Find the API documentation [here](https://soundata.readthedocs.io/). Inspired by and based on [mirdata](https://github.com/mir-dataset-loaders/mirdata). (https://github.com/soundata/soundata)

[![CircleCI](https://circleci.com/gh/soundata/soundata.svg?style=svg)](https://circleci.com/gh/soundata/soundata)
[![codecov](https://codecov.io/gh/soundata/soundata/branch/master/graph/badge.svg)](https://codecov.io/gh/soundata/soundata)
[![Documentation Status](https://readthedocs.org/projects/soundata/badge/?version=latest)](https://soundata.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/soundata/soundata.svg)


This library provides tools for working with common sound datasets, including tools for:
* Downloading datasets to a common location and format
* Validating that the files for a dataset are all present 
* Loading annotation files to a common format
* Parsing clip-level metadata for detailed evaluations


### Installation

To install, simply run:

```python
pip install soundata
```

### Quick example
```python
import soundata

urbansound8k = soundata.initialize('urbansound8k')
urbansound8k.download()  # download the dataset
urbansound8k.validate()  # validate that all the expected files are there

example_clip = urbansound8k.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data
```
See the [documentation](https://soundata.readthedocs.io/) for more examples and the API reference.


### Currently supported datasets

* ESC-50
* TAU Urban Acoustic Scenes 2019
* TAU Urban Acoustic Scenes 2020 Mobile
* TUT Sound events 2017
* URBAN-SED
* UrbanSound8K
* More added soon!

For the **complete list** of supported datasets, see the [documentation](https://soundata.readthedocs.io/en/latest/source/quick_reference.html)


### Citing

TODO

```
paper
```

```
bibtex
```

When working with datasets, please cite the version of `soundata` that you are using (given by the `DOI` above) **AND** include the reference of the dataset,
which can be found in the respective dataset loader using the `cite()` method. 

### Contributing a new dataset loader

We welcome contributions to this library, especially new datasets. Please see [contributing](https://soundata.readthedocs.io/en/latest/source/contributing.html) for guidelines.
