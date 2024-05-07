# soundata

<img src="docs/img/soundata.png" height="100px">

Python library for downloading, loading & working with sound datasets. Check the [API documentation](https://soundata.readthedocs.io/) and the [contributing instructions](https://soundata.readthedocs.io/en/latest/source/contributing.html). <br/>
For Music Information Retrieval (MIR) datasets please check [mirdata](https://github.com/mir-dataset-loaders/mirdata). 

![CI status](https://github.com/soundata/soundata/actions/workflows/ci.yml/badge.svg?branch=main)
![Formatting status](https://github.com/soundata/soundata/actions/workflows/formatting.yml/badge.svg?branch=main)
![Linting status](https://github.com/soundata/soundata/actions/workflows/lint-python.yml/badge.svg?branch=main)
[![Downloads](https://static.pepy.tech/badge/soundata)](https://pepy.tech/project/soundata)


[![codecov](https://codecov.io/gh/soundata/soundata/branch/master/graph/badge.svg)](https://codecov.io/gh/soundata/soundata)
[![Documentation Status](https://readthedocs.org/projects/soundata/badge/?version=latest)](https://soundata.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/soundata/soundata.svg)
[![PyPI version](https://badge.fury.io/py/soundata.svg)](https://badge.fury.io/py/soundata)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


This library provides tools for working with common sound datasets, including tools for:
* Downloading datasets to a common location and format
* Validating that the files for a dataset are all present 
* Loading annotation files to a common format
* Parsing clip-level metadata for detailed evaluations

Here's soundata's [list of currently supported datasets](https://soundata.readthedocs.io/en/latest/source/quick_reference.html).

### Installation

To install, simply run:

```python
pip install soundata
```

### Quick example
```python
import soundata

dataset = soundata.initialize('urbansound8k')
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data

```
See the [documentation](https://soundata.readthedocs.io/) for more examples and the API reference.


### Contributing a new dataset loader

We welcome and encourage contributions to this library, especially new dataset loaders. Please see [contributing](https://soundata.readthedocs.io/en/latest/source/contributing.html) for guidelines. Feel free to [open an issue](https://github.com/soundata/soundata/issues) if you have any doubt or your run into problems when working on the library.


### Citing


```
TBA
```


When working with datasets, please cite the version of `soundata` that you are using **AND** include the reference of the dataset, which can be found in the respective dataset loader using the `cite()` method. 
