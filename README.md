# soundata

<img src="docs/img/soundata.png" height="100px">

Python library for downloading, loading & working with sound datasets. Find the API documentation [here](https://soundata.readthedocs.io/). <br/>
Inspired by and based on [mirdata](https://github.com/mir-dataset-loaders/mirdata). (https://github.com/soundata/soundata)

[![CircleCI](https://circleci.com/gh/soundata/soundata.svg?style=svg)](https://circleci.com/gh/soundata/soundata)
[![codecov](https://codecov.io/gh/soundata/soundata/branch/master/graph/badge.svg)](https://codecov.io/gh/soundata/soundata)
[![Documentation Status](https://readthedocs.org/projects/soundata/badge/?version=latest)](https://soundata.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/soundata/soundata.svg)


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


### Citing


```
@misc{fuentes_salamon2021soundata,
      title={Soundata: A Python library for reproducible use of audio datasets}, 
      author={Magdalena Fuentes and Justin Salamon and Pablo Zinemanas and Martín Rocamora and 
      Genís Plaja and Irán R. Román and Marius Miron and Xavier Serra and Juan Pablo Bello},
      year={2021},
      eprint={2109.12690},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```


When working with datasets, please cite the version of `soundata` that you are using **AND** include the reference of the dataset,
which can be found in the respective dataset loader using the `cite()` method. 

### Contributing a new dataset loader

We welcome and encourage contributions to this library, especially new datasets. Please see [contributing](https://soundata.readthedocs.io/en/latest/source/contributing.html) for guidelines.
