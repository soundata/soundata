---
title: 'Soundata: Reproducible use of audio datasets'
tags:
  - Python
  - audio
  - dataset
  - urban-sound
  - environmental-sound
  - bioacoustics
authors:
  - name: Magdalena Fuentes
    corresponding: true
    orcid: 0000-0003-4506-6639
    affiliation: 1
  - name: Genís Plaja-Roglans
    orcid: 0000-0003-3450-3194
    affiliation: 2
  - name: Guillem Cortès-Sebastià
    orcid: 0000-0003-2827-8955
    affiliation: 2
  - name: Tanmay Khandelwal
    affiliation: 1
  - name: Marius Miron
    orcid: 0000-0002-2563-075X
    affiliation: 3
  - name: Xavier Serra
    affiliation: 2
  - name: Juan Pablo Bello
    affiliation: 1
  - name: Justin Salamon
    affiliation: 4     
affiliations:
 - name: New York University, New York, United States
   index: 1
 - name: Universitat Pompeu Fabra, Barcelona, Spain
   index: 2
 - name: Earth Species Project, Barcelona, Spain
   index: 3
 - name: Adobe Research, San Francisco, United States
   index: 4
date: 30 November 2023
bibliography: paper.bib

---

# Summary

`Soundata` is an open-source Python library for downloading, loading, and working with audio datasets in a programmatic and standardized way. It removes the need for writing custom loaders and improves reproducibility by providing tools to validate data against a canonical version. It speeds up research pipelines by allowing users to quickly download a dataset, validate that the dataset is complete and correct, and load it into memory in a standardized and reproducible way. It is designed to work with bioacoustics, environmental, urban, and spatial sound datasets; to be easy to use, easy to contribute to, and to increase reproducibility and standardize the usage of sound datasets in a flexible way.


# Statement of need

As research pipelines become increasingly complex, it is key that their different components are reproducible. In recent years, the machine listening research community has made considerable efforts towards standardization and reproducibility, by using common libraries for modeling and evaluation ([@tensorflow, @pedregosa2011scikit, @mesaros2016metrics](#)), open sourcing models ([@zinemanas2020dcase, @speechbrain](#)) and data dissemination using resources such as [Zenodo](https://zenodo.org). However, it has been previously shown that discrepancies in the local version of the data and different practices in loading and parsing datasets can lead to considerable differences in performance results, which is misleading when comparing methods ([@bittner2019mirdata](#)). Besides, from a practical point of view, it is extremely inefficient to develop pipelines from scratch for loading and parsing a dataset for each researcher or team each time, and this increases the chances of bugs and differences that hinder reproducibility.


There are other libraries that handle datasets, like `Tensorflow` ([@tensorflow](#)) or `DCASE-models` ([@zinemanas2020dcase](#)). However, using datasets in the context of those libraries makes it difficult to interchange models and software, as data loaders are designed to work with those environments and further adaptation is required to migrate them. Instead, we think that data should be handled separately, as a standalone library that can easily be plugged into different work pipelines, with different modeling software. Other alternatives such as Tensorflow-Datasets([@tensorflow_datasets](#)) or HuggingFace Datasets ([@lhoest2021datasets](#)), initially developed for other fields such as Natural Language Processing (NLP) datasets, started including computer vision as well as audio datasets. However, they do not standardize classes, so audio events can come in different formats in each dataset, requiring further work from the user each time. Having a community-centric, open-source, audio-specialized library allows us greater flexibility to incorporate more audio-specific API functionalities and align our priorities with those of the audio community.

`Soundata` was created following these design principles:

- **Easy to use**: Simplifies audio research pipelines considerably by having plug-and-play datasets in a standardized format. 

- **Easy to contribute to:** Users do not need to go through all the source code to contribute. `Soundata` provides extensive documentation explaining how to contribute a new loader.

- **Increase reproducibility:** `Soundata` provides a common framework for researchers to compare and validate their data. It also allows to easily propagate datasets' updates or fixes to the audio community, ensuring that methods are still comparable and researchers have the same up-to-date datasets' versions.

- **Standardize usage of sound datasets:** Standardizes common attributes of sound datasets such as audio or tags to simplify audio research pipelines, while preserving each dataset’s idiosyncrasies (e.g. if a dataset has ‘non-standard’ attributes, we include them as well).


`Soundata` is based on and inspired by `mirdata` with which shares goals and vision. `mirdata` ([@bittner2019mirdata](#)) is a `Python`[^python] package introduced as a tool for mitigating the lack of reproducibility and efficiency when working with datasets in the context of Music Information Research (MIR). In MIR, the mentioned issues are exacerbated due to the intrinsic commercial nature of music data, since it is very difficult to get licenses to distribute music recordings openly. Moreover, musical datasets are extremely complex compared to other audio datasets. Using the same software package for handling music and other audio datasets would lead to a very complex, hard-to-manage repository, which would be difficult to scale. Instead, we introduce `Soundata` as a separate effort that is inspired and based on `mirdata`. It specifically addresses the annotation types and formats required by communities like DCASE[^dcase], which work with bioacoustics, environmental, urban, and spatial sound datasets.

# Design Choices

`Soundata` has three main components, depicted in \autoref{fig:soundata_overview}, a ` core` module that implements the generic functions used by all the data loaders (e.g. Dataset), a `utils` module with the main utility functions such as downloading and validating the data, and the dataset `loaders` containing dataset-specific code to load and parse each dataset in a standardized way. Following this design, when a new dataset requires a new functionality, it is added to the core module so it can be used for similar loaders added later on.

![`Soundata`'s main components.\label{fig:soundata_overview}](images/architecture.png){ width=100% }

## Core

The core module forms the foundation of `Soundata`, encapsulating the primary `Dataset` class and essential functionalities such as annotations handling, dataset indexing, and dataset-level metadata. `Soundata` allows easy initialization of the available datasets, letting the user specify the home folder where the data is going to be stored. Attributes and paths specific to each audio clip, such as the audio itself, annotations and metadata are accessed using a class denoted `Clip`, included in the `Dataset` class. These attributes are parsed from a JSON file that acts as a dataset index, in which the dataset version, folder structure and md5 checksums (for checking consistency) are listed. 

## Utils 

This module offers utility functions for tasks like downloading, unzipping and validating datasets (i.e. making sure all files are present and not corrupted). It also contains utilities for converting `Soundata` annotations to `JAMS`[^jams] format. To validate the data integrity of a dataset, the utils module includes functions to perform md5 checksum comparison between the local copy and the official dataset release. 

## Loaders 

Dedicated to specific datasets, loaders are `Python` modules that instantiate core classes and implement specific loading functions for each dataset. Each dataset has its own loader, which contains the necessary code to convert the dataset audio, annotations and metadata to a standardized format. 


# Annotation Types

`Soundata` encompasses diverse annotation types that support a variety of tasks in bioacoustics, environmental, urban and spatial sound datasets. Further annotation types can be integrated into the library easily, but current annotation types include (see \autoref{fig:annotations}):

- **Tags**: Used typically for Acoustic Scene Classification (ASC), Sound Event Classification (SEC), and weak label Sound Event Detection (SED). These are essentially string labels with associated confidence values, spanning the full duration of the audio clip.
- **Events**: These annotations are for sound events with defined start times, end times, labels, and (optionally) confidence values. They are instrumental in strong label SED.
- **Spatial Events**: Spatial Events extends Events introducing additional attributes such as geographical coordinates (latitude, longitude), altitude, direction (azimuth and elevation), and distance from reference points. Spatial events are used for tasks such as Sound Event Detection and Localization (SELD).

![Annotation types included in `Soundata`.\label{fig:annotations}](images/annotation_types.png){ width=100% }


Annotation types in `Soundata` ensure compatibility with existing evaluation libraries from the DCASE community such as `sed_eval`, and are convertible to the `JAMS` format. `Soundata` can also be easily used together with `TensorFlow` and `PyTorch` ([@pytorch](#)). The "Example usage" section contains examples of how to do it.


# Supported Soundscapes and Tasks 

`Soundata` is designed to support a wide range of audio research tasks and soundscapes (or auditory domains) by providing a standardized interface for interacting with diverse audio datasets. Figure \autoref{fig:tasks} shows the tasks currently supported by `Soundata`.


![Audio tasks supported by `Soundata` as of today. \label{fig:tasks}](images/annotations_tasks.pdf){ width=100% }


## Example usage \label{sec:example_usage}

`Soundata` is designed to be user-friendly, so that users can start working with audio datasets right away after following a few steps, as summarized in \autoref{fig:supported_datasets}. 

![How to work with any supported dataset in `Soundata`.\label{fig:supported_datasets}](images/download.pdf){ width=100% }

Once the dataset is downloaded and validated, `Soundata` can be integrated into an audio research pipeline easily. Users can quickly explore the contents of the dataset by doing ```dataset.explore_dataset()```. The following code shows an example of how to get any SED dataset into a deep learning pipeline using `Soundata` and `Tensorflow`. 

```python
import soundata
import tensorflow as tf

def data_generator(dataset_name):
    dataset = soundata.initialize(dataset_name)
    dataset.download()  # Download dataset if needed
    for clip_id in dataset.clip_ids:
      clip = dataset.clip(clip_id)
      # Assume sample rate consistency or handle as needed
      audio_signal, _ = clip.audio
      if clip.tags.labels:
        label = clip.tags.labels[0]
      else:
        label = "Unknown"
      yield audio_signal.astype("float32"), label

# Create a Tensorflow dataset
tf_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator("urbansound8k"),
    output_types=(tf.float32, tf.string)
)

# Example: Iterate through the dataset
for audio, label in tf_dataset.take(1):
    print("Audio Shape:", audio.shape)
    print("Label:", label)
```


## Contributing \label{sec:contributing}

Contribution to `Soundata` is highly encouraged. To facilitate the process, `Soundata` provides an exhaustive contributing guide[^contributing] available in the `Soundata` documentation with all the necessary information on how to contribute. The contributing guide is maintained alongside the rest of the project, ensuring that it is up to date. The most common contribution in `Soundata` is the creation of new dataset loaders, as they play a crucial role in advancing `Soundata`'s objective of accommodating as many datasets as possible. \autoref{fig:contributing} summarizes the process of creating a new loader.

![Steps for contributing a dataset loader to `Soundata`. \label{fig:contributing}](images/contributing.pdf){ width=100% }


# Acknowledgements

We extend our sincere gratitude to all the contributors who have been invaluable in the development of this library. Their dedication and hard work have been instrumental in bringing `Soundata` to life and making it a valuable resource for the audio research community. We deeply appreciate contributions and look forward to continued collaboration and growth.


# References

[^python]: [https://www.python.org/](https://www.python.org/)
[^dcase]: [https://dcase.community/](https://dcase.community/)
[^jams]: [https://github.com/marl/jams](https://github.com/marl/jams)
[^supported_datasets]: [https://soundata.readthedocs.io/en/latest/source/quick_reference.html](https://soundata.readthedocs.io/en/latest/source/quick_reference.html)
[^contributing]: [https://soundata.readthedocs.io/en/latest/source/contributing.html](https://soundata.readthedocs.io/en/latest/source/contributing.html)
[^pr]: [https://github.com/soundata/soundata/pulls](https://github.com/soundata/soundata/pulls)