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

As research pipelines become increasingly complex, it is key that their different components are reproducible. In recent years, the machine listening research community has made considerable efforts towards standardization and reproducibility, by using common libraries for modeling and evaluation ([@tensorflow, @pedregosa2011scikit, @chollet2015keras, @mesaros2016metrics](#)), open sourcing models ([@zinemanas2020dcase, @speechbrain](#)) and data dissemination using resources such as [Zenodo](https://zenodo.org). However, it has been previously shown that discrepancies in the local version of the data and different practices in loading and parsing datasets can lead to considerable differences in performance results, which is misleading when comparing methods ([@bittner2019mirdata](#)). Besides, from a practical point of view, it is extremely inefficient to develop pipelines from scratch for loading and parsing a dataset for each researcher or team each time, and this increases the chances of bugs and differences that hinder reproducibility.


There are other libraries that handle datasets, like `Tensorflow` ([@tensorflow](#)) or `DCASE-models` ([@zinemanas2020dcase](#)). However, using datasets in the context of those libraries makes it difficult to interchange models and software, as data loaders are designed to work with those environments and further adaptation is required to migrate them. Instead, we think that data should be handled separately, as a standalone library that can easily be plugged into different work pipelines, with different modeling software. Other alternatives such as Tensorflow-Datasets([@tensorflow_datasets](#)) or HuggingFace Datasets ([@lhoest2021datasets](#)), initially developed for other fields such as Natural Language Processing (NLP) datasets, started including computer vision as well as audio datasets. However, they do not standardize classes, so audio events can come in different formats in each dataset, requiring further work from the user each time. Having a community-centric, open-source, audio-specialized library allows us greater flexibility to incorporate more audio-specific API functionalities and align our priorities with those of the audio community.

`Soundata` was created following these design principles:

- **Easy to use**: Simplifies audio research pipelines considerably by having plug-and-play datasets in a standardized format. 

- **Easy to contribute to:** Users do not need to go through all the source code to contribute. `Soundata` provides extensive documentation explaining how to contribute a new loader.

- **Increase reproducibility:** `Soundata` provides a common framework for researchers to compare and validate their data. It also allows to easily propagate datasets' updates or fixes to the audio community, ensuring that methods are still comparable and researchers have the same up-to-date datasets' versions.

- **Standardize usage of sound datasets:** Standardizes common attributes of sound datasets such as audio or tags to simplify audio research pipelines, while preserving each dataset’s idiosyncrasies (e.g. if a dataset has ‘non-standard’ attributes, we include them as well).


`Soundata` is based on and inspired by `mirdata` with which shares goals and vision. `mirdata` ([@bittner2019mirdata](#)) is a `Python`[^python] package introduced as a tool for mitigating the lack of reproducibility and efficiency when working with datasets in the context of Music Information Research (MIR). In MIR, the mentioned issues are exacerbated due to the intrinsic commercial nature of music data, since it is very difficult to get licenses to distribute music recordings openly. Moreover, musical datasets are extremely complex compared to other audio datasets. They usually convey much more metadata information (e.g. artists, musicians, instruments) and their annotations are of several different types and formats, reflecting the different tasks that there are in MIR such as melody estimation, beat tracking, and chord estimation. Using the same software package for handling music and other audio datasets would lead to a very complex, hard-to-manage repository, which would be difficult to scale. Instead, we introduce `Soundata` as a separate effort that is inspired and based on `mirdata`. It specifically addresses the annotation types and formats required by communities like DCASE[^dcase], which work with bioacoustics, environmental, urban, and spatial sound datasets.

# Design Choices

`Soundata` has three main components, depicted in \autoref{fig:soundata_overview}, a ` core` module that implements the generic functions used by all the data loaders (e.g. Dataset), a `utils` module with the main utility functions such as downloading and validating the data, and the dataset `loaders` containing dataset-specific code to load and parse each dataset in a standardized way. Following this design, when a new dataset requires a new functionality, it is added to the core module so it can be used for similar loaders added later on.

![`Soundata`'s main components.\label{fig:soundata_overview}](images/architecture.png){ width=100% }

## Core

The core module forms the foundation of `Soundata`, encapsulating the primary `Dataset` class and essential functionalities such as annotations handling, dataset indexing, and dataset-level metadata. `Soundata` allows easy initialization of the available datasets, letting the user specify the home folder where the data is going to be stored. Attributes and paths specific to each audio clip, such as the audio itself, annotations and metadata are accessed using a class denoted `Clip`, included in the `Dataset` class. These attributes are parsed from a JSON file that acts as a dataset index, in which the dataset version, folder structure and md5 checksums (for checking consistency) are listed. Such an index is used to ensure reproducibility by checking that different copies of a dataset are consistent and match the md5 hash keys. The contents of the datasets in `Soundata` are straightforwardly accessed using the specifically implemented methods in the library, including dataset information such as how to cite the dataset or plotting functionalities. 

## Utils 

This module offers utility functions for tasks like downloading, unzipping and validating datasets (i.e. making sure all files are present and not corrupted). It also contains utilities for converting `Soundata` annotations to `JAMS`[^jams] format. To validate the data integrity of a dataset, the utils module includes functions to perform md5 checksum comparison between the local copy and the official dataset release. One is also able to download just a part of a dataset if distributed in several splits, or even merging and unzipping the dataset that are multi-partly distributed due to an extensive size. This is all automatically performed by the downloading utils function. Utils support the main functionalities of the core classes.

## Loaders 

Dedicated to specific datasets, loaders are `Python` modules that instantiate core classes and implement specific loading functions for each dataset. Each dataset has its own loader, which contains the necessary code to convert the dataset audio, annotations and metadata to a standardized format. A loader will typically consist of code for loading the annotations of the dataset in one of `Soundata`'s supported annotation types, code for loading metadata, audio, and instantiating the functionalities from `core` and `utils` to serve that particular dataset. See the Contributing section of this document for an overview of the process of contributing with a loader to `Soundata`.


# Annotation Types

`Soundata` encompasses diverse annotation types that support a variety of tasks in bioacoustics, environmental, urban and spatial sound datasets. Further annotation types can be integrated into the library easily, but current annotation types include (see \autoref{fig:annotations}):

- **Tags**: Used typically for Acoustic Scene Classification (ASC), Sound Event Classification (SEC), and weak label Sound Event Detection (SED). These are essentially string labels with associated confidence values, spanning the full duration of the audio clip.
- **Events**: These annotations are for sound events with defined start times, end times, labels, and (optionally) confidence values. They are instrumental in strong label SED.
- **Spatial Events**: Spatial Events extends Events introducing additional attributes such as geographical coordinates (latitude, longitude), altitude, direction (azimuth and elevation), and distance from reference points. Spatial events are used for tasks such as Sound Event Detection and Localization (SELD).

![Annotation types included in `Soundata`.\label{fig:annotations}](images/annotation_types.png){ width=100% }


Annotation types in `Soundata` ensure compatibility with existing evaluation libraries from the DCASE community such as `sed_eval`, and are convertible to the `JAMS` format. `Soundata` can also be easily used together with `TensorFlow` and `PyTorch` ([@pytorch](#)). The "Example usage" section contains examples of how to do it.


# Supported Soundscapes and Tasks 

`Soundata` is designed to support a wide range of audio research tasks and soundscapes (or auditory domains) by providing a standardized interface for interacting with diverse audio datasets. 


![Audio tasks supported by `Soundata` as of today. \label{fig:tasks}](images/annotations_tasks.pdf){ width=100% }


Figure \autoref{fig:tasks} shows the tasks currently supported by `Soundata`, which include:

- **Sound Event Detection (SED)**: Is concerned with identifying the presence and duration of sound events within an audio stream. It uses both weak labels (Tags) for presence and strong labels (Events) for temporal localization of sound events.
- **Sound Event Localization (SEL)**: Involves determining the spatial location from where a sound originates within an environment. It goes beyond detection and classification to include the position in space relative to the listener or recording device.
- **Sound Event Classification (SEC)**: Categorizes sounds into predefined classes and involves analyzing audio to assign a category based on the type of sound event it contains, using Tags for the entire clip’s duration.
- **Acoustic Scene Classification (ASC)**: Classifies an entire audio stream into a scene category, characterizing the recording’s environment. Tags are used to indicate the single acoustic scene represented in the clip.
- **Audio Captioning (AC)**: Involves generating a textual description of the sound events and context within an audio clip. It is similar to image captioning but for audio content. This will involve the use of Tags or Events depending if the captions span the entire duration of the audio clip.

The library's modular design allows for the easy addition of new datasets spanning different tasks and soundscapes, and this list is in constant evolution. For the most up-to-date information, we refer the reader to the list of supported datasets[^supported_datasets].

Currently, `Soundata` includes five main auditory environments or soundscapes:

1. **Urban:** Sounds typically found in city environments (e.g. car, jackhammer).
2. **Environment:** Natural and ambient sounds from various ecosystems (e.g. running water).
3. **Machine:** Sounds originating from machinery and industrial activities (e.g. valve, fan).
4. **Bioacoustic:** Sounds produced by living organisms, particularly animals (e.g. a specific bird species).
5. **Music:** A range of musical compositions and performances (e.g. musical instruments). These are not MIR datasets, which are included in `mirdata` instead.

## Example usage \label{sec:example_usage}

`Soundata` is designed to be user-friendly, so that users can start working with audio datasets right away after following a few steps, as summarized in \autoref{fig:supported_datasets}. These steps are:

1. **Installation**: Begin by installing `Soundata` using the `Python` package manager `pip`. This ensures that all the necessary dependencies to work with the datasets are set up in the user's environment.

2. **Initialization of Dataset**: After installation, initialize the desired dataset using: ```dataset = soundata.initialize('dataset_name')```, replacing `'dataset_name'` with the actual name of the desired dataset. The list of all supported datasets can be displayed using ```soundata.list_datasets()```.

3. **Download Dataset**: With the dataset initialized, users can proceed to download it using ```dataset.download()```. The library manages the entire download process, including unzipping files, and handling multi-part downloads for large datasets.

4. **Validation**: To ensure the integrity and completeness of the dataset, validate it using ```dataset.validate()```. This method performs an md5 checksum comparison against a canonical version of the dataset, verifying that the data is correct and unchanged.

![How to work with any supported dataset in `Soundata`.\label{fig:supported_datasets}](images/download.pdf){ width=100% }

Once the dataset is downloaded and validated, `Soundata` can be integrated into an audio research pipeline easily. Users can use the `utils` module to quickly explore the contents of the dataset, by doing ```dataset.explore_dataset()```. This will trigger an interactive plot showing the dataset class distribution, some statistics such as the mean duration of its audio files, and the visualization of a random audio example which can be listened by the user.

The following code shows an example of how to get any SED dataset into a deep learning pipeline using `Soundata` and `Tensorflow`. 

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

Contribution to `Soundata` is highly encouraged. To facilitate the process, `Soundata` provides an exhaustive contributing guide[^contributing] available in the `Soundata` documentation with all the necessary information on how to contribute. The contributing guide is maintained alongside the rest of the project, ensuring that it is up to date.

Every contribution to `Soundata` is done via Pull Requests (PR) on GitHub[^pr]. The most common contribution in `Soundata` is the creation of new dataset loaders, as they play a crucial role in advancing `Soundata`'s objective of accommodating as many datasets as possible. \autoref{fig:contributing} summarizes the process of creating a new loader, which is explained below.

![Steps for contributing a dataset loader to `Soundata`. \label{fig:contributing}](images/contributing.pdf){ width=100% }

1. **Create Index**: The contributor has to first write a script that iterates over the data and creates an index to organize and manage the dataset files (examples are provided in the documentation and the GitHub repository). Indices contain identifiers and paths for audio files, annotations and metadata, their relative location in the dataset folder, and their checksums.
2. **Create Module**: The next step is to write a loader (i.e. a `.py` module) with functions to handle the logic of downloading, managing and loading the dataset audio and annotations. This is done by using `Soundata`'s core and utils functionalities, and examples are also provided in the documentation.
3. **Add tests**: Creating and running tests is required to contribute to `Soundata` to ensure the robustness of the library. `Soundata` has code coverage checks to make sure that every new loader is properly tested before incorporating into the repository. It is also necessary to run not only the loader-specific tests that the contributor writes, but also core `Soundata`'s tests too. These tackle OS compatibility, linting and formatting, among others.
4. **Write Docs**: To wrap up the new loader, it is important to provide descriptive documentation about the dataset task, size, contents and license, to help other users understand the loader and how they can interact with it. This documentation should be included in the loader module, and will be used to render `Soundata`'s documentation online.
5. **Submit Loader**: Finally, contributors will submit a pull request (PR) to the GitHub repository for review.


# Acknowledgements

We extend our sincere gratitude to all the contributors who have been invaluable in the development of this library. Their dedication and hard work have been instrumental in bringing `Soundata` to life and making it a valuable resource for the audio research community. We deeply appreciate contributions and look forward to continued collaboration and growth.


# References

[^python]: [https://www.python.org/](https://www.python.org/)
[^dcase]: [https://dcase.community/](https://dcase.community/)
[^jams]: [https://github.com/marl/jams](https://github.com/marl/jams)
[^supported_datasets]: [https://soundata.readthedocs.io/en/latest/source/quick_reference.html](https://soundata.readthedocs.io/en/latest/source/quick_reference.html)
[^contributing]: [https://soundata.readthedocs.io/en/latest/source/contributing.html](https://soundata.readthedocs.io/en/latest/source/contributing.html)
[^pr]: [https://github.com/soundata/soundata/pulls](https://github.com/soundata/soundata/pulls)