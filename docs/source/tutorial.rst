.. _tutorial:

========
Tutorial
========

This tutorial will cover how to use `soundata` to access and work with audio datasets. `soundata` is a Python library designed to make it easy to load and work with common music information retrieval (MIR) datasets.

This tutorial will cover:

* Downloading Soundata
* Initializing a dataset
* Downloading a dataset
* Validating a dataset
* Loading clips
* Accessing annotations
* Advanced options for download and clips
* Usage examples of Soundata in your pipline, with Tensorflow, with Pytorch, and in Google Colab

-----

----------
Quickstart
----------

This section provides a step-by-step guide to quickly get started with the basic functionality. It covers the essential setup steps, initial configuration, and demonstrates the most common use cases to help you begin using the system effectively.

First, install soundata. We recommend to do this inside a conda or virtual environment for reproducibility.

.. code-block:: bash

    pip install mirdata

Then, get your data by simply doing:

.. code-block:: python
    :linenos:

    # Basic Usage Example
    import soundata

    # 1. List all available datasets
    print(soundata.list_datasets())

    # 2. Initialize a dataset loader
    dataset = soundata.initialize("urbansound8k", data_home='/choose/where/data/live')

    # 3. Download the dataset
    dataset.download()

    # 4. validate the dataset
    dataset.validate()

    # 5. Load clips
    random_clip = dataset.choice_clip()

    # 6. Access metadata and annotations
    print(random_clip)

Below, we elaborate on each step a bit more: 


Initializing a dataset
----------------------

To use a loader, (for example, ``urbansound8k``) you need to initialize it by calling:

.. code-block:: python

    dataset = soundata.initialize('urbansound8k', data_home='/choose/where/data/live')

This will create a dataset loader object that you can use to access the dataset's clips, metadata, and annotations.
You can specify the directory where the soundata data is stored by passing a path to ``data_home``.


.. admonition:: Dataset versions
    :class: attention

    Soundata supports working with multiple dataset versions.
    To see all available versions of a specific dataset, run ``soundata.list_dataset_versions('urbansound8k')``.
    Use ``version`` parameter if you wish to use a version other than the default one. To check an example, see below.

    .. toggle::

        .. code-block:: python

            # To see all available versions of a specific dataset:
            soundata.list_dataset_versions('urbansound8k')
            
            #Use 'version' parameter if you wish to use a version other than the default one.
            dataset = soundata.initialize('urbansound8k', data_home='/choose/where/data/live', version="1.0")

    

Downloading a dataset
----------------------

To download the dataset, you can use the ``download()`` method of the dataset loader object:

.. code-block:: python

    dataset.download()  # Dataset is downloaded to ~/sound_datasets/urbansound8k

By default, the dataset will be downloaded to the ``sound_datasets`` folder in your home directory.

.. admonition:: Note
    :class: attention

    For downloading in a custom folder, partial downloads, and other advanced options, see the `Advanced download options`_ section below.

Validating a dataset
--------------------

To ensure that the dataset files are correctly downloaded and not corrupted, you can use the ``validate()`` method of the dataset loader object:

.. code-block:: python

    dataset.validate()

This method checks the integrity of the dataset files and raises an error if any files are missing or corrupted.

Loading a random clip
----------------------

We can choose a random clip from a dataset with the ``choice_clip()`` method:

.. code-block:: python

    random_clip = dataset.choice_clip()

This returns a random clip from the dataset, which can be useful for testing or exploration purposes.

.. admonition:: Note
    :class: attention

    For loading all clips, load a single clip, or load clips with specific IDs, see the `Advanced clip options`_ section below.

Annotations 
-----------

After choosing a clip, we can access its annotations.
To print the annotations associated with the clip, you can simply print the clip object:

.. code-block:: python

    # For this example, we will use the random_clip from above.
    print(random_clip)

This will print the annotations associated with the clip.

.. code-block:: python

    # Example output
    >>> Clip(
          audio_path="~/sound_datasets/urbansed/audio/test/soundscape_test_bimodal73.wav",
          clip_id="soundscape_test_bimodal73",
          jams_path="~/sound_datasets/urbansed/annotations/test/soundscape_test_bimodal73.jams",
          txt_path="~/sound_datasets/urbansed/annotations/test/soundscape_test_bimodal73.txt",
          audio: The clips audio
                    * np.ndarray - audio signal
                    * float - sample rate,
          events: The audio events
                    * annotations.Events - audio event object,
          split: The data splits (e.g. train)
                    * str - split,
        )


.. admonition:: Annotation classes
    

    Soundata defines annotation-specific data classes such as `Tags` or `Events`. 
    These data classes are meant to standardize the format for all loaders, so you can use the same code with different datasets.
    The list and descriptions of available annotation classes can be found in :ref:`annotations`.
    
    **Note: These classes are standardized to the point that the data allow for it
    In some cases where the dataset has its own idiosyncrasies, the classes may be extended e.g. adding a customize, uncommon attribute.**

Accessing data remotely
-----------------------

Annotations can also be accessed through ``load_*()`` methods which may be useful, for instance, when your data aren't available locally. 
If you specify the annotation's path, you can use the module's loading functions directly. Let's
see an example.

.. admonition:: Accessing annotations remotely example

    .. code-block:: python

        # Load list of clip ids of the dataset
        ids = dataset.clip_ids

        # Load a single clip, specifying the remote location
        example_clip = dataset.clip(ids[0], data_home='remote/data/path')
        audio_path = example_clip.audio_path

        print(audio_path)
        >>> remote/data/path/audio/fold1/135776-2-0-49.wav
        print(os.path.exists(audio_path))
        >>> False

        # Write code here to download the remote path, e.g., to a temporary file.
        def my_downloader(remote_path):
            # the contents of this function will depend on where your data lives, and how permanently you
            # want the files to remain on your local machine. We point you to libraries handling common use cases below.
            # for data you would download via scp, you could use the [scp](https://pypi.org/project/scp/) library
            # for data on google drive, use [pydrive](https://pythonhosted.org/PyDrive/)
            # for data on google cloud storage use [google-cloud-storage](https://pypi.org/project/google-cloud-storage/)
            return local_path_to_downloaded_data

        # Get path to where your data live
        temp_path = my_downloader(audio_path)

        # Accessing the clip audio
        example_audio = dataset.load_audio(temp_path)


------


-------------------------
Advanced download options
-------------------------

This section provides comprehensive coverage of advanced dataset download configurations and options available in soundata:

* Downloading the dataset to a custom folder
* Partially downloading a dataset
* Downloading a multipart dataset
* Working with non-available datasets to openly download

Downloading dataset in custom folder
------------------------------------

.. code-block:: python

    dataset = soundata.initialize('urbansound8k', data_home='Users/johnsmith/Desktop')
    dataset.download()  # Dataset is downloaded to John Smith's desktop

Now ``data_home`` is specified and so urbansound8k will be read from / written to this custom location.

Partially downloading a dataset
-------------------------------

The ``download()`` function allows partial downloads of a dataset. In other words, if applicable, the user can
select which elements of the dataset they want to download. Each dataset has a ``REMOTES`` dictionary where all
the available elements are listed.

.. code-block:: python

    # Elements should be specified as a list of keys in the REMOTES dictionary.
    dataset.download(partial_download=['element_A', 'element_B', 'element_C'])

.. admonition:: Partial downloads example

    .. toggle::
    
        ``tau2019uas`` has different elements as seen in the ``REMOTES`` dictionary. You can specify a subset of these elements to
        download by passing the ``download()`` function a list of the ``REMOTES`` keys that we are interested in via the 
        ``partial_download`` variable.

        .. code-block:: python

            REMOTES = {
            "development.audio.1": download_utils.RemoteFileMetadata(
                filename="TAU-urban-acoustic-scenes-2019-development.audio.1.zip",
                url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.1.zip?download=1",
                checksum="aca4ebfd9ed03d5f747d6ba8c24bc728",
            ),
            "development.audio.2": download_utils.RemoteFileMetadata(
                filename="TAU-urban-acoustic-scenes-2019-development.audio.2.zip",
                url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.2.zip?download=1",
                checksum="c4f170408ce77c8c70c532bf268d7be0",
            ),
            "development.audio.3": download_utils.RemoteFileMetadata(
                filename="TAU-urban-acoustic-scenes-2019-development.audio.3.zip",
                url="https://zenodo.org/record/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.3.zip?download=1",
                checksum="c7214a07211f10f3250290d05e72c37e",
            ),
            ....

        A partial download example for ``tau2019uas`` dataset could be:

        .. code-block:: python

            dataset = soundata.initialize('tau2019uas')
            dataset.download(partial_download=['development.audio.1', 'development.audio.2'])  # download only two remotes

.. admonition:: Note
    :class: warning

    Not all datasets support partial downloads. To check if a dataset supports partial downloads, check if the ``REMOTES``
    dictionary is not empty.

Downloading a multipart dataset
-------------------------------

In some cases, datasets consist of multiple remote files that have to be extracted together locally to correctly recover the data.
In those cases, remotes that need to be extracted together should be grouped in a list, so all the necessary files are downloaded at once
(even in a partial download). An example of this is the `fsd50k` loader:

.. admonition:: Example multipart REMOTES

    .. code-block:: python

        REMOTES = {
            "FSD50K.dev_audio": [
                download_utils.RemoteFileMetadata(
                    filename="FSD50K.dev_audio.zip",
                    url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1",
                    checksum="c480d119b8f7a7e32fdb58f3ea4d6c5a",
                ),
                download_utils.RemoteFileMetadata(
                    filename="FSD50K.dev_audio.z01",
                    url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1",
                    checksum="faa7cf4cc076fc34a44a479a5ed862a3",
                ),
                download_utils.RemoteFileMetadata(
                    filename="FSD50K.dev_audio.z02",
                    url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1",
                    checksum="8f9b66153e68571164fb1315d00bc7bc",
                ),
                download_utils.RemoteFileMetadata(
                    filename="FSD50K.dev_audio.z03",
                    url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1",
                    checksum="1196ef47d267a993d30fa98af54b7159",
                ),
                download_utils.RemoteFileMetadata(
                    filename="FSD50K.dev_audio.z04",
                    url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1",
                    checksum="d088ac4e11ba53daf9f7574c11cccac9",
                ),
                download_utils.RemoteFileMetadata(
                    filename="FSD50K.dev_audio.z05",
                    url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1",
                    checksum="81356521aa159accd3c35de22da28c7f",
                ),
            ],
            ...
            
Working with non-available datasets to openly download
------------------------------------------------------

Some datasets are private and cannot be downloaded directly. 
In these cases, the download function will only retrieve the index file and any publicly available components (e.g., annotations). 
Users must obtain the private data separately, store it in their chosen ``data_home`` location, and initialize the dataset normally.

.. note::
    Private datasets may be available to the public upon request. If you are interested in a dataset that is not openly available, please contact the dataset authors or the dataset maintainers to request access.


------

---------------------
Advanced clip options
---------------------

This section covers advanced options for working with clips in datasets. These methods provide flexible ways to access and manipulate clip data based on your specific research needs:

* Loading all clips and example
* Loading clips with clip ID

Loading Clips
-------------

.. code-block:: python
    :linenos:

    # Initialize the dataset
    dataset = soundata.initialize('urbansound8k')

    # Load all clips in the dataset as a dictionary with the clip_ids as keys and clip objects as values
    clips = dataset.load_clips()

    # Iterate over dataset
    for key, clip in clips.items():
        print(key, clip.audio_path)

To load clips from a dataset, you can use the ``load_clips()`` method. This method returns a dictionary where the keys are clip IDs and the values are clip objects.

.. code-block:: python
    
    clips = dataset.load_clips()

This will load all clips in the dataset, allowing you to access their audio and annotations

Next, you can iterate over the clips dictionary to access each clip's audio path and other attributes.

.. code-block:: python

    for key, clip in clips.items():
        print(key, clip.audio_path)

Loading clips with clip ID
--------------------------

.. code-block:: python
    :linenos:

    # Initialize the dataset
    dataset = soundata.initialize('urbansound8k')

    # Get the list of clip IDs
    clip_ids = dataset.clip_ids

    # Iterate over the clip_ids list to directly access each track in the dataset
    for clip_id in clip_ids:
        print(clip_id, dataset.clip(clip_id).audio_path)

To load clips with clip ids, first

.. code-block:: python
    
    clip_ids = dataset.clip_ids

Get the list of the clip_ids.

Next, loop over the ``clip_ids`` list to directly access each clip in the dataset

.. code-block:: python

    for clip_id in clip_ids:
        print(clip_id, dataset.clip(clip_id).audio_path)


----


--------------
Advanced Usage
--------------

This section covers advanced usage examples of soundata, including integration with machine learning frameworks, dataset exploration tools, and cloud-based development environments:

* Using soundata in your pipeline
* Using Soundata with Tensorflow
* Using Soundata with pytorch
* Using Soundata to explore dataset
* Using Soundata in Google Colab


Using soundata in your pipeline
-------------------------------

If you wanted to use ``urbansound8k`` to evaluate the performance of an urban sound classifier,
(in our case, ``random_classifier``), and then split the scores based on the metadata, you could do the following:



.. code-block:: python
    :linenos:
    
    import sed_eval
    import soundata
    import numpy as np
    from dcase_util.containers import MetaDataContainer, ProbabilityContainer

    def random_classifier(classes):
        return [np.random.random(1)[0] for c in classes]

    # Evaluate on the full dataset
    dataset = soundata.initialize('urbansound8k')
    scores = {}
    data = dataset.load_clips()

    classes = np.unique([c for _, clip_data in data.items() for c in clip_data.tags.labels])
    fold = 2  # Choose a fold to evaluate

    ref_tags, est_tags, est_tag_probs = [], [], []
    for id, clip in data.items():
        if clip.fold == 2:
            ref_tags.append({'filename': id, 'tags': clip.tags.labels[0]})  # Urbansound8k has one label per clip
            probs = random_classifier(classes)
            for c, p in zip(classes, probs):
                est_tag_probs.append({'filename': id, 'label': c, 'probability': p},)
                if p > 0.5:  # Detection threshold of 0.5
                    est_tags.append({'filename': id, 'tags': [c]})

    tag_evaluator = sed_eval.audio_tag.AudioTaggingMetrics(tags=MetaDataContainer(ref_tags).unique_tags)
    tag_evaluator.evaluate(
        reference_tag_list=MetaDataContainer(ref_tags),
        estimated_tag_list=MetaDataContainer(est_tags),
        estimated_tag_probabilities=ProbabilityContainer(est_tag_probs))



.. admonition:: Example result
    
    .. toggle::

        This is the result of the example above:

        .. code-block:: python

            print(tag_evaluator)
            >>> Audio tagging metrics
            ========================================
            Tags                              : 10
            Evaluated units                   : 888

            Overall metrics (micro-average)
            ======================================
            F-measure
                F-measure (F1)                  : 9.57 %
                Precision                       : 9.57 %
                Recall                          : 9.57 %
            Equal error rate
                Equal error rate (EER)          : 51.01 %

            Class-wise average metrics (macro-average)
            ======================================
            F-measure
                F-measure (F1)                  : 6.47 %
                Precision                       : 7.54 %
                Recall                          : 9.33 %
            Equal error rate
                Equal error rate (EER)          : 50.95 %

            Class-wise metrics
            ======================================
                Tag               | Nref        Nsys      | F-score     Pre         Rec       | EER
                ----------------- | ---------   --------- | ---------   ---------   --------- | ---------
                air_conditioner   | 100         419       | 19.3%       11.9        50.0      | 49.0%
                car_horn          | 42          227       | 4.5%        2.6         14.3      | 54.8%
                children_playing  | 100         126       | 9.7%        8.7         11.0      | 54.0%
                dog_bark          | 100         58        | 13.9%       19.0        11.0      | 47.1%
                drilling          | 100         31        | 9.2%        19.4        6.0       | 52.4%
                engine_idling     | 100         16        | 1.7%        6.2         1.0       | 50.0%
                gun_shot          | 35          7         | 0.0%        0.0         0.0       | 48.1%
                jackhammer        | 120         1         | 0.0%        0.0         0.0       | 52.5%
                siren             | 91          3         | 0.0%        0.0         0.0       | 51.6%
                street_music      | 100         0         | nan%        nan         0.0       | 50.0%

Using soundata with tensorflow
------------------------------
The following is a simple example of a generator that can be used to create a tensorflow Dataset.


.. code-block:: python
    :linenos:

    import soundata
    import numpy as np
    import tensorflow as tf

    def data_generator(dataset_name):
        # using the default data_home
        dataset = soundata.initialize(dataset_name)
        ids = dataset.clip_ids()

        for clip_id in ids:
            clip = dataset.clip(clip_id)
            audio_signal, sample_rate = clip.audio
            
            yield {
                "audio": audio_signal.astype(np.float32),
                "sample_rate": sample_rate,
                "label": clip.tags.labels[0],
                "metadata": {"clip_id": clip.clip_id, "fold": clip.fold}
            }

    dataset = tf.data.Dataset.from_generator(
        data_generator('urbansound8k'),
        {
            "audio": tf.float32,
            "sample_rate": tf.float32,
            "label": tf.string,
            "metadata": {'clip_id': tf.string, 'fold': tf.string}
        }
    )




Using soundata with pytorch
---------------------------

This example shows how to create a custom PyTorch Dataset class that loads audio data from Soundata.


.. code-block:: python
    :linenos:

    import soundata
    import torch
    from torch.utils.data import DataLoader, Dataset

    class SoundataTorchDataset(Dataset):
        """A PyTorch Dataset for loading audio data from Soundata"""
        def __init__(self, ds, split:str):
            self.dataset = ds
            
            # Filter clips by split
            self.clip_ids = [
                clip_id for clip_id in self.dataset.clip_ids
                if self.dataset.clip(clip_id).split == split
            ]
            
        def __len__(self):
            return len(self.clip_ids)

        def __getitem__(self, idx):
            clip = self.dataset.clip(self.clip_ids[idx])
            audio, sr = clip.audio
            
            audio_tensor = torch.tensor(audio.T, dtype=torch.float32)
            
            return audio_tensor, clip.captions

    # Initialize, download and validate the dataset
    dataset = soundata.initialize(dataset_name="dcase23_task6b")
    dataset.download()
    dataset.validate()

    # Pass the dataset to the custom dataset class specifying the split
    dev_dataset = SoundataTorchDataset(dataset, split='dev')

    def custom_collate(batch):
        """Custom collate function to handle variable-length sequences"""
        pass

    # Create a Torch DataLoader providing the dataset and a custom collate function
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate 
    )

Using soundata to explore dataset
---------------------------------

The ``explore_dataset()`` function in ``soundata`` allows you to visualize various aspects of the dataset. This can be particularly useful for understanding the distribution of events and the nature of the audio data before proceeding with analysis or model training.

First, you must install the optinal dependencies to use the plot functionalities

.. code-block:: console

    pip install soundata"[plots]"

To explore the dataset, first initialize it and then call the ``explore_dataset()`` method:

.. code-block:: python
    :linenos:

    #import soundata
    import soundata

    # Initialize the dataset
    dataset = soundata.initialize('urbansound8k', data_home='your_data_directory')

    # Explore the dataset
    dataset.explore_dataset()


When you run this function, an interface will appear with several options, allowing you to choose what to plot.


.. image:: ../img/dataset_exp.png
    :alt: class dataset explorer
    :scale: 80%

.. admonition:: Examples
    
    To check more details and plots about Class Distribution, Statistics (Computational), Audio Visualization, click below

    .. toggle::

        **Class Distribution** displays the distribution of different event classes in the dataset.

        .. image:: ../img/class_dist.png
            :alt: class distribution plot example
            :scale: 50%

        |
        **Statistics (Computational)** provides computational statistics about the dataset (Time-consuming operation).

        .. image:: ../img/class_stat.png
            :alt: statistics plot example
            :scale: 50%

        |
        **Audio Visualization** offers visualizations related to the audio data, such as waveforms or spectrograms.

        .. image:: ../img/audio_plot.png
            :alt: audio visualization plot example
            :scale: 50%

        |
        By using the ``explore_dataset()`` function, you can gain a comprehensive overview of the dataset's structure and content, which is crucial for effective analysis and model building.

.. admonition:: note
    :class: attention

    If you try to load the visualizations without the optional dependencies, you will be thrown an exception indicating that the dependencies are missing.
    Please do install the optional dependencies using the command above in order to use the visualization functionalities.

    If you encounter any error during the installation of ``simpleaudio``, please visit `simpleaudio installation <https://simpleaudio.readthedocs.io/en/latest/installation.html>`__ guide and check the dependencies.


Using soundata in Google Colab
-----------------------------

`Google Colab` provides a browser-based Python environment with free GPU support, which is useful for exploring datasets quickly.
You will have two options that you can use the dataset from ``soundata`` in Colab - ``Download Dataset directly in Google Colab``, or ``Access the Dataset Downloaded out of Google Colab``

.. admonition:: Colab Example Notebook

    | For Google Colab Example Notebook, check the link here: `Google Colab Example Notebook <https://colab.research.google.com/github/yujin-kimmm/soundata_colab_example/blob/main/Soundata_colab_example.ipynb>`_.
    | If you are willing to use the notebook, you can make a copy of it to your Google Drive by clicking on ``File -> Save a copy in Drive``.