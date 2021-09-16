.. _tutorial:

###############
Getting started
###############

Installation
^^^^^^^^^^^^

To install Soundata simply do:

    .. code-block:: console

        pip install soundata
meeting

Soundata is easily imported into your Python code by:

.. code-block:: python

    import soundata


Initializing a dataset
^^^^^^^^^^^^^^^^^^^^^

Print a list of all available dataset loaders by calling:

.. code-block:: python

    import soundata
    print(soundata.list_datasets())

To use a loader, (for example, ``urbansound8k``) you need to initialize it by calling:

.. code-block:: python

    import soundata
    dataset = soundata.initialize('urbansound8k', data_home='choose_where_data_live')

You can indicate where the data would be stored and access by passing a path to ``data_home``, as explained below. Now ``us8k`` is a ``Dataset``
object containing common methods, described below.

Downloading a dataset
^^^^^^^^^^^^^^^^^^^^^

All dataset loaders in soundata have a ``download()`` function that allows the user to download the :ref:`canonical <faq>`
version of the dataset (when available). When initializing a dataset it is important to correctly set up the directory
where the dataset is going to be stored and retrieved.

Downloading a dataset into the default folder:
    In this first example, ``data_home`` is not specified. Thus, UrbanSound8K will be downloaded and retrieved from 
    the default folder, ``sound_datasets``, created in the user's root folder:

    .. code-block:: python

        import soundata
        dataset = soundata.initialize('urbansound8k')
        dataset.download()  # Dataset is downloaded into "sound_datasets" folder inside user's root folder

Downloading a dataset into a specified folder:
    In the next example ``data_home`` is specified, so UrbanSound8K will be downloaded and retrieved from the specified location:

    .. code-block:: python

        dataset = soundata.initialize('urbansound8k', data_home='Users/johnsmith/Desktop')
        dataset.download()  # Dataset is downloaded to John Smith's desktop



Partially downloading a dataset
    The ``download()`` function allows to partially download a dataset. In other words, if applicable, the user can
    select which elements of the dataset they want to download. Each dataset has a ``REMOTES`` dictionary were all
    the available downloadable elements are listed.

    ``tau2019uas`` has different elements as seen in the ``REMOTES`` dictionary. You can specify a subset of these elements to
    download by passing the ``download()`` function a list of the ``REMOTES`` keys that we are interested in via the 
    ``partial_download`` variable.

    .. admonition:: Example REMOTES
        :class: dropdown

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

    An partial download example for ``tau2019uas`` dataset could be:

    .. code-block:: python

        dataset = soundata.initialize('tau2019uas')
        dataset.download(partial_download=['development.audio.1', 'development.audio.2'])  # download only two remotes

Validating a dataset
^^^^^^^^^^^^^^^^^^^^

Using the ``validate()`` method you can ensure that the files in our local copy of a dataset are identical to the :ref:`canonical <faq>` version
of the dataset. The function computes the md5 checksum of every downloaded file to ensure it was downloaded correctly and isn't corrupted.

For big datasets: In future ``soundata`` versions, a random validation will be included. This improvement will reduce validation time for very big datasets.

Accessing annotations
^^^^^^^^^^^^^^^^^^^^^

You can choose a random clip from a dataset with the ``choice_clip()`` method.

.. admonition:: Example Index
    :class: dropdown

    .. code-block:: python

        random_clip = dataset.choice_clip()
        print(random_clip)
        >>> Clip(
                audio_path="/Users/theuser/sound_datasets/urbansound8k/audio/fold4/176638-5-0-1.wav",
                clip_id="176638-5-0-1",
                audio: The clip's audio

                        Returns,
                class_id: ,
                class_label: ,
                fold: ,
                freesound_end_time: ,
                freesound_id: ,
                freesound_start_time: ,
                salience: ,
                slice_file_name: ,
                tags: ,
            )



You can also access specific clips by id. The available clip ids can be acessed by doing ``dataset.clip_ids``.
In the next example we take the first clip id, and then we retrieve its ``tags``
annotation.

.. code-block:: python

    dataset = soundata.initialize('urbansound8k')
    us8k_ids = dataset.clip_ids  # the list of urbansound8k's clip ids
    us8k_clips = dataset.load_clips()  # Load all clips in the dataset
    example_clip = us8k_clips[us8k_ids[0]]  # Get the first clip

    # Accessing the clip's tags annotation
    example_tags = example_clip.tags


You can also load a single clip without loading all clips int the dataset:

.. code-block:: python

    us8k_ids = us8k.clip_ids  # the list of urbansound8k's clip ids
    example_clip = us8k.clip(us8k_ids[0])  # load this particular clip
    example_tags = example_clip.tags  # Get the tags for the first clip


.. _Remote Data Example: 

Accessing data remotely
^^^^^^^^^^^^^^^^^^^^^^^

Annotations can also be accessed through ``load_*()`` methods which may be useful, for instance, when your data aren't available locally. 
If you specify the annotation's path, you can use the module's loading functions directly. Let's
see an example.

.. admonition:: Accessing annotations remotely example
    :class: dropdown

    .. code-block:: python

        # Load list of clip ids of the dataset
        us8k_ids = dataset.clip_ids

        # Load a single clip, specifying the remote location
        example_clip = dataset.clip(us8k_ids[0], data_home='remote/data/path')
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


Annotation classes
^^^^^^^^^^^^^^^^^^

``soundata`` defines annotation-specific data classes such as `Tags` or `Events`. These data classes are meant to standarize the format for
all loaders, so you can use the same code with different datasets. The list and descriptions of available annotation classes can be found in :ref:`annotations`.

.. note:: These classes are standarized to the point that the data allows for it. In some cases where the dataset has
its own idiosincracies, the classes may be extended e.g. adding a customize attribute.

Iterating over datasets and annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general, most datasets are a collection of clips, and in most cases each clip has an audio file along with annotations.

With the ``load_clips()`` method, all clips are loaded as a dictionary with the clip ids as keys and 
clip objects as values. The clip objects include their respective audio and annotations, which are lazy-loaded on access
to keep things speedy and memory efficient. 

.. code-block:: python

    dataset = soundata.initialize('urbansound8k')
    for key, clip in dataset.load_clips().items():
        print(key, clip.audio_path)


Alternatively, you can loop over the ``clip_ids`` list to directly access each clip in the dataset.

.. code-block:: python

    us8k = soundata.initialize('urbansound8k')
    for clip_id in orchset.clip_ids:
        print(clip_id, us8k.clip(clip_id).audio_path)



.. _Including soundata in your pipeline:

Including soundata in your pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wanted to use ``urbansound8k`` to evaluate the performance of an urban sound classifier,
(in our case, ``random_classifier``), and then split the scores based on the metadata, you could do the following:

.. admonition:: soundata usage example
    :class: dropdown

    .. code-block:: python

        import sed_eval
        import soundata
        import numpy as np

        def random_classifier(audio_path):
            # do something
            return np.random.randint(2, size=len(num_classes))

        # Evaluate on the full dataset
        dataset = soundata.initialize("urbansound8k")
        scores = {}
        data = dataset.load_clips()
        for clip_id, clip_data in data.items():
            categorical_vector = random_classifier(track_data.audio_path_mono)

            ref_melody_data = track_data.melody
            ref_times = ref_melody_data.times
            ref_freqs = ref_melody_data.frequencies

            score = mir_eval.melody.evaluate(ref_times, ref_freqs, est_times, est_freqs)
            orchset_scores[track_id] = score

        # Split the results by composer and by instrumentation
        composer_scores = {}
        strings_no_strings_scores = {True: {}, False: {}}
        for track_id, track_data in orchset_data.items():
            if track_data.composer not in composer_scores.keys():
                composer_scores[track_data.composer] = {}

            composer_scores[track_data.composer][track_id] = orchset_scores[track_id]
            strings_no_strings_scores[track_data.contains_strings][track_id] = \
                orchset_scores[track_id]


This is the result of the example above.

.. admonition:: Example result
    :class: dropdown

    .. code-block:: python

        print(strings_no_strings_scores)
        >>> {True: {
                'Beethoven-S3-I-ex1':OrderedDict([
                    ('Voicing Recall', 1.0),
                    ('Voicing False Alarm', 1.0),
                    ('Raw Pitch Accuracy', 0.029798422436459245),
                    ('Raw Chroma Accuracy', 0.08063102541630149),
                    ('Overall Accuracy', 0.0272654370489174)
                    ]),
                'Beethoven-S3-I-ex2': OrderedDict([
                    ('Voicing Recall', 1.0),
                    ('Voicing False Alarm', 1.0),
                    ('Raw Pitch Accuracy', 0.009221311475409836),
                    ('Raw Chroma Accuracy', 0.07377049180327869),
                    ('Overall Accuracy', 0.008754863813229572)]),
                ...

                'Wagner-Tannhauser-Act2-ex2': OrderedDict([
                    ('Voicing Recall', 1.0),
                    ('Voicing False Alarm', 1.0),
                    ('Raw Pitch Accuracy', 0.03685636856368564),
                    ('Raw Chroma Accuracy', 0.08997289972899729),
                    ('Overall Accuracy', 0.036657681940700806)])
                }}

You can see that ``very_bad_melody_extractor`` performs very badly!

.. _Using soundata with tensorflow:

Using soundata with tensorflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a simple example of a generator that can be used to create a tensorflow Dataset.

.. admonition:: soundata with tf.data.Dataset example
    :class: dropdown

    .. code-block:: python

        import soundata
        import numpy as np
        import tensorflow as tf

        def urbansound8k_generator():
            # using the default data_home
            us8k = soundata.initialize("urbansound8k")
            clip_ids = us8k.clip_ids()
            for clip_id in clip_ids:
                clip = us8k.clip(clip_id)
                audio_signal, sample_rate = clip.audio
                yield {
                    "audio": audio_signal.astype(np.float32),
                    "sample_rate": sample_rate,
                    "label": clip.tags.labels[0],
                    "metadata": {"clip_id": clip.clip_id, "fold": clip.fold}
                }

        dataset = tf.data.Dataset.from_generator(
            urbansound8k_generator,
            {
                "audio": tf.float32,
                "sample_rate": tf.float32,
                "label": tf.string,
                "metadata": {'clip_id': tf.string, 'fold': tf.string}
            }
        )

In future ``soundata`` versions, generators for Tensorflow and PyTorch will be included out-of-the-box.
