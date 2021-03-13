.. _tutorial:

########
Tutorial
########

Installation
------------

To install ``soundata``:

    .. code-block:: console

        pip install soundata

Usage
-----

``soundata`` is easily imported into your Python code by:

.. code-block:: python

    import soundata


Initializing a dataset
^^^^^^^^^^^^^^^^^^^^^^

Print a list of all available dataset loaders by calling:

.. code-block:: python

    import soundata
    print(soundata.list_datasets())

To use a loader, (for example, 'urbansound8k') you need to initialize it by calling:

.. code-block:: python

    import soundata
    us8k = soundata.initialize('urbansound8k')

Now ``us8k`` is a ``Dataset`` object containing common methods, described below.

Downloading a dataset
^^^^^^^^^^^^^^^^^^^^^

All dataset loaders in ``soundata`` have a ``download()`` function that allows the user to download the canonical
version of the dataset (when available). When initializing a dataset it is important to correctly set up the directory
where the dataset is going to be stored and retrieved.

Downloading a dataset into the default folder:
    In this first example, ``data_home`` is not specified. Thus, UrbanSound8K will be downloaded and retrieved from 
    the default folder, ``sound_datasets``, created in the user's root folder:

    .. code-block:: python

        import soundata
        us8k = soundata.initialize('urbansound8k')
        us8k.download()  # Dataset is downloaded into "sound_datasets" folder inside user's root folder

Downloading a dataset into a specified folder:
    In the next example ``data_home`` is specified, so UrbanSound8K will be downloaded and retrieved from the specified location:

    .. code-block:: python

        us8k = soundata.initialize('urbansound8k', data_home='Users/johnsmith/Desktop')
        us8k.download()  # Dataset is downloaded to John Smith's desktop

..
    Partially downloading a dataset
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    The ``download()`` function allows to partially download a dataset. In other words, if applicable, the user can
    select which elements of the dataset they want to download. Each dataset has a ``REMOTES`` dictionary were all
    the available downloadable elements are listed.

    ``cante100`` has different elements as seen in the ``REMOTES`` dictionary. We can specify a subset of these elements to
    download by passing the ``download()`` function a list of the ``REMOTES`` keys that we are interested in via the 
    ``partial_download`` variable.

    .. admonition:: Example REMOTES
        :class: dropdown

        .. code-block:: python

            REMOTES = {
                "spectrogram": download_utils.RemoteFileMetadata(
                    filename="cante100_spectrum.zip",
                    url="https://zenodo.org/record/1322542/files/cante100_spectrum.zip?download=1",
                    checksum="0b81fe0fd7ab2c1adc1ad789edb12981",  # the md5 checksum
                    destination_dir="cante100_spectrum",  # relative path for where to unzip the data, or None
                ),
                "melody": download_utils.RemoteFileMetadata(
                    filename="cante100midi_f0.zip",
                    url="https://zenodo.org/record/1322542/files/cante100midi_f0.zip?download=1",
                    checksum="cce543b5125eda5a984347b55fdcd5e8",  # the md5 checksum
                    destination_dir="cante100midi_f0",  # relative path for where to unzip the data, or None
                ),
                "notes": download_utils.RemoteFileMetadata(
                    filename="cante100_automaticTranscription.zip",
                    url="https://zenodo.org/record/1322542/files/cante100_automaticTranscription.zip?download=1",
                    checksum="47fea64c744f9fe678ae5642a8f0ee8e",  # the md5 checksum
                    destination_dir="cante100_automaticTranscription",  # relative path for where to unzip the data, or None
                ),
                "metadata": download_utils.RemoteFileMetadata(
                    filename="cante100Meta.xml",
                    url="https://zenodo.org/record/1322542/files/cante100Meta.xml?download=1",
                    checksum="6cce186ce77a06541cdb9f0a671afb46",  # the md5 checksum
                ),
                "README": download_utils.RemoteFileMetadata(
                    filename="cante100_README.txt",
                    url="https://zenodo.org/record/1322542/files/cante100_README.txt?download=1",
                    checksum="184209b7e7d816fa603f0c7f481c0aae",  # the md5 checksum
                ),
            }

    An partial download example for ``cante100`` dataset could be:

    .. code-block:: python

        cante100.download(partial_download=['spectrogram', 'melody', 'metadata'])

Validating a dataset
^^^^^^^^^^^^^^^^^^^^

Using the ``validate()`` method we can ensure that the files in our local copy of a dataset are identical to the canonical version
of the dataset. The function computes the md5 checksum of every downloaded file to ensure it was downloaded correctly and isn't corrupted.

For big datasets: In future ``soundata`` versions, a random validation will be included. This improvement will reduce validation time for very big datasets.

Accessing annotations
^^^^^^^^^^^^^^^^^^^^^

We can choose a random clip from a dataset with the ``choice_clip()`` method.

.. admonition:: Example Index
    :class: dropdown

    .. code-block:: python

        random_clip = us8k.choice_clip()
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


We can also access specific clips by id. 
The available clip ids can be acessed via the `.clip_ids` attribute.
In the next example we take the first clip id, and then we retrieve the tags
annotation.

.. code-block:: python

    us8k_ids = us8k.clip_ids  # the list of urbansound8k's clip ids
    us8k_data = us8k.load_clips()  # Load all clips in the dataset
    example_clip = us8k_data[us8k_ids[0]]  # Get the first clip

    # Accessing the clip's tags annotation
    example_tags = example_clip.tags


Alternatively, we don't need to load the whole dataset to get a single clip.

.. code-block:: python

    us8k_ids = us8k.clip_ids  # the list of orchset's track ids
    example_track = orchset.track(orchset_ids[0])  # load this particular track
    example_melody = example_track.melody  # Get the melody from first track


.. _Remote Data Example: 

Accessing data remotely
^^^^^^^^^^^^^^^^^^^^^^^

Annotations can also be accessed through ``load_*()`` methods which may be useful, for instance, when your data isn't available locally. 
If you specify the annotation's path, you can use the module's loading functions directly. Let's
see an example.

.. admonition:: Accessing annotations remotely example
    :class: dropdown

    .. code-block:: python

        # Load list of track ids of the dataset
        orchset_ids = orchset.track_ids

        # Load a single track, specifying the remote location
        example_track = orchset.track(orchset_ids[0], data_home='user/my_custom/remote_path')
        melody_path = example_track.melody_path

        print(melody_path)
        >>> user/my_custom/remote_path/GT/Beethoven-S3-I-ex1.mel
        print(os.path.exists(melody_path))
        >>> False

        # Write code here to locally download your path e.g. to a temporary file.
        def my_downloader(remote_path):
            # the contents of this function will depend on where your data lives, and how permanently you want the files to remain on the machine. We point you to libraries handling common use cases below.
            # for data you would download via scp, you could use the [scp](https://pypi.org/project/scp/) library
            # for data on google drive, use [pydrive](https://pythonhosted.org/PyDrive/)
            # for data on google cloud storage use [google-cloud-storage](https://pypi.org/project/google-cloud-storage/)
            return local_path_to_downloaded_data

        # Get path where youe data lives
        temp_path = my_downloader(melody_path)

        # Accessing to track melody annotation
        example_melody = orchset.load_melody(temp_path)

        print(example_melody.frequencies)
        >>> array([  0.   ,   0.   ,   0.   , ..., 391.995, 391.995, 391.995])
        print(example_melody.times)
        >>> array([0.000e+00, 1.000e-02, 2.000e-02, ..., 1.244e+01, 1.245e+01, 1.246e+01])



Annotation classes
^^^^^^^^^^^^^^^^^^

``soundata`` defines annotation-specific data classes. These data classes are meant to standarize the format for
all loaders, and are compatibly with `JAMS <https://jams.readthedocs.io/en/stable/>`_ and `mir_eval <https://craffel.github.io/mir_eval/>`_.

The list and descriptions of available annotation classes can be found in :ref:`annotations`.

.. note:: These classes may be extended in the case that a loader requires it.

Iterating over datasets and annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In general, most datasets are a collection of tracks, and in most cases each track has an audio file along with annotations.

With the ``load_tracks()`` method, all tracks are loaded as a dictionary with the ids as keys and 
track objects (which include their respective audio and annotations, which are lazy-loaded on access) as values.

.. code-block:: python

    orchset = soundata.initialize('orchset')
    for key, track in orchset.load_tracks().items():
        print(key, track.audio_path)


Alternatively, we can loop over the ``track_ids`` list to directly access each track in the dataset.

.. code-block:: python

    orchset = soundata.initialize('orchset')
    for track_id in orchset.track_ids:

        print(track_id, orchset.track(track_id).audio_path)


Basic example: including soundata in your pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we wanted to use ``orchset`` to evaluate the performance of a melody extraction algorithm
(in our case, ``very_bad_melody_extractor``), and then split the scores based on the
metadata, we could do the following:

.. admonition:: soundata usage example
    :class: dropdown

    .. code-block:: python

        import mir_eval
        import soundata
        import numpy as np
        import sox

        def very_bad_melody_extractor(audio_path):
            duration = sox.file_info.duration(audio_path)
            time_stamps = np.arange(0, duration, 0.01)
            melody_f0 = np.random.uniform(low=80.0, high=800.0, size=time_stamps.shape)
            return time_stamps, melody_f0

        # Evaluate on the full dataset
        orchset = soundata.initialize("orchset")
        orchset_scores = {}
        orchset_data = orchset.load_tracks()
        for track_id, track_data in orchset_data.items():
            est_times, est_freqs = very_bad_melody_extractor(track_data.audio_path_mono)

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a simple example of a generator that can be used to create a tensorflow Dataset.

.. admonition:: soundata with tf.data.Dataset example
    :class: dropdown

    .. code-block:: python

        import soundata
        import numpy as np
        import tensorflow as tf

        def orchset_generator():
            # using the default data_home
            orchset = soundata.initialize("orchset")
            track_ids = orchset.track_ids()
            for track_id in track_ids:
                track = orchset.track(track_id)
                audio_signal, sample_rate = track.audio_mono
                yield {
                    "audio": audio_signal.astype(np.float32),
                    "sample_rate": sample_rate,
                    "annotation": {
                        "times": track.melody.times.astype(np.float32),
                        "freqs": track.melody.frequencies.astype(np.float32),
                    },
                    "metadata": {"track_id": track.track_id}
                }

        dataset = tf.data.Dataset.from_generator(
            orchset_generator,
            {
                "audio": tf.float32,
                "sample_rate": tf.float32,
                "annotation": {"times": tf.float32, "freqs": tf.float32},
                "metadata": {'track_id': tf.string}
            }
        )

In future ``soundata`` versions, generators for Tensorflow and Pytorch will be included.
