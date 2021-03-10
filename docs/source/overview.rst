.. _overview:

########
Overview
########

.. code-block::

    pip install soundata


``soundata`` is a library which aims to standardize how audio datasets are accessed in Python,
removing the need for writing custom loaders in every project, and improving reproducibility.
Working with datasets usually requires an often cumbersome step of downloading data and writing 
load functions that load related files (for example, audio and annotations)
into a standard format to be used for experimenting or evaluating. ``soundata`` does all of this for you:

.. code-block:: Python

    import soundata

    print(soundata.list_datasets())

    dataset = soundata.initialize('urbansound8k')
    dataset.download()

    # get annotations and audio for a random clip
    example_clip = dataset.choice_clip()
    tags = example_clip.tags
    y, sr = example_clip.audio

``soundata`` loaders contain methods to:

- ``download()``: download (or give instructions to download) a dataset
- ``load_*()``: load a dataset's files (audio, metadata, annotations, etc.) into standard formats, so you don't have to write them yourself
  which are compatible with ``mir_eval`` and ``jams``.
- ``validate()``: validate that a dataset is complete and correct
- ``cite()``: quickly print a dataset's relevant citation
- access ``clip`` and ``clipgroup`` objects for grouping multiple annotations for a particular clip/clipgroup
- and more

See the :ref:`tutorial` for a detailed explanation of how to get started using this library.


soundata design principles
#########################

Ease of use and contribution
----------------------------

We designed ``soundata`` to be easy to use and easy to contribute to. ``soundata`` simplifies the research pipeline considerably,
facilitating research in a wider diversity of tasks and musical datasets. We provide detailed examples on how to interact with 
the library in the :ref:`tutorial`, as well as detail explanation on how to contribute in :ref:`contributing`. Additionally, 
we have a `repository of Jupyter notebooks <https://github.com/soundata/soundata-notebooks>`_ with usage
examples of the different datasets.


Reproducibility
---------------

We aim for ``soundata`` to aid in increasing research reproducibility by providing a common framework for MIR researchers to
compare and validate their data. If mistakes are found in annotations or audio versions change, using ``soundata``, the community
can fix mistakes while still being able to compare methods moving forward.

.. _canonical version:

canonical versions
^^^^^^^^^^^^^^^^^^
The ``dataset loaders`` in ``soundata`` are written for what we call the ``canonical version`` of a dataset. Whenever possible,
this should be the official release of the dataset as published by the dataset creator/s. When this is not possible, (e.g. for 
data that is no longer available), the procedure we follow is to find as many copies of the data as possible from different researchers 
(at least 4), and use the most common one. To make this process transparent, when there are doubts about the data consistency we open an 
`issue <https://github.com/soundata/soundata/issues>`_ and leave it to the community to discuss what to use.


Standardization
---------------

Different datasets have different annotations, metadata, etc. We try to respect the idiosyncracies of each dataset as much as we can. For this
reason, ``clips`` in each ``Dataset`` in ``soundata`` have different attributes, e.g. some may have ``fold`` information and some may not.
However there are some elements that are common in most datasets, and in these cases we standarize them to increase the usability of the library.
Some examples of this are the annotations in ``soundata``, e.g., ``Tags`` and ``Events``.


.. _indexes:

indexes
#######

Indexes in `soundata` are manifests of the files in a dataset and their corresponding md5 checksums.
Specifically, an index is a json file with the mandatory top-level key ``version`` and at least one of the optional
top-level keys ``metadata``, ``clips``, ``clipgroups`` or ``records``. An index might look like:


.. admonition:: Example Index
    :class: dropdown

    .. code-block:: javascript

        {   "version": "1.0.0",
            "metadata": {
                "metadata_file_1": [
                        // the relative path for metadata_file_1
                        "path_to_metadata/metadata_file_1.csv",
                        // metadata_file_1 md5 checksum
                        "bb8b0ca866fc2423edde01325d6e34f7"
                    ],
                "metadata_file_2": [
                        // the relative path for metadata_file_2
                        "path_to_metadata/metadata_file_2.csv",
                        // metadata_file_2 md5 checksum
                        "6cce186ce77a06541cdb9f0a671afb46"
                    ]
                }
            "clips": {
                "clip1": {
                    'audio': ["audio_files/clip1.wav", "6c77777ce77a06541cdb9f0a671afb46"],
                    'tags': ["annotations/clip1.tags.csv", "ab8b0ca866fc2423edde01325d6e34f7"],
                    'events': ["annotations/clip1.events.txt", "05abeca866fc2423edde01325d6e34f7"],
                }
                "clip2": {
                    'audio': ["audio_files/clip2.wav", "6c77777ce77a06542cdb9f0a672afb46"],
                    'tags': ["annotations/clip2.tags.csv", "ab8b0ca866fc2423edde02325d6e34f7"],
                    'events': ["annotations/clip2.events.txt", "05abeca866fc2423edde02325d6e34f7"],
                }
                ...
                }
        }


The optional top-level keys (`clips`, `clipgroups` and `records`) relate to different organizations of sound datasets.
`clips` are used when a dataset is organized as a collection of individual clips, namely mono or multi-channel audio, 
spectrograms only, and their respective annotations. `clipgroups` are used when a dataset comprises of
clipgroups - different groups of clips which are directly related to each other. Finally, `records` are used when a dataset 
consits of groups of tables (e.g. relational databases), as many recommendation datasets do.

See the contributing docs :ref:`create_index` for more information about soundata indexes.

.. annotations:

annotations
###########

soundata provdes ``Annotation`` classes of various kinds which provide a standard interface to different
annotation formats such as tags and sound events.


metadata
########

When available, we provide extensive and easy-to-access ``metadata`` to facilitate clip metadata-specific analysis. 
``metadata`` is available as attroibutes at the ``clip`` level, e.g. ``clip.fold``.
