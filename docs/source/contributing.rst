.. _contributing:

########################
Contributing to Soundata
########################

We encourage contributions to soundata, especially new dataset loaders. To contribute a new loader, follow the
steps indicated below and create a Pull Request (PR) to the github repository. For any doubt or comment about
your contribution, you can always submit an issue or open a discussion in the repository.

- `Issue Tracker <https://github.com/soundata/soundata/issues>`__
- `Source Code <https://github.com/soundata/soundata>`__

Quick link to contributing templates
####################################

If you're familiar with Soundata's API already, you can find the template files for contributing `here <https://github.com/soundata/soundata/tree/master/docs/source/contributing_examples>`__,
and the loader checklist for submitting your PR `here <https://github.com/soundata/soundata/blob/master/.github/PULL_REQUEST_TEMPLATE/new_loader.md>`__.




Installing soundata for development purposes
############################################

To install ``soundata`` for development purposes:

    - First run:

    .. code-block:: console

        git clone https://github.com/soundata/soundata.git

    - Then, after opening source data library you have to install the dependencies for updating the documentation
      and running tests:

    .. code-block:: console

        pip install .
        pip install ."[tests]"
        pip install ."[docs]"


We recommend using `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ or
`pyenv <https://github.com/pyenv/pyenv#installation>`__ to manage your Python versions
and install all ``soundata`` requirements. You will want to install the latest supported Python versions (see README.md).
Once ``conda`` or ``pyenv`` and the Python versions are configured, install ``pytest``. Make sure you've installed all the 
necessary pytest plugins needed (e.g. `pytest-cov`) to automatically test your code successfully.

Before running the tests, make sure to have formatted ``soundata/`` and ``tests/`` with ``black``.

.. code-block:: bash

    black soundata/ tests/


Also, make sure that they pass flake8 and mypy tests specified in lint-python.yml github action workflow.

.. code-block:: bash

    flake8 soundata --count --select=E9,F63,F7,F82 --show-source --statistics
    python -m mypy soundata --ignore-missing-imports --allow-subclassing-any


Finally, run:

.. code-block:: bash

    pytest -vv --cov-report term-missing --cov-report=xml --cov=soundata tests/ --local


All tests should pass!

.. note::
        Soundata assumes that your system has the zip library installed for unzipping files. 


Writing a new dataset loader
############################


The steps to add a new dataset loader to ``soundata`` are:

1. `Create an index <create_index_>`_
2. `Create a module <create_module_>`_
3. `Add tests <add_tests_>`_
4. `Submit your loader <submit_loader_>`_
5. `Upload JSON index to the Soundata index repository in Zenodo <upload_index_>`_

**Before starting**, if your dataset **is not fully downloadable** you should:


1. Contact the soundata team by opening an issue or PR so we can discuss how to proceed with the closed dataset.
2. Show that the version used to create the checksum is the "canonical" one, either by getting the version from the 
   dataset creator, or by verifying equivalence with several other copies of the dataset.

To reduce friction, we will make commits on top of contributors PRs by default unless
the ``please-do-not-edit`` flag is used.

.. _create_index:

1. Create an index
------------------

Soundata's structure relies on ``indexes``. Indexes are dictionaries that contain information about the structure of the
dataset which is necessary for the loading and validating functionalities of Soundata. In particular, indexes contain
information about the files included in the dataset, their location and checksums, see some example indexes below.
To create an index, the necessary steps are:

1. Create a script in ``scripts/``, called ``make_<datasetname>_index.py``, which generates an index file.
2. Then run the script on the canonical version of the dataset and save the index in ``soundata/datasets/indexes/`` as ``<datasetname>_index.json``.
3. When the dataloader is completed and the PR is accepted, upload the index in our `Zenodo community <https://zenodo.org/communities/audio-data-loaders/>`_. See more details `here <upload_index_>`_.

The function ``make_<datasetname>_index.py`` should automate the generation of an index by computing the MD5 checksums for given files in a dataset located at data_path. 
Users can adapt this function to create an index for their dataset by adding their file paths and using the md5 function to generate checksums for their files.

.. _index example:

Here's an example of an index to use as a guide:

.. admonition:: Example Make Index Script
    :class: dropdown

    .. literalinclude:: contributing_examples/make_example_index.py
        :language: python

More examples of scripts used to create dataset indexes can be found in the `scripts <https://github.com/soundata/soundata/tree/master/scripts>`_ folder.

    .. note::
        Users should be able to create the dataset indexes without the need for additional dependencies that are not included in soundata by default. Should you need an additional dependency for a specific reason, please open an issue to discuss with the Soundata maintainers the need for it.

Example index with clips
^^^^^^^^^^^^^^^^^^^^^^^^

Most sound datasets are organized as a collection of clips and annotations. In such case, the index should make use of the ``clips``
top-level key. Under this ``clips`` top-level key, you should store a dictionary where the keys are the unique clip ids of the dataset, and
the values are dictionaries of files associated with a clip id, along with their checksums. These files can be for instance audio files
or annotations related to the clip id. File paths are relative to the top level directory of a dataset.

    .. note::
        If your sound dataset does not fit into a structure around the clip class, please open an issue in the GitHub repository to discuss how to proceed. These are corner cases that we address especially to maintain the consistency of the library.

Currently, Soundata does not include built-in functions to automatically create train, test, and validation splits if these are not originally defined in the dataset. 
Users can do that using  external functions such as ``sklearn.model_selection.train_test_split``.
If a dataset has predefined splits, you can include the split name as an attribute of the ``Clip`` class. You should not create separate indexes for the different splits, or indicate the split in the index.
See an example of how an index should look like:


.. admonition:: Index Examples - Clips
    :class: dropdown

    If the version `1.0` of a given dataset has the structure:

    .. code-block:: javascript

        > Example_Dataset/
            > audio/
                clip1.wav
                clip2.wav
                clip3.wav
            > annotations/
                clip1.csv
                clip2.csv
                clip3.csv
            > metadata/
                metadata_file.csv

    The top level directory is ``Example_Dataset`` and the relative path for ``clip1.wav``
    would be ``audio/clip1.wav``. Any unavailable fields are indicated with `null`. A possible index file for this example would be:

    .. code-block:: javascript


        {
            "version": "1.0",
                "clips":
                    "clip1": {
                        "audio": [
                            "audio/clip1.wav",  // the relative path for clip1's audio file
                            "912ec803b2ce49e4a541068d495ab570"  // clip1.wav's md5 checksum
                        ],
                        "annotation": [
                            "annotations/clip1.csv",  // the relative path for clip1's annotation
                            "2cf33591c3b28b382668952e236cccd5"  // clip1.csv's md5 checksum
                        ]
                    },
                    "clip2": {
                        "audio": [
                            "audio/clip2.wav",
                            "65d671ec9787b32cfb7e33188be32ff7"
                        ],
                        "annotation": [
                            "annotations/Clip2.csv",
                            "e1964798cfe86e914af895f8d0291812"
                        ]
                    },
                    "clip3": {
                        "audio": [
                            "audio/clip3.wav",
                            "60edeb51dc4041c47c031c4bfb456b76"
                        ],
                        "annotation": [
                            "annotations/clip3.csv",
                            "06cb006cc7b61de6be6361ff904654b3"
                        ]
                    },
                }
            "metadata": {
                    "metadata_file": [
                        "metadata/metadata_file.csv",
                        "7a41b280c7b74e2ddac5184708f9525b"
                    ]
            }
        }


    .. note::
        In this example there is a (purposeful) mismatch between the name of the audio file ``clip2.wav`` and its corresponding annotation file, ``Clip2.csv``, compared with the other pairs. This mismatch should be included in the index. This type of slight difference in filenames happens often in publicly available datasets, making pairing audio and annotation files more difficult. We use a fixed, version-controlled index to account for this kind of mismatch, rather than relying on string parsing on load.

..
    Example index with multiclips
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. admonition:: Index Examples - Multiclips
        :class: dropdown, warning

     If the version `1.0` of a given multiclip dataset has the structure:

        .. code-block:: javascript

            > Example_Dataset/
                > audio/
                    multiclip1-voice1.wav
                    multiclip1-voice2.wav
                    multiclip1-accompaniment.wav
                    multiclip1-mix.wav
                    multiclip2-voice1.wav
                    multiclip2-voice2.wav
                    multiclip2-accompaniment.wav
                    multiclip2-mix.wav
                > annotations/
                    multiclip1-voice-f0.csv
                    multiclip2-voice-f0.csv
                    multiclip1-f0.csv
                    multiclip2-f0.csv
                > metadata/
                    metadata_file.csv

        The top level directory is ``Example_Dataset`` and the relative path for ``multiclip1-voice1``
        would be ``audio/multiclip1-voice1.wav``. Any unavailable fields are indicated with `null`. A possible index file for this
    example would be:

    .. code-block:: javascript

    { "version": 1,
      "clips": {
         "multiclip1-voice": {
              "audio_voice1": ('audio/multiclip1-voice1.wav', checksum),
              "audio_voice2": ('audio/multiclip1-voice1.wav', checksum),
              "voice-f0": ('annotations/multiclip1-voice-f0.csv', checksum)
         }
         "multiclip1-accompaniment": {
              "audio_accompaniment": ('audio/multiclip1-accompaniment.wav', checksum)
         }
         "multiclip2-voice" : {...}
         ...
      },
      "multiclips": {
        "multiclip1": {
             "clips": ['multiclip1-voice', 'multiclip1-accompaniment'],
             "audio": ('audio/multiclip1-mix.wav', checksum)
             "f0": ('annotations/multiclip1-f0.csv', checksum)
         }
        "multiclip2": ...
      },
      "metadata": {
        "metadata_file": [
            "metadata/metadata_file.csv",
            "7a41b280c7b74e2ddac5184708f9525b"
            ]
      }
    }

    Note that in this examples we group ``audio_voice1`` and ``audio_voice2`` in a single clip because the annotation ``voice-f0`` annotation corresponds to their mixture. In contrast, the annotation ``voice-f0`` is extracted from the multiclip mix and it is stored in the ``multiclips`` group. The multiclip ``multiclip1`` has an additional clip ``multiclip1-mix.wav`` which may be the master clip, the final mix, the recording of ``multiclip1`` with another microphone.


.. _upload_index:

3. Uploading the index to an online repository
----------------------------------------------

We store all dataset indexes in an online repository on Zenodo.
To use a dataloader, users may retrieve the index running the ``dataset.download()`` function that is also used to download the dataset.
To download only the index, you may run ``.download(["index"])``. The index will be automatically downloaded and stored in the expected folder in Soundata.

From a contributor point of view, you may create the index, store it locally, and develop the dataloader.

    .. note::
        All JSON files in ``soundata/indexes/`` are included in the .gitignore file, therefore there is no need to remove it when pushing, since it will be ignored by git.

**Important!** When creating the PR, please `submit your index to our Zenodo community <https://zenodo.org/communities/audio-data-loaders/>`_:

* First, click on ``New upload``. 
* Add your index in the ``Upload files`` section.
* Let Zenodo create a DOI for your index, so click *No*.
* Resource type is *Other*.
* Title should be *soundata-<dataset-id>_index_<version>*, e.g. soundata-tau2021sse_nigens_index_1.2.0.
* Add yourself as the Creator of this entry.
* The license of the index should be the `same as Soundata <https://github.com/soundata/soundata/blob/main/LICENSE>`_. 

    .. note::
        *<dataset-id>* is the identifier we use to initialize the dataset using ``soundata.initialize()``. It's also the filename of your dataset module.

Visibility should be set as *Public*. There is no need to fill up anything else. 
All the information that users may need is found in the dataloader and the corresponding documentation.


.. _create_module:

2. Create a module
------------------

Once the index is created you can create the loader. For that, we suggest you use the following template and adjust it for your dataset.
To quickstart a new module:

1. Copy the example below and save it to ``soundata/datasets/<your_dataset_name>.py``
2. Find & Replace ``Example`` with the <your_dataset_name>.
3. Remove any lines beginning with `# --` which are there as guidelines. 

.. admonition:: Example Module
    :class: dropdown

    .. literalinclude:: contributing_examples/example.py
        :language: python

You may find these examples useful as references:

* `A simple, fully downloadable dataset <https://github.com/soundata/soundata/blob/master/soundata/datasets/urbansed.py>`_
* `A dataset which uses dataset-level metadata <https://github.com/soundata/soundata/blob/master/soundata/datasets/esc50.py#L217>`_
* `A dataset which does not use dataset-level metadata <https://github.com/soundata/soundata/blob/master/soundata/datasets/urbansed.py#L294>`_

Please, do remember to include the variables ``BIBTEX``, ``REMOTES``, and ``LICENSE_INFO`` at the beginning of your module.
You should follow the provided template as much as possible, and use the recommended functions and classes.
An important example of that is ``download_utils.RemoteFileMetadata``. Please use this class to parse the dataset from an online repository, which takes cares of the download process and the checksum validation, and addresses corner carses. Please do not use specific functions like ``download_zip_file`` or ``download_and_extract`` individually in your loader.

Make sure to include, in the docstring of the dataloader, information about the following list of relevant aspects about the dataset you are integrating:

* The dataset name.
* A general purpose description, the task it is used for.
* Details about the coverage: how many clips, how many hours of audio, how many classes, the annotations available, etc.
* The license of the dataset (even if you have included the ``LICENSE_INFO`` variable already).
* The authors of the dataset, the organization in which it was created, and the year of creation (even if you have included the ``BIBTEX`` variable already).
* Please reference also any relevant link or website that users can check for more information.

.. note::  
    In addition to the module docstring, you should write docstrings for every new class and function you write. See :ref:`the documentation tutorial <documentation_tutorial>` for practical information on best documentation practices.


This docstring is important for users to understand the dataset and its purpose.
Having proper documentation also enhances transparency, and helps users to understand the dataset better.
Please do not include complicated tables, big pieces of text, or unformatted copy-pasted text pieces. 
It is important that the docstring is clean, and the information is very clear to users.
This will also engage users to use the dataloader!

For many more examples, see the `datasets folder <https://github.com/soundata/soundata/tree/master/soundata/datasets>`_.

.. note::  
    If the dataset you are trying to integrate stores every clip in a separated compressed file, it cannot be currently supported by soundata. Feel free to open and issue to discuss a solution (hopefully for the near future!)


.. _add_tests:

3. Add tests
------------

To finish your contribution, please include tests that check the integrity of your loader. For this, follow these steps:

1. Make a toy version of the dataset in the tests folder ``tests/resources/sound_datasets/my_dataset/``,
   so you can test against little data. For example:

    * Include all audio and annotation files for one clip of the dataset.
    * For each audio/annotation file, reduce the audio length to 1-2 seconds and remove all but a few of the annotations.
    * If the dataset has a metadata file, reduce the length to a few lines.

2. Test all of the dataset specific code, e.g. the public attributes of the Clip class, the load functions and any other
   custom functions you wrote. See the `tests folder <https://github.com/soundata/soundata/tree/master/tests>`_ for reference.
   If your loader has a custom download function, add tests similar to
   `this mirdata loader <https://github.com/soundata/soundata/blob/master/tests/test_groove_midi.py#L96>`_.
3. Locally run ``pytest -s tests/test_full_dataset.py --local --dataset my_dataset`` before submitting your loader to make 
   sure everything is working.


.. note::  We have written automated tests for all loader's ``cite``, ``download``, ``validate``, ``load``, ``clip_ids`` functions,
           as well as some basic edge cases of the ``Clip`` class, so you don't need to write tests for these!


.. _test_file:

.. admonition:: Example Test File
    :class: dropdown

    .. literalinclude:: contributing_examples/test_example.py


Running your tests locally
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before creating a PR you should run the tests. But before that, make sure to have formatted ``soundata/`` and ``tests/`` with ``black``.

.. code-block:: bash

    black soundata/ tests/


Also, make sure that they pass flake8 and mypy tests specified in lint-python.yml github action workflow.

.. code-block:: bash

    flake8 soundata --count --select=E9,F63,F7,F82 --show-source --statistics
    python -m mypy soundata --ignore-missing-imports --allow-subclassing-any


Finally, run all the tests locally like this:

.. code-block:: bash

    pytest -vv --cov-report term-missing --cov-report=xml --cov=soundata tests/ --local


The ``--local`` flag skips tests that are built to run only on the remote testing environment.

To run one specific test file:

::

    pytest tests/test_urbansed.py


Finally, there is one local test you should run, which we can't easily run in our testing environment.

::

    pytest -s tests/test_full_dataset.py --local --dataset dataset


Where ``dataset`` is the name of the module of the dataset you added. The ``-s`` tells pytest not to skip print 
statments, which is useful here for seeing the download progress bar when testing the download function.

This tests that your dataset downloads, validates, and loads properly for every clip. This test takes a long time
for some datasets, but it's important to ensure the integrity of the library.

The ``--skip-download`` flag can be added to ``pytest`` command to run the tests skipping the download.
This will skip the downloading step. Note that this is just for convenience during debugging - the tests should eventually all pass without this flag.

.. _working_big_datasets:

Working with big datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

In the development of large datasets, it is advisable to create an index as small as possible to optimize the implementation process
of the dataset loader and pass the tests.


Working with remote indexes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the end-user there is no difference between the remote and local indexes. However, indexes can get large when there are a lot of clips
in the dataset. In these cases, storing and accessing an index remotely can be convenient. Large indexes can be added to REMOTES, 
and will be downloaded with the rest of the dataset. For example:

.. code-block:: python

    "index": download_utils.RemoteFileMetadata(
        filename="remote_index.json.zip",
        url="https://zenodo.org/record/.../remote_index.json.zip?download=1",
        checksum="810f1c003f53cbe58002ba96e6d4d138",
    )


Unlike local indexes, the remote indexes will live in the ``data_home`` directory. When creating the ``Dataset``
object, specify the ``custom_index_path`` to where the index will be downloaded (as a relative path to ``data_home``).


.. _reducing_test_space:

Reducing the testing space usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We are trying to keep the test resources folder size as small as possible, because it can get really heavy as new loaders are added. We
kindly ask the contributors to **reduce the size of the testing data** if possible (e.g. trimming the audio clips, keeping just two rows for
csv files).


.. _submit_loader:

4. Submit your loader
---------------------

Before you submit your loader make sure to:

1. Add your module to ``docs/source/soundata.rst`` following an alphabetical order.
2. Add your module to ``docs/source/table.rst`` following an alphabetical order as follows:

.. code-block:: rst

    * - Dataset
      - Downloadable?
      - Annotations
      - Clips
      - Hours
      - Usecase
      - License

An example of this for the ``UrbanSound8k`` dataset:

.. code-block:: rst

   * - UrbanSound8K
     - - audio: ✅
       - annotations: ✅
     - :ref:`tags`
     - 8732
     - 8.75
     - Urban sound classification
     - .. image:: https://licensebuttons.net/l/by-nc/4.0/80x15.png
          :target: https://creativecommons.org/licenses/by-nc/4.0


You can find license badges images and links `here <https://gist.github.com/lukas-h/2a5d00690736b4c3a7ba>`_.

Pull Request template
^^^^^^^^^^^^^^^^^^^^^

When starting your PR please use the `new_loader.md template <https://github.com/soundata/soundata/blob/master/.github/PULL_REQUEST_TEMPLATE/new_loader.md>`_,
it will simplify the reviewing process and also help you make a complete PR. You can do that by adding
``&template=new_loader.md`` at the end of the url when you are creating the PR :

``...soundata/soundata/compare?expand=1`` will become
``...soundata/soundata/compare?expand=1&template=new_loader.md``.

Troubleshooting
^^^^^^^^^^^^^^^

If github shows a red ``X`` next to your latest commit, it means one of our checks is not passing. This could mean:

1. running ``black`` has failed -- this means that your code is not formatted according to ``black``'s code-style. To fix this, simply run
   the following from inside the top level folder of the repository:

::

    black soundata/ tests/

2. Your code does not pass ``flake8`` test.

::

    flake8 soundata --count --select=E9,F63,F7,F82 --show-source --statistics


3. Your code does not pass ``mypy`` test.

::

    python -m mypy soundata --ignore-missing-imports --allow-subclassing-any

4. the test coverage is too low -- this means that there are too many new lines of code introduced that are not tested.

5. the docs build has failed -- this means that one of the changes you made to the documentation has caused the build to fail. 
   Check the formatting in your changes and make sure they are consistent.

6. the tests have failed -- this means at least one of the tests is failing. Run the tests locally to make sure they are passing. 
   If they are passing locally but failing in the check, open an `issue` and we can help debug.


.. _documentation_tutorial:

Documentation
#############

This documentation is in `rst format <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_.
It is built using `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ and hosted on `readthedocs <https://readthedocs.org/>`_.
The API documentation is built using `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_, which autogenerates
documentation from the code's docstrings. We use the `napoleon <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ plugin
for building docs in Google docstring style. See the next section for docstring conventions.

Docstring conventions
---------------------

soundata uses `Google's Docstring formatting style <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`_.
Here are some common examples.

.. note::
    The small formatting details in these examples are important. Differences in new lines, indentation, and spacing make
    a difference in how the documentation is rendered. For example writing ``Returns:`` will render correctly, but ``Returns``
    or ``Returns :`` will not. 


Functions:

.. code-block:: python

    def add_to_list(list_of_numbers, scalar):
        """Add a scalar to every element of a list.
        You can write a continuation of the function description here on the next line.

        You can optionally write more about the function here. If you want to add an example
        of how this function can be used, you can do it like below.

        Example:
            .. code-block:: python

            foo = add_to_list([1, 2, 3], 2)

        Args:
            list_of_numbers (list): A short description that fits on one line.
            scalar (float):
                Description of the second parameter. If there is a lot to say you can
                overflow to a second line.

        Returns:
            list: Description of the return. The type here is not in parentheses

        """
        return [x + scalar for x in list_of_numbers]


Functions with more than one return value:

.. code-block:: python

    def multiple_returns():
        """This function has no arguments, but more than one return value. Autodoc with napoleon doesn't handle this well,
        and we use this formatting as a workaround.

        Returns:
            * int - the first return value
            * bool - the second return value

        """
        return 42, True


One-line docstrings

.. code-block:: python

    def some_function():
        """
        One line docstrings must be on their own separate line, or autodoc does not build them properly
        """
        ...


Objects

.. code-block:: python

    """Description of the class
    overflowing to a second line if it's long

    Some more details here

    Args:
        foo (str): First argument to the __init__ method
        bar (int): Second argument to the __init__ method

    Attributes:
        foobar (str): First clip attribute
        barfoo (bool): Second clip attribute

    Cached Properties:
        foofoo (list): Cached properties are special soundata attributes
        barbar (None): They are lazy loaded properties.
        barf (bool): Document them with this special header.

    """

Documenting your contribution
-----------------------------

Staged docs for every new PR are built and accessible at ``soundata--<#PR_ID>.org.readthedocs.build/en/<#PR_ID>/`` in which ``<#PR_ID>`` is the pull request ID. 
To quickly troubleshoot any issues, you can build the docs locally by navigating to the ``docs`` folder, and running 
``make clean html`` (note, you must have ``sphinx`` installed). Then open the generated ``soundata/docs/_build/source/index.html`` 
file in your web browser to view.

**Important:** Make sure to check out the ``WARNINGS`` and ``ERROR`` messages that may show up in the terminal when running ``make clean html``. 
These will indicate formatting, listing, and indentation problems that may be present in your docstrings and that need to be fixed for a proper rendering of the documentation.
See the examples aboove and also the docstrings of ``docs/source/contributing_examples/example.py`` to see a list of examples of how to write the docstrings to prevent Sphinx errors and warning messages.




Conventions
###########

Loading from files
------------------

We use the following libraries for loading data from files:

+-------------------------+-------------+
| Format                  | library     |
+=========================+=============+
| audio (wav, mp3, ...)   | librosa     |
+-------------------------+-------------+
| json                    | json        |
+-------------------------+-------------+
| csv                     | csv         |
+-------------------------+-------------+
| jams                    | jams        |
+-------------------------+-------------+

Clip Attributes
----------------
Custom clip attributes should be global, clip-level data.
For some datasets, there is a separate, dataset-level metadata file
with clip-level metadata, e.g. as a csv. When a single file is needed
for more than one clip, we recommend using writing a ``_metadata`` cached property (which
returns a dictionary, either keyed by clip_id or freeform)
in the Dataset class (see the dataset module example code above). When this is specified,
it will populate a clip's hidden ``_clip_metadata`` field, which can be accessed from
the clip class.

For example, if ``_metadata`` returns a dictionary of the form:

.. code-block:: python

    {
        'clip1': {
            'microphone-type': 'Awesome',
            'recording-date': '27.10.2021'
        },
        'clip2': {
            'microphone-type': 'Less_awesome',
            'recording-date': '27.10.2021'
        }
    }

the ``_clip metadata`` for ``clip_id=clip2`` will be:

.. code-block:: python

    {
        'microphone-type': 'Less_awesome',
        'recording-date': '27.10.2021'
    }


Load methods vs Clip properties
--------------------------------
Clip properties and cached properties should be simple, and directly call a ``load_*`` method. Like this example from ``urbansed``:

.. code-block:: python

    @property
    def split(self):
        """The data splits (e.g. train)

        Returns
            * str - split

        """
        return self._clip_metadata.get("split")

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        """The audio events

        Returns
            * annotations.Events - audio event object

        """
        return load_events(self.txt_path)

There should be no additional logic in a clip property/cached property, and instead all logic
should be done in the load method. We separate these because the clip properties are only usable
when data is available locally - when data is remote, the load methods are used instead.

Missing Data
------------
Clip properties that are available for some clips and not for others should be set to ``None`` when whey are not available.
Like this example in the ``tau2019aus`` loader:

.. code-block:: python

    @property
    def tags(self):
        scene_label = self._clip_metadata.get("scene_label")
        if scene_label is None:
            return None
        else:
            return annotations.Tags([scene_label], "open", np.array([1.0]))


The index should only contain key-values for files that exist.

Custom Decorators
#################

cached_property
---------------
This is used primarily for Clip classes.

This decorator causes an Object's function to behave like
an attribute (aka, like the ``@property`` decorator), but caches
the value in memory after it is first accessed. This is used
for data which is relatively large and loaded from files.

docstring_inherit
-----------------
This decorator is used for children of the Dataset class, and
copies the Attributes from the parent class to the docstring of the child.
This gives us clear and complete docs without a lot of copy-paste.

copy_docs
---------
This decorator is used mainly for a dataset's ``load_`` functions, which
are attached to a loader's Dataset class. The attached function is identical,
and this decorator simply copies the docstring from another function.

coerce_to_bytes_io/coerce_to_string_io
--------------------------------------
These are two decorators used to simplify the loading of various `Clip` members
in addition to giving users the ability to use file streams instead of paths in
case the data is in a remote location e.g. GCS. The decorators modify the function
to:

- Return `None` if `None` if passed in.
- Open a file if a string path is passed in either `'w'` mode for `string_io` or `wb` for `bytes_io` and
  pass the file handle to the decorated function.
- Pass the file handle to the decorated function if a file-like object is passed.

This cannot be used if the function to be decorated takes multiple arguments.
`coerce_to_bytes_io` should not be used if trying to load an mp3 with librosa as libsndfile does not support
`mp3` yet and `audioread` expects a path.
