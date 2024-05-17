About Soundata
==============

Soundata is a Python library for loading and working with audio datasets in a standardized way,
removing the need for writing custom loaders in every project, and improving reproducibility by providing
tools validate data against a canonical version. It speeds up research pipelines by allowing you to quickly
``download`` a dataset, ``load`` it into memory in a ``standardized`` and ``reproducible`` way, ``validate``
that the dataset is complete and correct, and more.

Soundata is based and inspired on `mirdata <https://mirdata.readthedocs.io/en/latest/index.html>`_, and was created
following these desig principles:

- **Easy to use:** Soundata is designed to be easy to use and to simplify the research pipeline considerably. Check out the examples in the :ref:`tutorial` page.
- **Easy to contribute to:** we welcome and encourage contributions, especially new datasets. You can contribute following the instructions in our :ref:`contributing` page.
- **Increase reproducibility:** by providing a common framework for researchers to compare and validate their data, when mistakes are found in annotations or audio versions change, using Soundata the audio community can fix mistakes while still being able to compare methods moving forward.
- **Standardize usage of sound datasets:** we standardized common attributes of sound datasets such as ``audio`` or ``tags`` to simplify audio research pipelines, while preserving each dataset's idiosyncracies: if a dataset has 'non-standard' attributes, we include them as well.


------------


Installation
""""""""""""

To install Soundata simply do:

    .. code-block:: console

        pip install soundata


We recommend to do this inside a conda or virtual environment for reproducibility. To install optional dependencies for plots functionality, please follow :ref:`tutorial`.

------------

Example Usage
"""""""""""""

.. code-block:: Python

    import soundata

    # learn wich datasets are available in soundata
    print(soundata.list_datasets())

    # choose a dataset and download it
    dataset = soundata.initialize('urbansound8k', data_home='choose_where_data_live')
    dataset.download()

    # get annotations and audio for a random clip
    example_clip = dataset.choice_clip()
    tags = example_clip.tags
    y, sr = example_clip.audio


------------

Contributing
""""""""""""

We welcome and encourage contributions to soundata, especially new dataset loaders. To contribute a new loader,
please follow the steps indicated in the :ref:`contributing` page and submit a Pull Request (PR) to the github
repository. For any doubt or comment about your contribution, you can always submit an issue or open a discussion
in the repository.

- `Issue Tracker <https://github.com/soundata/soundata/issues>`_
- `Source Code <https://github.com/soundata/soundata>`_

------------


Citing soundata
"""""""""""""""

`TBA`


.. toctree::
   :hidden:
   :maxdepth: 0

   self
   source/tutorial
   source/contributing
   source/quick_reference



.. toctree::
   :hidden:
   :caption: API documentation
   :maxdepth: 0

   source/soundata


.. toctree::
   :hidden:
   :caption: Reference
   :maxdepth: 0

   source/changelog
   source/faq


