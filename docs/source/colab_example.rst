.. _colab_example:

###################
Colab Usage Example
###################

This page shows how to use ``soundata`` on `Google Colab <https://colab.research.google.com>`_.
Colab provides a browser-based Python environment with free GPU support, which is useful for exploring datasets quickly.

############
Installation
############

First, install ``soundata`` inside your Colab notebook:

.. code-block:: console

    !pip install soundata

Soundata is easily imported into your Python code by:

.. code-block:: python

    import soundata

######################
Initializing a dataset
######################

Print a list of all available dataset loaders by calling:

.. code-block:: python

    import soundata
    print(soundata.list_datasets())

To use a loader, (for example, ``urbansound8k``) you need to initialize it by calling:

.. code-block:: python

    import soundata
    dataset = soundata.initialize('urbansound8k', data_home='/choose/where/data/live')

You can specify the directory where the Soundata data is stored by passing a path to ``data_home``.

Soundata supports working with multiple dataset versions.
To see all available versions of a specific dataset, run ``soundata.list_dataset_versions('urbansound8k')``.
Use ``version`` parameter if you wish to use a version other than the default one.

.. code-block:: python

    import soundata
    dataset = soundata.initialize('urbansound8k', data_home='/choose/where/data/live', version="1.0")


#####################
Downloading a dataset
#####################

All dataset loaders in soundata have a ``download()`` function that allows the user to download:

* The :ref:`canonical <faq>` version of the dataset (when available).
* The dataset index, which indicates the list of clips in the dataset and the paths to audio and annotation files.

The index, which is considered part of the source files of Soundata, is specifically downloaded by running ``download(["index"])``.
Indexes will be directly stored in Soundata's indexes folder (``soundata/datasets/indexes``) whereas users can indicate where the dataset files will be stored via ``data_home``.

Downloading a dataset into the default folder
    In this first example, ``data_home`` is not specified. Thus, UrbanSound8K will be downloaded and retrieved from 
    the default folder, ``sound_datasets``, created in the user's root folder:

    .. code-block:: python

        import soundata
        dataset = soundata.initialize('urbansound8k')
        dataset.download()  # Dataset is downloaded into "sound_datasets" folder inside user's root folder

Downloading a dataset into a specified folder
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

    A partial download example for ``tau2019uas`` dataset could be:

    .. code-block:: python

        dataset = soundata.initialize('tau2019uas')
        dataset.download(partial_download=['development.audio.1', 'development.audio.2'])  # download only two remotes


Downloading a multipart dataset
    In some cases, datasets consist of multiple remote files that have to be extracted together locally to correctly recover the data.
    In those cases, remotes that need to be extracted together should be grouped in a list, so all the necessary files are downloaded at once
    (even in a partial download). An example of this is the `fsd50k` loader:

    .. admonition:: Example multipart REMOTES
        :class: dropdown

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
    Some datasets are private, and therefore it is not possible to directly retrieve them from an online repository.
    In those cases, the download function will only download the index file, and if available, the dataset parts that are not private (for some cases, the annotations are available but not the audio).
    The user will have to gather the private data themselves, store it in the preferred ``data_home`` location, and then initialize the dataset as usual, indicating the data location in the ``data_home`` parameter.


    .. note::
        Private datasets may be available to the public upon request. If you are interested in a dataset that is not openly available, please contact the dataset authors or the dataset maintainers to request access.




#############
Storage Notes
#############

Keep in mind:

- By default, data is downloaded to ``~/.soundata`` — this is reset every time you restart your Colab session.
- To persist data, mount your Google Drive:

.. code-block:: python

    from google.colab import drive
    drive.mount('/content/drive')

    # Optional: change default soundata directory
    import os
    os.environ["SOUNDATA_HOME"] = "/content/drive/MyDrive/soundata_data"



