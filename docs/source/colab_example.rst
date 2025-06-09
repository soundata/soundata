.. _colab_example:


Colab Usage Example
===================

This page shows how to use ``soundata`` on `Google Colab <https://colab.research.google.com>`_.
Colab provides a browser-based Python environment with free GPU support, which is useful for exploring datasets quickly.
You will have two options that you can use the dataset from ``soundata`` in Colab.

.. _Dataset_download_from_colab:

Dataset Download from Colab
===========================

Installation
------------

First, install ``soundata`` inside your Colab notebook:

.. code-block:: python

    !pip install soundata

Initializing a dataset
----------------------

Soundata is easily imported into your Python code by:

.. code-block:: python

    import soundata


Print a list of all available dataset loaders by calling:

.. code-block:: python

    print(soundata.list_datasets())

To use a loader, (i.e., ``urbansound8k``) you need to initialize it by calling:

.. code-block:: python
    
    import soundata
    dataset = soundata.initialize('urbansound8k')


Soundata supports working with multiple dataset versions.
To see all available versions of a specific dataset, run ``soundata.list_dataset_versions('urbansound8k')``.
Use ``version`` parameter if you wish to use a version other than the default one.

.. code-block:: python

    import soundata
    dataset = soundata.initialize('urbansound8k', version="1.0")

Download the Dataset
--------------------

All dataset loaders in soundata have a ``download()`` function that allows the user to download:

Downloading a dataset

.. code-block:: python

        import soundata
        dataset = soundata.initialize('urbansound8k')
        dataset.download() 


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


.. _Dataset_Storage:

Dataset storage
---------------

**By default**, data is downloaded to:

.. code-block::

    /root/sound_datasets/<Dataset_Name>

.. note::
    This directory is temporary and will be reset every time you restart your Colab session.

To keep the dataset without downloading everytime you start the session, you should:

1. Copy it to google drive:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from google.colab import drive
    drive.mount('/content/drive')

    !cp -r /root/soundata /content/drive/MyDrive/


2. Or set a custom download path when loading the dataset:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from google.colab import drive
    drive.mount('/content/drive')
    
    import soundata
    dataset = soundata.initialize('urbansound8k', data_home='/content/drive/MyDrive/<Folder_Name>')
    dataset.download()

.. _Dataset_download_out_of_colab:

Dataset Download out of Colab
=============================

First, make sure that the dataset is located in **``/content/drive/My Drive/<Dataset_Name>/``** from your Google Drive.

Next, import google drive by:

.. code-block:: python 

    from google.drive import drive
    drive.mount('/content/drive')

To use the dataset in ``soundata``, set the dataset path and initialize by:

.. code-block:: python


    import soundata

    data_path = '/content/drive/My Drive/urbansound8k' # Example: urbansound8k
    dataset = soundata.initialize('urbansound8k', data_home=data_path)
    

Lastly, validate the dataset by :

.. code-block:: python

    dataset.validate()

    # Optional: See what files are loaded
    print(dataset.clip_ids[:5])
    clip = dataset.clip(dataset.clip_ids[0])
    print(clip.audio_path)


``dataset.validate()`` will check if the dataset files are present and follow the expected format. 