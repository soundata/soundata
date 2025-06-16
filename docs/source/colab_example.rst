.. _colab_example:


Using Soundata in Colab
=======================

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


For more information about partial download, multipart download, non-available datasets to openly download, please refer to `Downloading a dataset <https://soundata.readthedocs.io/en/stable/source/tutorial.html#downloading-a-dataset>`_.

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