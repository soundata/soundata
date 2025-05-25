"""
TFGBirdSongs Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **TFGBirdSongs**

    *Created By*

        | Sergi García Fornés

    Version 1.0

    *Description*
        The TFGBirdSongs dataset consists of 10,000 ten-second audio files, collected via the Warblr app from users across the UK in 2015-2016.
        Using a classification method by Stowell and Plumbley (2014a), this app aims to identify bird species from user-submitted recordings.
        The dataset, inclusive of various human and environmental noises, is broadly distributed over different times and seasons but has biases towards mornings, weekends, and populated areas. Despite having initial automated bird species estimates, the recordings underwent manual annotation due to precision inadequacies for establishing ground-truth data.
        The dataset proves instrumental for research and development in bird species detection amidst variable noise conditions.

    *Audio Files Included*
        10,000 ten-second audio recordings in WAV format, amassed through the Warblr app during 2015-2016 from users throughout the UK.

    *Meta-data Files Included*
        A table containing a binary label "hasbird" associated to every recording in Warblr is available on the website of the DCASE "Bird Audio Detection" challenge: http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/

    *Please Acknowledge Warblr in Academic Research*
        When the Warblr dataset is employed for academic research, we sincerely request that scientific publications of works partially based on this dataset cite the following publication:

        .. code-block:: latex

            Stowell, Dan and Wood, Michael and Pamuła, Hanna and Stylianou, Yannis and Glotin, Hervé. "Automatic acoustic detection of birds through deep learning: The first Bird Audio Detection challenge", Methods in Ecology and Evolution, 2018.

        The creation and curating of this dataset were possible through the participation and contributions of the general public using the Warblr app, enabling a comprehensive collection of bird sound recordings from various regions within the UK during 2015-2016.

    *Conditions of Use*
        Dataset created by [Creators/Researchers involved].

        The Warblr dataset is offered free of charge under the terms of the Creative Commons Attribution 4.0 International (CC BY 4.0) license:
        https://creativecommons.org/licenses/by/4.0/

        The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, [Affiliated Institution/Organization] is not liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of the Warblr dataset or any part of it.

"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv

from soundata import download_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
@article{article,
author = {Sergi García Fornés},
year = {2025},
month = {05},
pages = {},
title = {Bird Sound Detection},
volume = {10},
journal = {Methods in Ecology and Evolution},
doi = {10.1111/2041-210X.13103}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="tfgbirdsongs_index.json",
        url="https://drive.google.com/file/d/1_z7AbxBMqMPKah0WqA-sYXnTcrtS-cgz/view?usp=drive_link",
        checksum="1b16e1e45ba0c6db6506241edc7b611c",
    ),
    "sample": core.Index(filename="tfgbirdsongs_index.json"),
}

REMOTES = {
    "train": download_utils.RemoteFileMetadata(
        filename="train_set_audio.zip",
        url="https://drive.google.com/file/d/1Xe3uzHcqF4dsShWGaTgp4wylJzpVQ8fZ/view?usp=drive_link",
        checksum="9710b1b3161320a7092372c9db3eb0cc",
        unpack_directories=["wav"],
    ),
    "test": download_utils.RemoteFileMetadata(
        filename="test_set_audio.zip",
        url="https://drive.google.com/file/d/1Av1XVCgHamuDr3-U3V_t8EayEA4un8ff/view?usp=drive_link",
        checksum="8f3496dac443305854b0b47f092d6113",
        unpack_directories=["wav"],
    ),
    "train_metadata": download_utils.RemoteFileMetadata(
        filename="bird_songs_metadata.csv",
        url="https://drive.google.com/file/d/1Jv9xI8NKnBpg1lwxOxP4J4I782jjf8wh/view?usp=drive_link",
        checksum="47101c84dd30aeb0416768261788d107",
    ),
}

LICENSE_INFO = "Sergi García Fornés TFG 1.0"


class Clip(core.Clip):
    """warblrb10k Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        item_id (str): clip id
        has_bird (str): indication of whether the clips contains bird sounds (0/1)
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def item_id(self):
        """The clip's item ID.

        Returns:
            * str - ID of the clip

        """
        return self._clip_metadata.get("itemid")

    @property
    def has_bird(self):
        """The flag to tell whether the clip has bird sound or not.

        Returns:
            * str - 1/0 depending on whether the clip contains bird sound

        """
        return self._clip_metadata.get("hasbird")


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=44100) -> Tuple[np.ndarray, float]:
    """Load a Warblrb10k audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, 44100 Hz by default.
            If different from file's sample rate it will be resampled on load.
            Use None to load the file using its original sample rate (sample rate
            varies from file to file).

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The TFGBirdSongs dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="tfgbirdsongs",
            clip_class=Clip,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "bird_songs_metadata.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. Did you run .download()?"
            )

        with open(metadata_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            raw_data = [line for line in reader if line[0] != "slice_file_name"]

        metadata_index = {}
        for line in raw_data:
            clip_id = line[0].replace(".wav", "")

            metadata_index[clip_id] = {
                "itemid": line[0],
                "hasbird": line[1],
            }

        return metadata_index
