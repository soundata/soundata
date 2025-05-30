"""
BirdSongs Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **BirdSongs**

    *Created By*

        | Vinay Shanbag

    Version 1.0

    *Description*
        The BirdSongs dataset consists of 9107 3-second audio files, used from Vinay Shanbag's dataset Bird Songs in Kaggle.

    *Audio Files Included*
        9107 3-second audio recordings in WAV format.

    *Meta-data Files Included*

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
author = {Vinay Shanbag},
year = {2020},
month = {05},
pages = {},
title = {bird song data set},
volume = {10},
journal = {Xeno Canto},
doi = {}
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
    "sample": core.Index(filename="birdsongs_index_1.0_sample.json"),
}

REMOTES = {
    "dataset": download_utils.RemoteFileMetadata(
        filename="archive.zip",
        url="https://www.kaggle.com/api/v1/datasets/download/vinayshanbhag/bird-song-data-set",
        checksum="",
    )
}

LICENSE_INFO = "//creativecommons.org/licenses/by-nc-sa/4.0/"


I    """birdsongs Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        filename (str): Name of the audio file (e.g., '11713-2.wav').
        genus (str): Genus of the bird (e.g., 'Cardinalis').
        species (str): Species name of the bird (e.g., 'cardinalis').
        subspecies (str): Subspecies of the bird, may be empty (e.g., '').
        name (str): Common name of the bird (e.g., 'Northern Cardinal').
        recordist (str): Name of the person who recorded the clip (e.g., 'Chris Parrish').
        country (str): Country where the clip was recorded (e.g., 'United States').
        location (str): Specific location of the clip was recorded e.g., 'Sewanee, Franklin County, Tennessee').
        latitude (str): Latitude of the recording location (e.g., '35.2176').
        longitude (str): Longitude of the recording location (e.g., '-85.922').
        altitude (str): Altitude of the recording location in meters (e.g., '580').
        sound_type (str): Type of sound recorded, may be empty (e.g., 'song' or '').
        source_url (str): URL of the source recording (e.g., '//www.xeno-canto.org/11713').
        license (str): License of the clip, may be empty (e.g., '//creativecommons.org/licenses/by-nc-sa/4.0/' or '').
        time (str): Time of day the clip was recorded, may be empty (e.g., '06:21').
        date (str): Date the clip was recorded, may be empty (e.g., '2007-03-31').
        remarks (str): Additional remarks about the clip, may be empty (e.g., 'Normalized to -3dB' or '').
        id (str): Identifier for the clip, shared across multiple recordings (e.g., '11713')
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
    def filename(self):
        """The clip's item ID (filename).

        Returns:
            * str - ID of the clip (filename)

        """
        return self._clip_metadata.get("filename")

    @property
    def genus(self):
        """The clip's genus.

        Returns:
            * str - Genus of the bird
        """
        return self._clip_metadata.get("genus")
    
    @property
    def species(self):
        """The clip's species.

        Returns:
            * str - Species of the bird
        """
        return self._clip_metadata.get("species")
    
    @property
    def subspecies(self):
        """The clip's subspecies.

        Returns:
            * str - Subspecies of the bird (may be empty)
        """
        return self._clip_metadata.get("subspecies")
    
    @property
    def name(self):
        """The clip's common name.

        Returns:
            * str - Common name of the bird
        """
        return self._clip_metadata.get("name")
    
    @property
    def recordist(self):
        """The clip's recordist.

        Returns:
            * str - Name of the person who recorded the clip
        """
        return self._clip_metadata.get("recordist")

    @property
    def country(self):
        """The clip's country.

        Returns:
            * str - Country where the clip was recorded
        """
        return self._clip_metadata.get("country")

    @property
    def location(self):
        """The clip's location.

        Returns:
            * str - Specific location where the clip was recorded
        """
        return self._clip_metadata.get("location")
    
    @property
    def latitude(self):
        """The clip's latitude.

        Returns:
            * str - Latitude of the recording location
        """
        return self._clip_metadata.get("latitude")
    
    @property
    def longitude(self):
        """The clip's longitude.

        Returns:
            * str - Longitude of the recording location
        """
        return self._clip_metadata.get("longitude")
    
    @property
    def altitude(self):
        """The clip's altitude.

        Returns:
            * str - Altitude of the recording location in meters
        """
        return self._clip_metadata.get("altitude")
    
    @property
    def sound_type(self):
        """The clip's sound type.

        Returns:
            * str - Type of sound (e.g., song, call)
        """
        return self._clip_metadata.get("sound_type")
    
    @property
    def source_url(self):
        """The clip's source URL.

        Returns:
            * str - URL of the source recording
        """
        return self._clip_metadata.get("source_url")

    @property
    def license(self):
        """The clip's license.

        Returns:
            * str - License of the recording
        """
        return self._clip_metadata.get("license")

    @property
    def time(self):
        """The clip's recording time.

        Returns:
            * str - Time of day the clip was recorded
        """
        return self._clip_metadata.get("time")

    @property
    def date(self):
        """The clip's recording date.

        Returns:
            * str - Date the clip was recorded
        """
        return self._clip_metadata.get("date")

    @property
    def remarks(self):
        """The clip's remarks.

        Returns:
            * str - Additional remarks about the clip (may be empty)
        """
        return self._clip_metadata.get("remarks")

    @property
    def id(self):
        """The clip's id (it does not differentiate as there are X different recordings for each id).

        Returns:
            * str - Clip's id
        """
        return self._clip_metadata.get("id")
        
@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=44100) -> Tuple[np.ndarray, float]:
    """Load a tfgbirdsongs audio file.

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
    The BirdSongs dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="birdsongs",
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
            clip_id = line[17].replace(".wav", "")

            metadata_index[clip_id] = {
                "id": line[0],
                "genus": line[1],
                "species": line[2],
                "subspecies": line[3],
                "name": line[4],
                "recordist": line[5],
                "country": line[6],
                "location": line[7],
                "latitude": line[8],
                "longitude": line[9],
                "altitude": line[10],
                "sound_type": line[11],
                "source_url": line[12],
                "license": line[13],
                "time": line[14],
                "date": line[15],
                "remarks": line[16],
                "filename": line[17]
            }

        return metadata_index