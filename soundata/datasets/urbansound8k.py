"""UrbanSound8K Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **UrbanSound8K**

    **Created By:**

        | Justin Salamon*^, Christopher Jacoby* and Juan Pablo Bello*
        | * Music and Audio Research Lab (MARL), New York University, USA
        | ^ Center for Urban Science and Progress (CUSP), New York University, USA
        | https://urbansounddataset.weebly.com/
        | https://steinhardt.nyu.edu/marl
        | http://cusp.nyu.edu/

    Version 1.0

    *Description:*
        This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, 
        children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music. The classes are 
        drawn from the urban sound taxonomy described in the following article, which also includes a detailed description of 
        the dataset and how it was compiled:

        .. code-block:: latex

            J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
            22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

        All excerpts are taken from field recordings uploaded to www.freesound.org. The files are pre-sorted into ten folds
        (folders named fold1-fold10) to help in the reproduction of and comparison with the automatic classification results
        reported in the article above.

        In addition to the sound excerpts, a CSV file containing metadata about each excerpt is also provided.


    *Audio Files Included:*
        8732 audio files of urban sounds (see description above) in WAV format. The sampling rate, bit depth, and number of 
        channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file).


    *Meta-data Files Included:*

        UrbanSound8k.csv

        This file contains meta-data information about every audio file in the dataset. This includes:

        * slice_file_name: 

        The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, where:
        [fsID] = the Freesound ID of the recording from which this excerpt (slice) is taken
        [classID] = a numeric identifier of the sound class (see description of classID below for further details)
        [occurrenceID] = a numeric identifier to distinguish different occurrences of the sound within the original recording
        [sliceID] = a numeric identifier to distinguish different slices taken from the same occurrence

        * fsID:

        The Freesound ID of the recording from which this excerpt (slice) is taken

        * start

        The start time of the slice in the original Freesound recording

        * end:

        The end time of slice in the original Freesound recording

        * salience:

        A (subjective) salience rating of the sound. 1 = foreground, 2 = background.

        * fold:

        The fold number (1-10) to which this file has been allocated.

        * classID:

        A numeric identifier of the sound class:
        0 = air_conditioner
        1 = car_horn
        2 = children_playing
        3 = dog_bark
        4 = drilling
        5 = engine_idling
        6 = gun_shot
        7 = jackhammer
        8 = siren
        9 = street_music

        * class:

        The class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, 
        siren, street_music.


    *Please Acknowledge EigenScape in Academic Research:*

        When UrbanSound8K is used for academic research, we would highly appreciate it if scientific publications of works 
        partly based on the UrbanSound8K dataset cite the following publication:

        .. code-block:: latex

            J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
            22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

        The creation of this dataset was supported by a seed grant by NYU's Center for Urban Science and Progress (CUSP).


    *Conditions of Use*

        Dataset compiled by Justin Salamon, Christopher Jacoby and Juan Pablo Bello. All files are excerpts of recordings
        uploaded to www.freesound.org. Please see FREESOUNDCREDITS.txt for an attribution list.
        
        The UrbanSound8K dataset is offered free of charge for non-commercial use only under the terms of the Creative Commons
        Attribution Noncommercial License (by-nc), version 3.0: http://creativecommons.org/licenses/by-nc/3.0/
        
        The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including 
        without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or 
        completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, NYU is not 
        liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of
        the UrbanSound8K dataset or any part of it.

    *Feedback*
        | Please help us improve UrbanSound8K by sending your feedback to: justin.salamon@nyu.edu
        | In case of a problem report please include as many details as possible.

"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
@inproceedings{Salamon:UrbanSound:ACMMM:14,
	Address = {Orlando, FL, USA},
	Author = {Salamon, J. and Jacoby, C. and Bello, J. P.},
	Booktitle = {22nd {ACM} International Conference on Multimedia (ACM-MM'14)},
	Month = {Nov.},
	Pages = {1041--1044},
	Title = {A Dataset and Taxonomy for Urban Sound Research},
	Year = {2014}}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="urbansound8k_index_1.0.json",
        url="https://zenodo.org/records/11176928/files/urbansound8k_index_1.0.json?download=1",
        checksum="1c4940e08c1305c49b592f3d9c103e6f",
    ),
    "sample": core.Index(filename="urbansound8k_index_1.0_sample.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="UrbanSound8K.tar.gz",
        url="https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1",
        checksum="9aa69802bbf37fb986f71ec1483a196e",
        unpack_directories=["UrbanSound8K"],
    ),
}

LICENSE_INFO = "Creative Commons Attribution Non Commercial 4.0 International"


class Clip(core.Clip):
    """urbansound8k Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        class_id (int): integer representation of the class label (0-9). See Dataset Info in the documentation for mapping
        class_label (str): string class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
        clip_id (str): clip id
        fold (int): fold number (1-10) to which this clip is allocated. Use these folds for cross validation
        freesound_end_time (float): end time in seconds of the clip in the original freesound recording
        freesound_id (str): ID of the freesound.org recording from which this clip was taken
        freesound_start_time (float): start time in seconds of the clip in the original freesound recording
        salience (int): annotator estimate of class sailence in the clip: 1 = foreground, 2 = background
        slice_file_name (str): The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav
            Please see the Dataset Info in the soundata documentation for further details
        tags (soundata.annotations.Tags): tag (label) of the clip + confidence. In UrbanSound8K every clip has one tag
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
    def slice_file_name(self):
        """The clip's slice filename.

        Returns:
            * str - The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav

        """
        return self._clip_metadata.get("slice_file_name")

    @property
    def freesound_id(self):
        """The clip's Freesound ID.

        Returns:
            * str - ID of the freesound.org recording from which this clip was taken

        """
        return self._clip_metadata.get("freesound_id")

    @property
    def freesound_start_time(self):
        """The clip's start time in Freesound.

        Returns:
            * float - start time in seconds of the clip in the original freesound recording

        """
        return self._clip_metadata.get("freesound_start_time")

    @property
    def freesound_end_time(self):
        """The clip's end time in Freesound.

        Returns:
            * float - end time in seconds of the clip in the original freesound recording

        """
        return self._clip_metadata.get("freesound_end_time")

    @property
    def salience(self):
        """The clip's salience.

        Returns:
            * int - annotator estimate of class sailence in the clip: 1 = foreground, 2 = background

        """
        return self._clip_metadata.get("salience")

    @property
    def fold(self):
        """The clip's fold.

        Returns:
            * int - fold number (1-10) to which this clip is allocated. Use these folds for cross validation

        """
        return self._clip_metadata.get("fold")

    @property
    def class_id(self):
        """The clip's class id.

        Returns:
            * int - integer representation of the class label (0-9). See Dataset Info in the documentation for mapping

        """
        return self._clip_metadata.get("class_id")

    @property
    def class_label(self):
        """The clip's class label.

        Returns:
            * str - string class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music

        """
        return self._clip_metadata.get("class_label")

    @property
    def tags(self):
        """The clip's tags.

        Returns:
            * annotations.Tags - tag (label) of the clip + confidence. In UrbanSound8K every clip has one tag

        """
        return annotations.Tags(
            [self._clip_metadata.get("class_label")], "open", np.array([1.0])
        )

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, tags=self.tags, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=44100) -> Tuple[np.ndarray, float]:
    """Load a UrbanSound8K audio file.

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
    The urbansound8k dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="urbansound8k",
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
        metadata_path = os.path.join(self.data_home, "metadata", "UrbanSound8K.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        with open(metadata_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            raw_data = []
            for line in reader:
                if line[0] != "slice_file_name":
                    raw_data.append(line)

        metadata_index = {}
        for line in raw_data:
            clip_id = line[0].replace(".wav", "")

            metadata_index[clip_id] = {
                "slice_file_name": line[0],
                "freesound_id": line[1],
                "freesound_start_time": float(line[2]),
                "freesound_end_time": float(line[3]),
                "salience": int(line[4]),
                "fold": int(line[5]),
                "class_id": int(line[6]),
                "class_label": line[7],
            }

        return metadata_index
