"""FSD50K Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Created By
    ----------

    TODO

    Version 1.0


    Description
    -----------

    TODO

    Audio Files Included
    --------------------

    TODO

    Meta-data Files Included
    ------------------------

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


    Please Acknowledge UrbanSound8K in Academic Research
    ----------------------------------------------------

    When UrbanSound8K is used for academic research, we would highly appreciate it if scientific publications of works 
    partly based on the UrbanSound8K dataset cite the following publication:

    .. code-block:: latex
        J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
        22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

    The creation of this dataset was supported by a seed grant by NYU's Center for Urban Science and Progress (CUSP).


    Conditions of Use
    -----------------

    Dataset compiled by Justin Salamon, Christopher Jacoby and Juan Pablo Bello. All files are excerpts of recordings
    uploaded to www.freesound.org. Please see FREESOUNDCREDITS.txt for an attribution list.
    
    The UrbanSound8K dataset is offered free of charge for non-commercial use only under the terms of the Creative Commons
    Attribution Noncommercial License (by-nc), version 3.0: http://creativecommons.org/licenses/by-nc/3.0/
    
    The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including 
    without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or 
    completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, NYU is not 
    liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of
    the UrbanSound8K dataset or any part of it.


    Feedback
    --------

    Please help us improve UrbanSound8K by sending your feedback to: justin.salamon@nyu.edu
    In case of a problem report please include as many details as possible.

"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import json

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
TODO
"""
REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="UrbanSound8K.tar.gz",
        url="https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz",
        checksum="9aa69802bbf37fb986f71ec1483a196e",
        unpack_directories=["UrbanSound8K"],
    )
}

LICENSE_INFO = "Creative Commons Attribution Non Commercial 4.0 International"


class Clip(core.Clip):
    """FSD50K Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        labels (soundata.annotation.Tags): tag (label) of the clip + confidence.
        audio_path (str): path to the audio file
        clip_id (str): clip id

    """

    def __init__(
        self,
        clip_id,
        data_home,
        dataset_name,
        index,
        metadata,
    ):
        super().__init__(
            clip_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        self.audio_path = self.get_path("audio")
        self.sub_set = self.audio_path.split("/")[-2].replace("FSD50K.", "").replace("_audio", "")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def labels(self):
        return annotations.Tags(
            self._metadata()['ground_truth_dev'][self.clip_id]['tags'],
            np.array([1.0] * len(self._metadata()['ground_truth_dev'][self.clip_id]['tags']))
        )

    @property
    def split(self):
        return self._metadata()['ground_truth_dev'][self.clip_id]['split']

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            tags=self.labels,
            metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=44100) -> Tuple[np.ndarray, float]:
    """Load a FSD50K audio file.

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


def load_ground_truth(data_path):
    """
    TODO
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError("Ground truth file not found. Did you run .download()?")

    ground_truth_dict = {}
    with open(data_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        next(reader)
        for line in reader:
            if len(line) == 3:
                ground_truth_dict[line[0]] = {
                    'tags': list(line[1].split(',')) if ',' in line[1] else line[1],
                    'mids': list(line[2].split(',')) if ',' in line[2] else line[2],
                }
            if len(line) == 4:
                if ',' in line[2]:
                    ground_truth_dict[line[0]] = {
                        'tags': list(line[1].split(',')) if ',' in line[1] else line[1],
                        'mids': list(line[2].split(',')) if ',' in line[2] else line[2],
                        'split': line[3],
                    }

    return ground_truth_dict


def load_fsd50k_vocabulary(data_path):
    """
    TODO
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError("FSD50K vocabulary file not found. Did you run .download()?")

    fsd50k_to_audioset = {}
    audioset_to_fsd50k = {}
    with open(data_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            fsd50k_to_audioset[line[1]] = line[2]
            audioset_to_fsd50k[line[2]] = line[1]

    return fsd50k_to_audioset, audioset_to_fsd50k


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The FSD50K dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="fsd50k",
            clip_class=Clip,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_ground_truth)
    def load_ground_truth(self, *args, **kwargs):
        return load_ground_truth(*args, **kwargs)

    @core.copy_docs(load_fsd50k_vocabulary)
    def load_fsd50k_vocabulary(self, *args, **kwargs):
        return load_fsd50k_vocabulary(*args, **kwargs)

    @property
    def fsd50k_to_audioset(self):
        data_path = os.path.join(self.data_home, "FSD50K.ground_truth", "vocabulary.csv")
        return load_fsd50k_vocabulary(data_path)[0]

    @property
    def audioset_to_fsd50k(self):
        data_path = os.path.join(self.data_home, "FSD50K.ground_truth", "vocabulary.csv")
        return load_fsd50k_vocabulary(data_path)[1]

    @property
    def class_info(self):
        data_path = os.path.join(self.data_home, "FSD50K.metadata", "class_info_FSD50K.json")
        return json.load(open(data_path, 'r')) if os.path.exists(data_path) else None

    @core.cached_property
    def _metadata(self):
        # Ground_truth path
        ground_truth_dev_path = os.path.join(self.data_home, "FSD50K.ground_truth", "dev.csv")
        ground_truth_eval_path = os.path.join(self.data_home, "FSD50K.ground_truth", "eval.csv")

        # Load clip metadata path
        clips_info_dev_path = os.path.join(self.data_home, "FSD50K.metadata", "dev_clips_info_FSD50K.json")
        clips_info_eval_path = os.path.join(self.data_home, "FSD50K.metadata", "eval_clips_info_FSD50K.json")

        metadata_index = {
            'ground_truth_dev': load_ground_truth(ground_truth_dev_path),
            'ground_truth_eval': load_ground_truth(ground_truth_eval_path),
            'clips_info_dev': json.load(open(clips_info_dev_path, "r")) if os.path.exists(clips_info_dev_path) else None,
            'clips_info_eval': json.load(open(clips_info_eval_path, "r")) if os.path.exists(clips_info_eval_path) else None,
        }

        return metadata_index








