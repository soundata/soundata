"""Urbansas Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Created By
    ----------

    Justin Salamon*^, Christopher Jacoby* and Juan Pablo Bello*
    * Music and Audio Research Lab (MARL), New York University, USA
    ^ Center for Urban Science and Progress (CUSP), New York University, USA
    https://urbansounddataset.weebly.com/
    https://steinhardt.nyu.edu/marl
    http://cusp.nyu.edu/

    Version 1.0


    Description
    -----------

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


    Audio Files Included
    --------------------

    8732 audio files of urban sounds (see description above) in WAV format. The sampling rate, bit depth, and number of 
    channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file).


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
from moviepy.editor import VideoFileClip

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
@inproceedings{Fuentes:Urbansas:ICASSP2022,
	Address = {Singapore},
	Author = {Fuentes, M.; Steers, B.; Zinemanas, P.; Rocamora, M.; Bondi, L.; Wilkins, J.; Shi, Q.; Hou Y.; Das S.;, Serra, X. and Bello, J.P.},
	Booktitle = {2022 IEEE International Conference on Acoustics, Speech and Signal Processing},
	Month = {May.},
	Pages = {},
	Title = {Urban Sound & Sight: Dataset and Benchmark for Audio-visual Urban Scene Understanding},
	Year = {2022}}
"""
REMOTES = {
    # "all": download_utils.RemoteFileMetadata(
    #    filename="UrbanSound8K.tar.gz",
    #    url="https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1",
    #    checksum="9aa69802bbf37fb986f71ec1483a196e",
    #    unpack_directories=["UrbanSound8K"],
    # )
}

LICENSE_INFO = ""  # TODO: define this


class Clip(core.Clip):
    """urbansas Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        sound_events (soundata.annotation.Events): sound events with start time, end time, label and confidence.
        video_annotations (soundata.annotation.VideoAnnotations): bounding boxes with x, y, dx, dy, label, vehicle_id and confidence.
        audio_path (str): path to the audio file
        video_path (str): path to the audio file
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
        self.video_path = self.get_path("video")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def video(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's video

        Returns:
            * np.ndarray - video frames
            * float - fps

        """
        return load_video(self.video_path)

    @property
    def city(self):
        """The city where was recorded

        Returns
            * str - city

        """
        return self._clip_metadata.get("city")

    @property
    def location_id(self):
        """The location id

        Returns
            * str - location_id

        """
        return self._clip_metadata.get("location_id")

    @property
    def night(self):
        """Night flag

        Returns
            * str - night

        """
        return self._clip_metadata.get("night")

    @property
    def non_identifiable_vehicle_sound(self):
        """non_identifiable_vehicle_sound flag

        Returns
            * str - non_identifiable_vehicle_sound

        """
        return self._clip_metadata.get("non_identifiable_vehicle_sound")

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        """The audio events

        Returns
            * annotations.Events - audio event object

        """
        times = []
        labels = []
        confidence = []
        events_list = self._clip_metadata.get("sound_events")
        for event in events_list:
            if event["start"] > -1:
                times.append([event["start"], event["end"]])
                labels.append(event["label"])
                confidence.append(1.0)

        if len(times) > 0:
            return annotations.Events(
                np.array(times), "seconds", labels, "open", np.array(confidence)
            )
        else:
            return None

    @core.cached_property
    def video_annotations(self) -> Optional[annotations.VideoAnnotations]:
        """The video annotations

        Returns
            * annotations.VideoAnnotations- video annotations object

        """
        positions = []
        labels = []
        frames_id = []
        tracks_id = []
        visibility = []
        times = []

        ann_list = self._clip_metadata.get("video_annotations")
        for ann in ann_list:
            positions.append([ann["x"], ann["y"], ann["w"], ann["h"]])
            labels.append(ann["label"])
            frames_id.append(ann["frame_id"])
            tracks_id.append(ann["track_id"])
            visibility.append(ann["visibility"])
            times.append(ann["time"])

        if len(ann_list) > 0:
            return annotations.VideoAnnotations(
                np.array(positions),
                labels,
                np.array(frames_id),
                np.array(tracks_id),
                np.array(visibility),
                np.array(times),
            )
        else:
            return None

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, events=self.events, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=44100) -> Tuple[np.ndarray, float]:
    """Load a Urbansas audio file.

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
    audio, sr = librosa.load(fhandle, sr=sr, mono=False)
    return audio, sr


def load_video(fhandle: str) -> Tuple[np.ndarray, float]:
    """Load a Urbansas video file.

    Args:
        fhandle (str): Path to video file

    Returns:
        * np.ndarray - the mono audio signal
        * float - The fps of the video file

    """
    frames = []
    video_clip = VideoFileClip(fhandle)
    for frame in video_clip.iter_frames():
        frames.append(frame)
    return frames, video_clip.fps


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The urbansas dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="urbansas",
            clip_class=Clip,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_video)
    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    @core.cached_property
    def _metadata(self):

        audio_annotations = os.path.join(self.data_home, "audio_annotations.csv")
        video_annotations = os.path.join(self.data_home, "video_annotations.csv")

        if not os.path.exists(audio_annotations):
            raise FileNotFoundError(
                "Audio annotations file not found. Did you run .download()?"
            )

        if not os.path.exists(video_annotations):
            raise FileNotFoundError(
                "Video annotations file not found. Did you run .download()?"
            )

        metadata_index = {}

        with open(audio_annotations, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            for line in reader:
                if line[0] == "":
                    continue

                clip_id = line[1]
                if clip_id not in metadata_index:
                    metadata_index[clip_id] = {
                        "sound_events": [],
                        "video_annotations": [],
                    }

                class_id = int(line[2])
                label = line[3]
                non_ident_vehicle_sound = int(line[4]) == 1
                start = float(line[5])
                end = float(line[6])

                if "non_identifiable_vehicle_sound" not in metadata_index[clip_id]:
                    metadata_index[clip_id][
                        "non_identifiable_vehicle_sound"
                    ] = non_ident_vehicle_sound

                metadata_index[clip_id]["sound_events"].append(
                    {"class_id": class_id, "label": label, "start": start, "end": end}
                )

        with open(video_annotations, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            for line in reader:
                if line[0] == "":
                    continue

                frame_id = int(line[1])
                track_id = int(line[2])
                x = float(line[3])
                y = float(line[4])
                w = float(line[5])
                h = float(line[6])
                class_id = int(line[7])
                visibility = float(line[8])
                label = line[9]
                clip_id = line[10]
                city = line[11]
                location_id = line[12]
                time = float(line[13])
                night = int(line[14]) == 1

                if clip_id not in metadata_index:
                    metadata_index[clip_id] = {
                        "sound_events": [],
                        "video_annotations": [],
                    }
                if "city" not in metadata_index[clip_id]:
                    metadata_index[clip_id]["city"] = city
                    metadata_index[clip_id]["night"] = night
                    metadata_index[clip_id]["location_id"] = location_id

                metadata_index[clip_id]["video_annotations"].append(
                    {
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "visibility": visibility,
                        "label": label,
                        "time": time,
                    }
                )

        return metadata_index
