"""TAU SSE 2019 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **TAU SSE 2019**

    *Created By:*

        | Sharath Adavanne; Archontis Politis; Tuomas Virtanen
        | Audio Research Group, Tampere University. 
        
    Version 2

    *Description:*
        Recordings with stationary point sources (events) from multiple sound classes.
        Up to two temporally overlaping sound events. 
        Recordings of identical scenes are available in both 1st-order ambisonics and corresponding four-channel tetrahedral microphone format.
        Recordings can happen in one of five different rooms.
        The sound classes are the 11 different ones from the `DCASE 2016 challenge task 2 <http://dcase.community/challenge2016/task-sound-event-detection-in-synthetic-audio>`_. Each class has 20 different examples.

    *Audio Files Included:*
        * 500 one-minute-long recordings (400 development and 100 evaluation; 48kHz sampling rate and 16-bit precision).

    *Annotations Included:*
        * sound event category with:
            * start time
            * end time 
            * elevation 
            * azimuth
            * distance
        * Moreover, the clip id indicates:
            * data split number (4 in development and 1 in evaluation)
            * room number (IR: impulse response)
            * whether there are temporally-overlapping events

    *Please Acknowledge TAU SSE 2019 in Academic Research:*
    If you use this dataset please cite its original publication: 

    .. code-block:: latex

        Sharath Adavanne, Archontis Politis, and Tuomas Virtanen. A multi-room reverberant dataset for sound event localization and uetection. In Submitted to Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019). 2019. URL: https://arxiv.org/abs/1905.08546.

    *License:*
        * Copyright (c) 2019 Tampere University and its licensors All rights reserved. Permission is hereby granted, without written agreement and without license or royalty fees, to use and copy the TAU Spatial Sound Events 2019 - Ambisonic and Microphone Array described in this document and composed of audio and metadata. This grant is only for experimental and non-commercial purposes, provided that the copyright notice in its entirety appear in all copies of this Work, and the original source of this Work, (Audio Research Group at Tampere University), is acknowledged in any publication that reports research using this Work.
        * Any commercial use of the Work or any part thereof is strictly prohibited. Commercial use include, but is not limited to:
            * selling or reproducing the Work
            * selling or distributing the results or content achieved by use of the Work
            * providing services by using the Work.
        * IN NO EVENT SHALL TAMPERE UNIVERSITY OR ITS LICENSORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS WORK AND ITS DOCUMENTATION, EVEN IF TAMPERE UNIVERSITY OR ITS LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        * TAMPERE UNIVERSITY AND ALL ITS LICENSORS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE WORK PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE TAMPERE UNIVERSITY HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import jams
import glob
import json

from soundata import download_utils, jams_utils, core, annotations, io

ELEVATIONS_UNITS = {"degrees": "degrees"}
AZIMUTHS_UNITS = {"degrees": "degrees"}
DISTANCES_UNITS = {"meters": "meters"}

BIBTEX = """
@inproceedings{adavanne2019multi,
  title={A Multi-room Reverberant Dataset for Sound Event Localization and Detection},
  author={Adavanne, Sharath and Politis, Archontis and Virtanen, Tuomas},
  booktitle={Workshop on Detection and Classification of Acoustic Scenes and Events},
  pages={10--14},
  year={2019}
}
"""

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "2.0": core.Index(
        filename="tau2019sse_index_2.0.json",
        url="https://zenodo.org/records/11176857/files/tau2019sse_index_2.0.json?download=1",
        checksum="6fdfe1ec087ceeaef421b264dd390e24",
    ),
    "sample": core.Index(filename="tau2019sse_index_2.0_sample.json"),
}

REMOTES = {
    "foa_dev": [
        download_utils.RemoteFileMetadata(
            filename="foa_dev.z01",
            url="https://zenodo.org/record/2599196/files/foa_dev.z01?download=1",
            checksum="bd5b18a47a3ed96e80069baa6b221a5a",
        ),
        download_utils.RemoteFileMetadata(
            filename="foa_dev.z02",
            url="https://zenodo.org/record/2599196/files/foa_dev.z02?download=1",
            checksum="5194ebf43ae095190ed78691ec9889b1",
        ),
        download_utils.RemoteFileMetadata(
            filename="foa_dev.zip",
            url="https://zenodo.org/record/2599196/files/foa_dev.zip?download=1",
            checksum="2154ad0d9e1e45bfc933b39591b49206",
        ),
    ],
    "mic_dev": [
        download_utils.RemoteFileMetadata(
            filename="mic_dev.z01",
            url="https://zenodo.org/record/2599196/files/mic_dev.z01?download=1",
            checksum="3234cf0bfa7b71465ae1d67c833f7c12",
        ),
        download_utils.RemoteFileMetadata(
            filename="mic_dev.zip",
            url="https://zenodo.org/record/2599196/files/mic_dev.zip?download=1",
            checksum="6426da74fecb351dd5add56716499e40",
        ),
    ],
    "metadata_dev": download_utils.RemoteFileMetadata(
        filename="metadata_dev.zip",
        url="https://zenodo.org/record/2599196/files/metadata_dev.zip?download=1",
        checksum="c2e5c8b0ab430dfd76c497325171245d",
    ),
    "foa_eval": download_utils.RemoteFileMetadata(
        filename="foa_eval.zip",
        url="https://zenodo.org/record/3377088/files/foa_eval.zip?download=1",
        checksum="4a8ca8bfb69d7c154a56a672e3b635d5",
    ),
    "mic_eval": download_utils.RemoteFileMetadata(
        filename="mic_eval.zip",
        url="https://zenodo.org/record/3377088/files/mic_eval.zip?download=1",
        checksum="0ec2f743a61213480dae7d0b2f2e6c9d",
    ),
    "metadata_eval": download_utils.RemoteFileMetadata(
        filename="metadata_eval.zip",
        url="https://zenodo.org/record/3377088/files/metadata_eval.zip?download=1",
        checksum="a0ec7640284ade0744dfe299f7ba107b",
    ),
}

LICENSE_INFO = "Copyright (c) 2019 Tampere University and its licensors All rights reserved. Permission is hereby granted, without written agreement and without license or royalty fees, to use and copy the TAU Spatial Sound Events 2019 - Ambisonic and Microphone Array described in this document and composed of audio and metadata. This grant is only for experimental and non-commercial purposes, provided that the copyright notice in its entirety appear in all copies of this Work, and the original source of this Work, (Audio Research Group at Tampere University), is acknowledged in any publication that reports research using this Work. Any commercial use of the Work or any part thereof is strictly prohibited. Commercial use include, but is not limited to: selling or reproducing the Work, selling or distributing the results or content achieved by use of the Work providing services by using the Work. IN NO EVENT SHALL TAMPERE UNIVERSITY OR ITS LICENSORS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS WORK AND ITS DOCUMENTATION, EVEN IF TAMPERE UNIVERSITY OR ITS LICENSORS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. TAMPERE UNIVERSITY AND ALL ITS LICENSORS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE WORK PROVIDED HEREUNDER IS ON AN AS IS BASIS, AND THE TAMPERE UNIVERSITY HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS."


class TAU2019_SpatialEvents(annotations.SpatialEvents):
    """TAU SSE 2019 Spatial Events

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals (as floats) in seconds in the form [start_time, end_time]
                                with positive time stamps and end_time >= start_time.
        elevations (np.ndarray): (n,) array of elevations
        azimuths (np.ndarray): (n,) array of azimuths
        distances (np.ndarray): (n,) array of distances
        labels (list): list of event labels (as strings)
        confidence (np.ndarray or None): array of confidence values, float in [0, 1]
        labels_unit (str): labels unit, one of LABELS_UNITS
        intervals_unit (str): intervals unit, one of TIME_UNITS
    """

    def __init__(
        self,
        intervals,
        intervals_unit,
        elevations,
        elevations_unit,
        azimuths,
        azimuths_unit,
        distances,
        distances_unit,
        labels,
        labels_unit,
        confidence=None,
    ):
        super().__init__(
            None,
            intervals_unit,
            None,
            elevations_unit,
            None,
            azimuths_unit,
            None,
            distances_unit,
            labels,
            labels_unit,
            clip_number_index=None,
            time_step=None,
            confidence=None,
        )

        annotations.validate_array_like(elevations, np.ndarray, float)
        annotations.validate_array_like(azimuths, np.ndarray, float)
        annotations.validate_array_like(distances, np.ndarray, float)
        annotations.validate_lengths_equal(
            [intervals, elevations, azimuths, distances, labels, confidence]
        )
        validate_locations(
            np.concatenate(
                [
                    elevations[:, np.newaxis],
                    azimuths[:, np.newaxis],
                    distances[:, np.newaxis],
                ],
                axis=1,
            )
        )
        self.intervals = intervals
        self.elevations = elevations
        self.azimuths = azimuths
        self.distances = distances
        self.elevations_unit = elevations_unit
        self.azimuths_unit = azimuths_unit
        self.distances_unit = distances_unit


class Clip(core.Clip):
    """TAU SSE 2019 Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        spatial_events (SpatialEvents): sound events with start time, end time, elevation, azimuth, distance, label and confidence.
        audio_path (str): path to the audio file
        set (str): subset the clip belongs to (development or evaluation)
        format (str): whether the clip is in foa or mic format
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
        self.csv_path = self.get_path("events")
        self.format = self._clip_metadata.get("format")
        self.set = self._clip_metadata.get("set")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @core.cached_property
    def spatial_events(self) -> Optional[TAU2019_SpatialEvents]:
        """The clip's spatial events

        Returns:
            * SpatialEvents class with attributes
                * intervals (np.ndarray): (n x 2) array of intervals
                    (as floats) in seconds in the form [start_time, end_time]
                    with positive time stamps and end_time >= start_time.
                * elevations (np.ndarray): (n,) array of elevations
                * azimuths (np.ndarray): (n,) array of azimuths
                * distances (np.ndarray): (n,) array of distances
                * labels (list): list of event labels (as strings)
                * confidence (np.ndarray or None): array of confidence values, float in [0, 1]
                * labels_unit (str): labels unit, one of LABELS_UNITS
                * intervals_unit (str): intervals unit, one of TIME_UNITS
        """
        return load_spatialevents(self.csv_path)

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a TAU SSE 2019 audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate of 48000 without resampling.

    Returns:
        * np.ndarray - the multichannel audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=False)
    return audio, sr


@io.coerce_to_string_io
def load_spatialevents(fhandle: TextIO) -> TAU2019_SpatialEvents:
    """Load an TAU SSE 2019 annotation file
    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file
    Raises:
        IOError: if csv_path doesn't exist
    Returns:
        Events: sound events annotation data
    """

    labels = []
    times = []
    elevations = []
    azimuths = []
    distances = []
    confidence = []
    reader = csv.reader(fhandle, delimiter=",")
    next(reader, None)  # skip header
    for line in reader:
        labels.append(line[0])
        times.append([float(line[1]), float(line[2])])
        elevations.append(float(line[3]))
        azimuths.append(float(line[4]))
        distances.append(float(line[5]))
        confidence.append(1.0)

    events_data = TAU2019_SpatialEvents(
        np.array(times),
        "seconds",
        np.array(elevations),
        "degrees",
        np.array(azimuths),
        "degrees",
        np.array(distances),
        "meters",
        labels,
        "open",
        np.array(confidence),
    )
    return events_data


def validate_locations(locations):
    """Validate if TAU SSE 2019 locations are well-formed.

    If locations is None, validation passes automatically

    Args:
        locations (np.ndarray): (n x 3) array

    Raises:
        ValueError: if locations have an invalid shape or
                have cartesian coordinate values outside the expected ranges.
    """
    if locations is None:
        return

    # validate that locations have the correct shape
    locations_shape = np.shape(locations)
    if len(locations_shape) != 2 or locations_shape[1] != 3:
        raise ValueError(
            f"Locations should be arrays with three columns, but array has shape {locations_shape}"
        )

    # validate that values are within expected ranges
    if (np.abs(locations[:, 0]) > 90).any():
        raise ValueError(f"Elevation values should have magnitude less than 90")
    if (np.abs(locations[:, 1]) > 180).any():
        raise ValueError(f"Azimuth values should have magnitude less than 180")
    elif (locations[:, 2] < 0).any():
        raise ValueError(f"Distance values should be nonnegative numbers")


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The TAU SSE 2019 dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="tau2019sse",
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
        # parsing the data from the filenames due to lack of metadata file
        metadata_index = {}

        with open(self.index_path) as f:
            tausse2019_index = json.load(f)
            all_paths_filenames = list(tausse2019_index["clips"].keys())

        for path_filename in all_paths_filenames:
            clip_id = path_filename
            path, filename = path_filename.split("/")
            fmt, subset = path.split("_")

            metadata_index[clip_id] = {
                "format": fmt,
                "set": subset,
            }

        return metadata_index
