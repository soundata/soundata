"""Sony-TAu Realistic Spatial Soundscapes (STARSS) 2022 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    *Sony-TAu Realistic Spatial Soundscapes: sound scenes in various rooms and environments, together with temporal and spatial annotations of prominent events belonging to a set of target classes.

    *Created By:*
        Archontis Politis, Parthasaarathy Sudarsanam, Sharath Adavanne, Daniel Krause, Tuomas Virtanen
        Audio Research Group, Tampere University (Finland). Version 1.0.0

        Yuki Mitsufuji, Kazuki Shimada, Naoya Takahashi, Yuichiro Koyama, Shusuke Takahashi 
        SONY


    *Description:*
        Contains multichannel recordings of sound scenes in various rooms and environments, 
        together with temporal and spatial annotations of prominent events belonging to a 
        set of target classes. The dataset is collected in two different countries, in Tampere, 
        Finland by the Audio Researh Group (ARG) of Tampere University (TAU), and in Tokyo, 
        Japan by SONY, using a similar setup and annotation procedure. The dataset is delivered 
        in two 4-channel spatial recording formats, a microphone array one (MIC), and first-order 
        Ambisonics one (FOA). These recordings serve as the development dataset for the DCASE 2022 
        Sound Event Localization and Detection Task of the DCASE 2022 Challenge.

        Contrary to the three previous datasets of synthetic spatial sound scenes of 
            * TAU Spatial Sound Events 2019 (development/evaluation), 
            * TAU-NIGENS Spatial Sound Events 2020, and 
            * TAU-NIGENS Spatial Sound Events 2021 
        associated with the previous iterations of the DCASE Challenge, the STARS22 dataset contains 
        recordings of real sound scenes and hence it avoids some of the pitfalls of synthetic 
        generation of scenes. Some such key properties are:
            * annotations are based on a combination of human annotators for sound event activity and optical tracking for spatial positions,
            * the annotated target event classes are determined by the composition of the real scenes,
            * the density, polyphony, occurences and co-occurences of events and sound classes is not random, and it follows actions and interactions of participants in the real scenes.

        The recordings were collected between September 2021 and January 2022. Collection of data 
        from the TAU side has received funding from Google.

    *Audio Files Included:*
    	* 70 recording clips of 30 sec ~ 5 min durations, with a total time of ~2hrs, contributed by SONY (development dataset).
    	* 51 recording clips of 1 min ~ 5 min durations, with a total time of ~3hrs, contributed by TAU (development dataset).
    	* 40 recordings contributed by SONY for the training split, captured in 2 rooms (dev-train-sony).
    	* 30 recordings contributed by SONY for the testing split, captured in 2 rooms (dev-test-sony).
    	* 27 recordings contributed by TAU for the training split, captured in 4 rooms (dev-train-tau).
    	* 24 recordings contributed by TAU for the testing split, captured in 3 rooms (dev-test-tau).
    	* A total of 11 unique rooms captured in the recordings, 4 from SONY and 7 from TAU (development set).
    	* Sampling rate 24kHz.
    	* Two 4-channel 3-dimensional recording formats: first-order Ambisonics (FOA) and tetrahedral microphone array (MIC).
    	* Recordings are taken in two different countries and two different sites.
    	* Each recording clip is part of a recording session happening in a unique room.
    	* Groups of participants, sound making props, and scene scenarios are unique for each session (with a few exceptions).
    	* 13 target classes are identified in the recordings and strongly annotated by humans.
    	* Spatial annotations for those active events are captured by an optical tracking system.
    	* Sound events out of the target classes are considered as interference and are not labeled.

    *Annotations Included:*
        * Each recording in the development set has labels of events and DoAs in a plain csv file with the same filename.
        * Each row in the csv file has a frame number, active class index, source number index, azimuth, and elevation.
        * Frame, class, and source enumeration begins at 0. 
        * Frames correspond to a temporal resolution of 100msec. 
        * Azimuth and elevation angles are given in degrees, rounded to the closest integer value, with azimuth and elevation being zero at the front, azimuth :math:`\phi \in [-180^{\circ}, 180^{\circ}]`, and elevation :math:`\\theta \in [-90^{\circ}, 90^{\circ}]`. Note that the azimuth angle is increasing counter-clockwise (:math:`\phi = 90^{\circ}` at the left).
	* The source index is a unique integer for each source in the scene, and it is provided only as additional information. Note that each unique actor gets assigned one such identifier, but not individual events produced by the same actor; e.g. a clapping event and a laughter event produced by the same person have the same identifier. Independent sources that are not actors (e.g. a loudspeaker playing music in the room) get a 0 identifier. Note that source identifier information is only included in the development metadata and is not required to be provided by the participants in their results.
        * Overlapping sound events are indicated with duplicate frame numbers, and can belong to a different or the same class.

    *Organization*
        * The development dataset is split in training and test sets.
        * The training set consists of 67 recordings.
        * The test set consists of 54 recordings.

    *Please Acknowledge Sony-TAu Realistic Spatial Soundscapes (STARSS) 2022 in Academic Research:*
        * If you use this dataset please cite the report on its creation, and the corresponding DCASE2022 task setup:

            * Politis, Adavanne, Mitsufuji, Yuki, Sudarsanam, Parthasaarathy, Shimada, Kazuki, Adavanne, Sharath, Koyama, Yuichiro, Krause, Daniel, Takahashi, Naoya, Takahashi, Shusuke, & Virtanen, Tuomas. (2022). STARSS22: Sony-TAu Realistic Spatial Soundscapes 2022 dataset (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6387880 

    *License:*
        * This datast is licensed under the [MIT](https://opensource.org/licenses/MIT) license
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import jams
import json
import glob
import numbers
from itertools import cycle

from soundata import download_utils, jams_utils, core, annotations, io

BIBTEX = """
@dataset{politis_adavanne_2022_6387880,
  author       = {Politis, Adavanne and
                  Mitsufuji, Yuki and
                  Sudarsanam, Parthasaarathy and
                  Shimada, Kazuki and
                  Adavanne, Sharath and
                  Koyama, Yuichiro and
                  Krause, Daniel and
                  Takahashi, Naoya and
                  Takahashi, Shusuke and
                  Virtanen, Tuomas},
  title        = {{STARSS22: Sony-TAu Realistic Spatial Soundscapes 
                   2022 dataset}},
  month        = mar,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.6387880},
  url          = {https://doi.org/10.5281/zenodo.6387880}
}
"""

REMOTES = {
    "foa_dev": download_utils.RemoteFileMetadata(
        filename="foa_dev.zip",
        url="https://zenodo.org/record/6387880/files/foa_dev.zip?download=1",
        checksum="165dd033b262dc11a8853635c1def59b",
    ),
    "mic_dev": download_utils.RemoteFileMetadata(
        filename="mic_dev.zip",
        url="https://zenodo.org/record/6387880/files/mic_dev.zip?download=1",
        checksum="46b55d0be507afa986cd29120e42b188",
    ),
    "metadata_dev": download_utils.RemoteFileMetadata(
        filename="metadata_dev.zip",
        url="https://zenodo.org/record/6387880/files/metadata_dev.zip?download=1",
        checksum="b460e17e0848c49f03f238afb89fa87e",
    ),
}

LICENSE_INFO = """
This datast is licensed under the [MIT](https://opensource.org/licenses/MIT) license
"""

#: position units
ELEVATIONS_UNITS = {
    "degrees": "degrees",
}
AZIMUTHS_UNITS = {
    "degrees": "degrees",
}
DISTANCES_UNITS = {
    "meters": "meters",
}

#: Time units
TIME_UNITS = {
    "seconds": "seconds",
    "miliseconds": "miliseconds",
}

#: Label units
LABEL_UNITS = {"open": "no strict schema or units"}


class SpatialEvents:
    """SpatialEvents class
    Attributes:
        intervals (list): list of size n np.ndarrays of shape (m, 2), with intervals
            (as floats) in TIME_UNITS in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
            n is the number of sound events.
            m is the number of sounding instances for each sound event.
        intervals_unit (str): intervals unit, one of TIME_UNITS
        time_step (int, float, or None): the time-step between events
            over time in intervals_unit
        elevations (list): list of size n with np.ndarrays with dtype int,
            indicating the elevation of the sound event per time_step if moving
            or a single value if static. Values between -90 and 90
        elevations_unit (str): elevations unit, one of ELEVATIONS_UNITS
        azimuths (list): list of size n with np.ndarrays with dtype int,
            indicating the azimuth of the sound event per time_step if moving
            or a single value if static. Values between -180 and 180
        azimuths_unit (str): azimuths unit, one of AZIMUTHS_UNITS
        distances (list): list of size n with np.ndarrays with dtype int,
            indicating the distance of the sound event per time_step if moving
            or a single value if static. Values must be positive or None
        distances_unit (str): distances unit, one of DISTANCES_UNITS
        labels (list): list of event labels (as strings)
        labels_unit (str): labels unit, one of LABELS_UNITS
        track_number_indices (list): list of track number indices (as strings)
        confidence (np.ndarray or None): array of confidence values, float in [0, 1]
    """

    def __init__(
        self,
        intervals,
        intervals_unit,
        time_step,
        elevations,
        elevations_unit,
        azimuths,
        azimuths_unit,
        distances,
        distances_unit,
        labels,
        labels_unit,
        track_number_index,
        confidence=None,
    ):
        annotations.validate_array_like(intervals, list, list)
        annotations.validate_array_like(labels, list, str)
        annotations.validate_array_like(
            confidence, np.ndarray, float, none_allowed=True
        )
        [
            [
                annotations.validate_intervals(intervals[np.newaxis, :])
                for intervals in event_intervals
            ]
            for event_intervals in intervals
        ]
        annotations.validate_confidence(confidence)
        annotations.validate_unit(labels_unit, LABEL_UNITS)
        annotations.validate_unit(intervals_unit, TIME_UNITS)

        self.intervals = intervals
        self.intervals_unit = intervals_unit
        self.labels = labels
        self.labels_unit = labels_unit
        self.confidence = confidence

        annotations.validate_array_like(track_number_index, list, str)
        annotations.validate_array_like(elevations, list, list)
        annotations.validate_array_like(azimuths, list, list)
        annotations.validate_array_like(distances, list, list)
        annotations.validate_lengths_equal(
            [
                intervals,
                elevations,
                azimuths,
                distances,
                labels,
                track_number_index,
                confidence,
            ]
        )
        # validate location information for each event are numpy arrays
        [
            [
                [
                    annotations.validate_array_like(subitem, np.ndarray, int)
                    for subitem in sitem
                ]
                for sitem in item
            ]
            for item in [elevations, azimuths]
        ]
        [
            [
                annotations.validate_array_like(
                    subitem, np.ndarray, np.array([None]).dtype, none_allowed=True
                )
                for subitem in item
            ]
            for item in distances
        ]
        # validate length of location information is consistent
        # for each event
        [
            [
                annotations.validate_lengths_equal([e, a, d])
                for e, a, d in zip(els, azs, dis)
            ]
            for els, azs, dis in zip(elevations, azimuths, distances)
        ]
        [
            [
                validate_locations(
                    np.concatenate(
                        [e[:, np.newaxis], a[:, np.newaxis], d[:, np.newaxis]], axis=1
                    )
                )
                for e, a, d in zip(els, azs, dis)
            ]
            for els, azs, dis in zip(elevations, azimuths, distances)
        ]
        [
            [
                validate_time_steps(
                    time_step,
                    np.concatenate(
                        [e[:, np.newaxis], a[:, np.newaxis], d[:, np.newaxis]], axis=1
                    ),
                    i,
                )
                for e, a, d, i in zip(els, azs, dis, ivl)
            ]
            for els, azs, dis, ivl in zip(elevations, azimuths, distances, intervals)
        ]

        annotations.validate_unit(elevations_unit, ELEVATIONS_UNITS)
        annotations.validate_unit(azimuths_unit, AZIMUTHS_UNITS)
        annotations.validate_unit(distances_unit, DISTANCES_UNITS)

        self.time_step = time_step
        self.elevations = elevations
        self.azimuths = azimuths
        self.distances = distances
        self.track_number_index = track_number_index
        self.elevations_unit = elevations_unit
        self.azimuths_unit = azimuths_unit
        self.distances_unit = distances_unit


def validate_time_steps(time_step, locations, interval):
    """Validate if STARSS 2022 timesteps are well-formed.
    If locations is None, validation passes automatically
    Args:
        time_step (float): spacing between location steps
        locations (np.ndarray): (n x 3) array
        interval (np.ndarray): (n x 2) expected start and end time
            for the locations
    Raises:
        ValueError: if the number of locations does not match
            the number of time_steps that fit in the interval
    """
    if interval[0] > interval[1]:
        raise ValueError("The interval has a start_time greater than the end_time")
    elif len(locations) == 1 and interval[1] - interval[0] > 0:
        pass  # the event is static
    # if the object is static, validation passes
    elif not np.isclose(len(locations) - 1, (interval[1] - interval[0]) / time_step):
        raise ValueError(
            "The number of locations does not fit in the interval, given the time_step"
        )


def validate_locations(locations):
    """Validate if STARSS 2022 locations are well-formed.
    If locations is None, validation passes automatically
    Args:
        locations (np.ndarray): (n x 3) array
    Raises:
        ValueError: if locations have an invalid shape or
                have cartesian coordinate values outside the expected ranges.
    """

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
    elif (locations[:, 2] != None).any():
        raise ValueError(f"Distance values should be nonnegative numbers")


class Clip(core.Clip):
    """STARSS 2022 Track class
    Args:
        clip_id (str): id of the clip
    Attributes:
        audio_path (str): path to the audio file
        csv_path (str): path to the csv file
        format (str): whether the clip is in FOA or MIC format
        set (str): the data subset the clip belongs to (development or evaluation)
        split (str): the set slip the clip belongs to (training or test)
        track_id (str): track id
        spatial_events (SpatialEvents): sound events with time step, elevation, azimuth, distance, label, track_number and confidence.
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):

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
        self.split = self._clip_metadata.get("split")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio
        Returns:
            * np.ndarray - audio signal
            * float - sample rate
        """
        return load_audio(self.audio_path)

    @core.cached_property
    def spatial_events(self) -> Optional[SpatialEvents]:
        """The clip's event annotations
        Returns:
            * SpatialEvents with attributes
                * intervals (list): list of size n np.ndarrays of shape (m, 2), with intervals
                    (as floats) in TIME_UNITS in the form [start_time, end_time]
                * intervals_unit (str): intervals unit, one of TIME_UNITS
                * time_step (int, float, or None): the time-step between events
                * elevations (list): list of size n with np.ndarrays with dtype int,
                    indicating the elevation of the sound event per time_step.
                * elevations_unit (str): elevations unit, one of ELEVATIONS_UNITS
                * azimuths (list): list of size n with np.ndarrays with dtype int,
                    indicating the azimuth of the sound event per time_step if moving
                * azimuths_unit (str): azimuths unit, one of AZIMUTHS_UNITS
                * distances (list): list of size n with np.ndarrays with dtype int,
                    indicating the distance of the sound event per time_step if moving
                * distances_unit (str): distances unit, one of DISTANCES_UNITS
                * labels (list): list of event labels (as strings)
                * labels_unit (str): labels unit, one of LABELS_UNITS
                * track_number_indices (list): list of track number indices (as strings)
                * confidence (np.ndarray or None): array of confidence values

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
def load_audio(fhandle: BinaryIO, sr=24000) -> Tuple[np.ndarray, float]:
    """Load a STARSS 2022 audio file.
    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file
        sr (int or None): sample rate for loaded audio, 24000 Hz by default.
        If different from file's sample rate it will be resampled on load.
        Use None to load the file using its original sample rate (24000)
    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file
    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=False)
    return audio, sr


@io.coerce_to_string_io
def load_spatialevents(fhandle: TextIO, dt=0.1) -> SpatialEvents:
    """Load a STARSS 2022 annotation file
    Args:
        fhandle (str or file-like): File-like object or path to
            the sound events annotation file
        dt (float): time step
    Raises:
        IOError: if fhandle doesn't exist
    Returns:
        SpatialEvents: sound spatial events annotation data
    """

    def _process_raw_events(raw_reader, dt):

        # unpack columns in csv
        time_frames, labels, event_num, azimuths, elevations = list(
            map(list, zip(*raw_reader))
        )

        # find unique label+event_num pairs
        # processing as dictionary to preserve order
        unique_events_set = list({l_e: None for l_e in zip(labels, event_num)})

        # find all the indices of each unique label+event_num pair
        unique_events_indices = [
            (np.array(list(zip(labels, event_num))) == event).all(axis=1).nonzero()[0]
            for event in unique_events_set
        ]

        # get sets of continuous indices for each unique label+event_num pair
        unique_events_indices_grouped = [
            np.split(
                event_indices,
                np.where(np.diff(np.array(time_frames)[event_indices]) != 1)[0] + 1,
            )
            for event_indices in unique_events_indices
        ]

        # get start_time end_time pairs for all events
        intervals = [
            [
                np.round(
                    np.array(
                        [
                            np.array(time_frames)[indices_grouped[0]],
                            np.array(time_frames)[indices_grouped[-1]],
                        ]
                    )
                    * dt,
                    decimals=1,
                )
                for indices_grouped in unique_event_indices_grouped
            ]
            for unique_event_indices_grouped in unique_events_indices_grouped
        ]

        # get azimuth arrays for all event instances
        azimuths = [
            [
                np.array(azimuths)[indices_grouped]
                for indices_grouped in unique_event_indices_grouped
            ]
            for unique_event_indices_grouped in unique_events_indices_grouped
        ]

        # get elevations arrays for all event instances
        elevations = [
            [
                np.array(elevations)[indices_grouped]
                for indices_grouped in unique_event_indices_grouped
            ]
            for unique_event_indices_grouped in unique_events_indices_grouped
        ]

        # keep only one value if the event is static
        azimuths_elevations = [
            [
                np.array([azimuth[0], elevation[0]])
                if (azimuth == azimuth[0]).all() and (elevation == elevation[0]).all()
                else np.concatenate(
                    [azimuth[:, np.newaxis], elevation[:, np.newaxis]], axis=1
                )
                for azimuth, elevation in zip(event_azimuths, event_elevations)
            ]
            for event_azimuths, event_elevations in zip(azimuths, elevations)
        ]

        # separate azimuths and elevations again
        azimuths = [
            [
                azimuth_elevation[:, 0]
                if len(azimuth_elevation.shape) == 2
                else np.array([azimuth_elevation[0]])
                for azimuth_elevation in event_azimuths_elevations
            ]
            for event_azimuths_elevations in azimuths_elevations
        ]
        elevations = [
            [
                azimuth_elevation[:, 1]
                if len(azimuth_elevation.shape) == 2
                else np.array([azimuth_elevation[1]])
                for azimuth_elevation in event_azimuths_elevations
            ]
            for event_azimuths_elevations in azimuths_elevations
        ]

        # list of labels and track_number_indices in str
        labels, track_number_indices = list(zip(*unique_events_set))
        labels = [str(l) for l in labels]
        track_number_indices = [str(l) for l in track_number_indices]

        # create dummy distances with None
        distances = [
            [np.array([None] * len(azimuth)) for azimuth in event_azimuths]
            for event_azimuths in azimuths
        ]

        return intervals, labels, track_number_indices, azimuths, elevations, distances

    raw_reader = csv.reader(fhandle, delimiter=",")
    raw_events = []
    for line in raw_reader:
        raw_events.append([int(val) for val in line])
    (
        intervals,
        labels,
        track_number_indices,
        azimuths,
        elevations,
        distances,
    ) = _process_raw_events(raw_events, dt)
    confidence = np.array([1.0] * len(labels))

    events_data = SpatialEvents(
        intervals,
        "seconds",
        dt,
        elevations,
        "degrees",
        azimuths,
        "degrees",
        distances,
        "meters",
        labels,
        "open",
        track_number_indices,
        confidence,
    )

    return events_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The STARSS 2022 dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="starss2022",
            clip_class=Clip,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.cached_property
    def _metadata(self):

        # parsing the data from the filenames due to lack of metadata file
        json_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "indexes/starss2022_index.json",
        )

        metadata_index = {}

        with open(json_path) as f:
            starss2022_index = json.load(f)
            all_paths_filenames = list(starss2022_index["clips"].keys())

        for path_filename in all_paths_filenames:

            clip_id = path_filename
            path, split, filename = path_filename.split("/")
            fmt, subset = path.split("_")
            _, split, site = split.split("-")

            metadata_index[clip_id] = {"format": fmt, "set": subset, "split": split}

        return metadata_index
