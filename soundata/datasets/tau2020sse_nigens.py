"""TAU NIGENS SSE 2020 Dataset Loader


.. admonition:: Dataset Info
    :class: dropdown
    
    
    *TAU NIGENS Spatial Sound Events: scene recordings with (moving) sound events of distinct categories*
    
    *Created By:*
        Archontis Politis, Sharath Adavanne, Tuomas Virtanen.
        Audio Research Group, Tampere University (Finland). Version 1.2.0
    *Description:*
        Spatial sound-scene recordings, consisting of sound events of distinct categories 
        in a variety of acoustical spaces, and from multiple source directions and distances.
        The spatialization of all sound events is based on filtering through real spatial 
        room impulse responses (RIRs) of diverse acoustic environments. The sound events are 
        spatialized as either stationary sound sources, or moving sound sources, in which case 
        time-variant RIRs are used. 
        Each scene recording is delivered in microphone array (MIC) and first-order Ambisonics (FOA) 
        format. 
    *Audio Files Included:*
   	* 600 one-minute long sound scene recordings (development dataset).
   	* 200 one-minute long sound scene recordings (evaluation dataset).
        * Sampling rate is 24 kHz (16-bit signed integer PCM).
   	* About 700 sound event samples spread over 14 classes (see here for more details).
   	* 8 provided cross-validation folds of 100 recordings each, with unique sound event samples and rooms in each of them.
   	* Two 4-channel 3-dimensional recording formats: first-order Ambisonics (FOA) and tetrahedral microphone array.
   	* Realistic spatialization and reverberation through RIRs collected in 15 different enclosures.
   	* From about 1500 to 3500 possible RIR positions across the different rooms.
   	* Both static reverberant and moving reverberant sound events.
   	* Up to two overlapping sound events allowed, temporally and spatially.
   	* Realistic spatial ambient noise collected from each room is added to the spatialized sound events, at varying signal-to-noise ratios (SNR) ranging from noiseless (30dB) to noisy (6dB).
    *Annotations Included:*
        * Each recording in the development set has labels of events and Directions of arrival in a plain csv file with the same filename.
        * Each row in the csv file has a frame number, active class index, track number index, azimuth, and elevation.
        * Frame, class, and track enumeration begins at 0. 
        * Frames correspond to a temporal resolution of 100msec. 
        * Azimuth and elevation angles are given in degrees, rounded to the closest integer value, with azimuth and elevation being zero at the front, azimuth :math:`\phi \in [-180^{\circ}, 180^{\circ}]`, and elevation :math:`\\theta \in [-90^{\circ}, 90^{\circ}]`. Note that the azimuth angle is increasing counter-clockwise (:math:`\phi = 90^{\circ}` at the left).
        * The event number index is a unique integer for each event in the recording, enumerating them in the order of appearance. This event identifiers are useful to disentangle directions of co-occuring events through time in the metadata file. 
        * Overlapping sound events are indicated with duplicate frame numbers, and can belong to a different or the same class.
    *Please Acknowledge TAU-NIGENS SSE 2020 in Academic Research:*
        * If you use this dataset please cite the report on its creation, and the corresponding DCASE2020 task setup:
            * Politis., Archontis, Adavanne, Sharath, & Virtanen, Tuomas (2020). A Dataset of Reverberant Spatial Sound Scenes with Moving Sources for Sound Event Localization and Detection. In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020), Tokyo, Japan.
    *License:*
        * Creative Commons Attribution Non Commercial 4.0 International
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
@inproceedings{politis2020dataset,
    author = "Politis, Archontis and Adavanne, Sharath and Virtanen, Tuomas",
    title = "A Dataset of Reverberant Spatial Sound Scenes with Moving Sources for Sound Event Localization and Detection",
    year = "2020",
    booktitle = "Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE2020)",
    month = "November",
}
"""

REMOTES = {
    "foa_dev": [
        download_utils.RemoteFileMetadata(
            filename="foa_dev.zip",
            url="http://zenodo.org/record/4064792/files/foa_dev.zip?download=1",
            checksum="6aad48e7346884b3929245e7553fd97d",
        ),
        download_utils.RemoteFileMetadata(
            filename="foa_dev.z01",
            url="http://zenodo.org/record/4064792/files/foa_dev.z01?download=1",
            checksum="86acab46854a57f5ba3e5b80a19c01b5",
        ),
        download_utils.RemoteFileMetadata(
            filename="foa_dev.z02",
            url="http://zenodo.org/record/4064792/files/foa_dev.z02?download=1",
            checksum="363c8c159be003271c05a71a57b2ced4",
        ),
    ],
    "mic_dev": [
        download_utils.RemoteFileMetadata(
            filename="mic_dev.zip",
            url="http://zenodo.org/record/4064792/files/mic_dev.zip?download=1",
            checksum="9174daca52f393425120308ab5c14477",
        ),
        download_utils.RemoteFileMetadata(
            filename="mic_dev.z01",
            url="http://zenodo.org/record/4064792/files/mic_dev.z01?download=1",
            checksum="3a2b0986d2a302498cd874d584d17689",
        ),
        download_utils.RemoteFileMetadata(
            filename="mic_dev.z02",
            url="http://zenodo.org/record/4064792/files/mic_dev.z02?download=1",
            checksum="92f715cb74406d5556bce0fdf27f54e4",
        ),
    ],
    "foa_eval": download_utils.RemoteFileMetadata(
        filename="foa_eval.zip",
        url="http://zenodo.org/record/4064792/files/foa_eval.zip?download=1",
        checksum="24c6ce2441df242d4e3b61e9bb27d0d7",
    ),
    "mic_eval": download_utils.RemoteFileMetadata(
        filename="mic_eval.zip",
        url="http://zenodo.org/record/4064792/files/mic_eval.zip?download=1",
        checksum="bca79b5f71b46e4cb191c54a611348a4",
    ),
    "metadata_dev": download_utils.RemoteFileMetadata(
        filename="metadata_dev.zip",
        url="http://zenodo.org/record/4064792/files/metadata_dev.zip?download=1",
        checksum="979f5551e987ed247404b80a2f1c3db1",
    ),
    "metadata_eval": download_utils.RemoteFileMetadata(
        filename="metadata_eval.zip",
        url="http://zenodo.org/record/4064792/files/metadata_eval.zip?download=1",
        checksum="f3584166d9a63b43c1e301b6fb722293",
    ),
}

LICENSE_INFO = """
Creative Commons Attribution Non Commercial 4.0 International
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
    """Validate if TAU NIGENS SSE 2020 timesteps are well-formed.
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
    """Validate if TAU NIGENS SSE 2020 locations are well-formed.
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
    """TAU NIGENS SSE 2020 Track class
    Args:
        clip_id (str): id of the clip
    Attributes:
        audio_path (str): path to the audio file
        tags (soundata.annotation.Tags): tag
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
    """Load a TAU NIGENS SSE 2020 audio file.
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
    """Load an TAU NIGENS SSE 2020 annotation file
    Args:
        fhandle (str or file-like): File-like object or path to
            the sound events annotation file
        dt (float): time step
    Raises:
        IOError: if txt_path doesn't exist
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
    The TAU NIGENS SSE 2020 dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="tau2020sse_nigens",
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
            "indexes/tau2020sse_nigens_index.json",
        )

        metadata_index = {}

        with open(json_path) as f:
            taunigenssse2020_index = json.load(f)
            all_paths_filenames = list(taunigenssse2020_index["clips"].keys())

        for path_filename in all_paths_filenames:

            clip_id = path_filename
            path, filename = path_filename.split("/")
            fmt, subset = path.split("_")

            metadata_index[clip_id] = {
                "format": fmt,
                "set": subset,
            }

        return metadata_index
