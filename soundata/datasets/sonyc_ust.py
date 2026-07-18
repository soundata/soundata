"""SONYC-UST Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    SONYC-UST
    =========

    Created by
    ----------
    
    Mark Cartwright (1,2,3), Jason Cramer (1), Ana Elisa Mendez Mendez (1), Yu Wang (1), Ho-Hsiang Wu (1), Vincent Lostanlen (1,2,4), Magdalena Fuentes (1), Graham Dove (2), Charlie Mydlarz (1, 2), Justin Salamon (1,5), Oded Nov (6), Juan Pablo Bello (1,2,3)

    1. Music and Audio Resarch Lab, New York University
    2. Center for Urban Science and Progress, New York University
    3. Department of Computer Science and Engineering, New York University
    4. Cornell Lab of Ornithology
    5. Adobe Research
    6. Department of Technology Management and Innovation, New York University

    Version 2.3
    -----------
    - Added the ground truth annotations for the test set, and regrouped the audio files for upload to Zenodo.

    Publication
    -----------
    If using this data in an academic work, please reference the DOI and version, as well as cite the following paper, which presented the data collection procedure and the first version of the dataset:

    Cartwright, M., Cramer, J., Mendez, A.E.M., Wang, Y., Wu, H., Lostanlen, V., Fuentes, M., Dove, G., Mydlarz, C., Salamon, J., Nov, O., Bello, J.P. SONYC-UST-V2: An Urban Sound Tagging Dataset with Spatiotemporal Context. In *Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE)*, 2020.
    [pdf](https://arxiv.org/abs/2009.05188)

    Description
    -----------

    SONYC Urban Sound Tagging (SONYC-UST) is a dataset for the development and evaluation of machine listening systems for realistic urban noise monitoring. The audio was recorded from the [SONYC](https://wp.nyu.edu/sonyc) acoustic sensor network. Volunteers on the  [Zooniverse](https://zooniverse.org) citizen science platform tagged the presence of 23 classes that were chosen in consultation with the New York City Department of Environmental Protection. These 23 fine-grained classes can be grouped into 8 coarse-grained classes. The recordings are split into three sets: training, validation, and test. The training and validation sets are disjoint with respect to the sensor from which each recording came, and the test set is displaced in time. For increased reliability, three volunteers annotated each recording. In addition, members of the SONYC team subsequently created a subset of verified, ground-truth tags using a two-stage annotation procedure in which two annotators independently tagged and then collectively resolved any disagreements. This subset of recordings with verified annotations intersects with all three recording splits. All of the recordings in the test set have these verified annotations. In v2 version of this dataset, we have also included coarse spatiotemporal context information to aid in tag prediction when time and location is known. For more details on the motivation and creation of this dataset see the [DCASE 2020 Urban Sound Tagging with Spatiotemporal Context Task website](http://dcase.community/challenge2020/task-urban-sound-tagging-with-spatiotemporal-context).
    
    Audio data
    ----------

    The provided audio has been acquired using the SONYC acoustic sensor network for urban noise pollution monitoring. Over 60 different sensors have been deployed in New York City, and these sensors have collectively gathered the equivalent of over 50 years of audio data, of which we provide a small subset. The data was sampled by selecting the nearest neighbors on VGGish features of recordings known to have classes of interest. All recordings are 10 seconds and were recorded with identical microphones at identical gain settings. To maintain privacy, we quantized the spatial information to the level of a city block, and we quantized the temporal information to the level of an hour. We also limited the occurrence of recordings with positive human voice annotations to one per hour per sensor.

    Label taxonomy
    --------------

    The label taxonomy is as follows:
    1. engine
        1: small-sounding-engine
        2: medium-sounding-engine
        3: large-sounding-engine
        X: engine-of-uncertain-size
    2. machinery-impact
        1: rock-drill
        2: jackhammer
        3: hoe-ram
        4: pile-driver
        X: other-unknown-impact-machinery
    3. non-machinery-impact
        1: non-machinery-impact
    4. powered-saw
        1: chainsaw
        2: small-medium-rotating-saw
        3: large-rotating-saw
        X: other-unknown-powered-saw
    5. alert-signal
        1: car-horn
        2: car-alarm
        3: siren
        4: reverse-beeper
        X: other-unknown-alert-signal
    6. music
        1: stationary-music
        2: mobile-music
        3: ice-cream-truck
        X: music-from-uncertain-source
    7. human-voice
        1: person-or-small-group-talking
        2: person-or-small-group-shouting
        3: large-crowd
        4: amplified-speech
        X: other-unknown-human-voice
    8. dog
        1: dog-barking-whining

    The classes preceded by an `X` code indicate when an annotator was able to identify the coarse class, but couldn't identify the fine class because either they were uncertain which fine class it was or the fine class was not included in the taxonomy. `dcase-ust-taxonomy.yaml` contains this taxonomy in an easily machine-readable form.

    Data splits
    -----------
    
    This release contains a training subset (13538 recordings from 35 sensors), and validation subset (4308 recordings from 9 sensors), and a test subset (664 recordings from 48 sensors). The training and validation subsets are disjoint with respect to the sensor from which each recording came. The sensors in the test set will not disjoint from the training and validation subsets, but the test recordings are displaced in time, occurring after any of the recordings in the training and validation subset. The subset of recordings with verified annotations (1380 recordings) intersects with all three recording splits. All of the recordings in the test set have these verified annotations. 

    Annotation data
    ---------------

    The annotation data are contained in annotations.csv, and encompass the training, validation, and test subsets. Each row in the file represents one multi-label annotation of a recording---it could be the annotation of a single citizen science volunteer, a single SONYC team member, or the agreed-upon ground truth by the SONYC team (see the annotator_id column description for more information). Note that since the SONYC team members annotated each class group separately, there may be multiple annotation rows by a single SONYC team annotator for a particular audio recording.

    Columns
    -------

    *split*

    : The data split. (*train*, *validate*)

    *sensor\_id*

    : The ID of the sensor the recording is from.

    *audio\_filename*
    : The filename of the audio recording

    *annotator\_id*
    : The anonymous ID of the annotator. If this value is positive, it is a citizen science volunteer from the Zooniverse platform. If it is negative, it is a SONYC team member. If it is `0`, then it is the ground truth agreed-upon by the SONYC team.

    *year*
    : The year the recording is from.

    *week*
    : The week of the year the recording is from.

    *day*
    : The day of the week the recording is from, with Monday as the start (i.e. `0`=Monday).

    *hour*
    : The hour of the day the recording is from

    *borough*
    : The NYC borough in which the sensor is located (`1`=Manhattan, `3`=Brooklyn, `4`=Queens). This corresponds to the first digit in the 10-digit NYC parcel number system known as Borough, Block, Lot (BBL).

    *block*
    : The NYC block in which the sensor is located. This corresponds to digits 2—6 digit in the 10-digit NYC parcel number system known as Borough, Block, Lot (BBL).

    *latitude*
    : The latitude coordinate of the **block** in which the sensor is located.

    *longitude*
    : The longitude coordinate of the **block** in which the sensor is located.

    *<coarse\_id\>-<fine_id\>\_<fine_name\>_presence*
    : Columns of this form indicate the presence of fine-level class. `1` if present, `0` if not present. If `-1`, then the class was not labeled in this annotation because the annotation was performed by a SONYC team member who only annotated one coarse group of classes at a time when annotating the verified subset.

    *<coarse\_id\>\_<coarse\_name\>\_presence*
    : Columns of this form indicate the presence of a coarse-level class. `1` if present, `0` if not present. If `-1`, then the class was not labeled in this annotation because the annotation was performed by a SONYC team member who only annotated one coarse group of classes at a time when annotating the verified subset. These columns are computed from the fine-level class presence columns and are presented here for convenience when training on only coarse-level classes.

    *<coarse\_id\>-<fine_id\>\_<fine_name\>\_proximity*
    : Columns of this form indicate the proximity of a fine-level class. After indicating the presence of a fine-level class, citizen science annotators were asked to indicate the proximity of the sound event to the sensor. Only the citizen science volunteers performed this task, and therefore this data is not included in the verified annotations. This column may take on one of the following four values: (`near`, `far`, `notsure`, `-1`). If `-1`, then the proximity was not annotated because either the annotation was not performed by a citizen science volunteer, or the citizen science volunteer did not indicate the presence of the class.


    Conditions of Use
    -----------------

    Dataset created by Mark Cartwright, Jason Cramer, Ana Elisa Mendez Mendez, Yu Wang, Ho-Hsiang Wu, Vincent Lostanlen, Magdalena Fuentes, Graham Dove, Charlie Mydlarz, Justin Salamon, Oded Nov, and Juan Pablo Bello.

    The SONYC-UST dataset is offered free of charge under the terms of the Creative  Commons Attribution 4.0 International (CC BY 4.0) license: https://creativecommons.org/licenses/by/4.0/

    The dataset and its contents are made available on an "as is" basis and without  warranties of any kind, including 
    without limitation satisfactory quality and  conformity, merchantability, fitness for a particular purpose, accuracy or 
    completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, New York University is not 
    liable for, and expressly excludes all liability for, loss or damage however and whenever caused to anyone by any use of the SONYC-UST dataset or any part of it.

    Feedback
    --------

    Please help us improve SONYC-UST by sending your feedback to: mcartwright@gmail.com
    In case of a problem, please include as many details as possible.
"""


import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import jams
import glob

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io

BIBTEX = """
@inproceedings{cartwright2019sonyc,
    Author = "Cartwright, Mark and Mendez, Ana Elisa Mendez and Cramer, Jason and Lostanlen, Vincent and Dove, Graham and Wu, Ho-Hsiang and Salamon, Justin and Nov, Oded and Bello, Juan",
    title = "{SONYC} Urban Sound Tagging ({SONYC-UST}): A Multilabel Dataset from an Urban Acoustic Sensor Network",
    year = "2019",
    booktitle = "Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE)",
    month = "October",
    pages = "35--39",
}

"""
REMOTES = {
    "audio-0": download_utils.RemoteFileMetadata(
        filename="audio-0.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-0.tar.gz?download=1",
        checksum="bbb4dbae7d2e58e18d24878b9ee1eb51",
    ),
    "audio-1": download_utils.RemoteFileMetadata(
        filename="audio-1.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-1.tar.gz?download=1",
        checksum="7c369ff37ac6a1fdd8493fdffe62e0a1",
    ),
    "audio-2": download_utils.RemoteFileMetadata(
        filename="audio-2.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-2.tar.gz?download=1",
        checksum="412241c063d7f196953d3dd1c44aeb5e",
    ),
    "audio-3": download_utils.RemoteFileMetadata(
        filename="audio-3.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-3.tar.gz?download=1",
        checksum="d2a872392d95993a2c238d305b940812",
    ),
    "audio-4": download_utils.RemoteFileMetadata(
        filename="audio-4.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-4.tar.gz?download=1",
        checksum="b15b0e0cb3f8584259ef0d24293a9be3",
    ),
    "audio-5": download_utils.RemoteFileMetadata(
        filename="audio-5.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-5.tar.gz?download=1",
        checksum="de745f887067433757a2f2c8f99f99bb",
    ),
    "audio-6": download_utils.RemoteFileMetadata(
        filename="audio-6.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-6.tar.gz?download=1",
        checksum="b0286c67468369d66336f6f5ddede31f",
    ),
    "audio-7": download_utils.RemoteFileMetadata(
        filename="audio-7.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-7.tar.gz?download=1",
        checksum="ff300183ab7a9d3f2f74c3b730ffeb52",
    ),
    "audio-8": download_utils.RemoteFileMetadata(
        filename="audio-8.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-8.tar.gz?download=1",
        checksum="7be76b821fa6dbe20f6b50ca440e1024",
    ),
    "audio-9": download_utils.RemoteFileMetadata(
        filename="audio-9.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-9.tar.gz?download=1",
        checksum="959f7edfbb26eadad9865069f38aa9dd",
    ),
    "audio-10": download_utils.RemoteFileMetadata(
        filename="audio-10.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-10.tar.gz?download=1",
        checksum="84e94de273666c537a3e6b709ee88d9b",
    ),
    "audio-11": download_utils.RemoteFileMetadata(
        filename="audio-11.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-11.tar.gz?download=1",
        checksum="44fe1f43121a7d1178aa0cdf7477793b",
    ),
    "audio-12": download_utils.RemoteFileMetadata(
        filename="audio-12.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-12.tar.gz?download=1",
        checksum="a7eb21011f460b5ac289f8e285e875ab",
    ),
    "audio-13": download_utils.RemoteFileMetadata(
        filename="audio-13.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-13.tar.gz?download=1",
        checksum="27945a8b1008eec787f97213164951e6",
    ),
    "audio-14": download_utils.RemoteFileMetadata(
        filename="audio-14.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-14.tar.gz?download=1",
        checksum="fb8c7d81c3bde3a0c86c1e55e6a85a24",
    ),
    "audio-15": download_utils.RemoteFileMetadata(
        filename="audio-15.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-15.tar.gz?download=1",
        checksum="f6af9b65d876ef96d199b9e5f0473cb9",
    ),
    "audio-16": download_utils.RemoteFileMetadata(
        filename="audio-16.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-16.tar.gz?download=1",
        checksum="e8337c61a90c30989d500f950fbe443a",
    ),
    "audio-17": download_utils.RemoteFileMetadata(
        filename="audio-17.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-17.tar.gz?download=1",
        checksum="3e47e85eb4564a30fe7f442ee97ec7b9",
    ),
    "audio-18": download_utils.RemoteFileMetadata(
        filename="audio-18.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-18.tar.gz?download=1",
        checksum="c6b2ef4d0d5b7269d465c469cdbbdc4b",
    ),
    "annotations.csv": download_utils.RemoteFileMetadata(
        filename="annotations.csv",
        url="https://zenodo.org/record/3966543/files/annotations.csv?download=1",
        checksum="70b507b15bb4cfcce4870925302f276b",
    ),
    "dcase-ust-taxonomy.yaml": download_utils.RemoteFileMetadata(
        filename="dcase-ust-taxonomy.yaml",
        url="https://zenodo.org/record/3966543/files/dcase-ust-taxonomy.yaml?download=1",
        checksum="6c1cca1c4c383a6ebb0cb71cb74fe3a9",
    )
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Clip(core.Clip):
    """SONYC-UST Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        events (sonyc.annotation.Events): sound events with start time, end time, label and confidence.
        audio_path (str): path to the audio file
        split (str): subset the clip belongs to (for experiments): train, validate, or test.
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

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def split(self):
        return self._clip_metadata.get("split")

    @property
    def sensor_id(self):
        return self._clip_metadata.get("sensor_id")

    @property
    def audio_filename(self):
        return self._clip_metadata.get("audio_filename")

    @property
    def annotator_id(self):
        return self._clip_metadata.get("annotator_id")

    @property
    def year(self):
        return self._clip_metadata.get("year")

    @property
    def week(self):
        return self._clip_metadata.get("week")

    @property
    def day(self):
        return self._clip_metadata.get("day")

    @property
    def hour(self):
        return self._clip_metadata.get("hour")

    @property
    def borough(self):
        return self._clip_metadata.get("borough")

    @property
    def block(self):
        return self._clip_metadata.get("block")

    @property
    def latitude(self):
        return self._clip_metadata.get("latitude")

    @property
    def longitude(self):
        return self._clip_metadata.get("longitude")

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        return load_events(self.txt_path)


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a SONYC-UST audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate of 44100 without resampling.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@io.coerce_to_string_io
def load_events(fhandle: TextIO) -> annotations.Events:
    """Load an SONYC-UST sound events annotation file
    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file
    Raises:
        IOError: if txt_path doesn't exist
    Returns:
        Events: sound events annotation data
    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        times.append([float(line[0]), float(line[1])])
        labels.append(line[2])
        confidence.append(1.0)

    events_data = annotations.Events(
        np.array(times), "seconds", labels, "open", np.array(confidence)
    )
    return events_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The SONYC-UST dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="sonyc_ust",
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

        metadata_path = os.path.join(self.data_home, "annotations.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        with open(metadata_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            raw_data = []
            for line in reader:
                if line[0] != "split":
                    raw_data.append(line)

        splits = ["train", "validate", "test"]
        expected_sizes = [13538, 4308, 664]
        metadata_index = {}
            

        for split, es in zip(splits, expected_sizes):

            annotation_folder = os.path.join(self.data_home, "annotations", split)

            for tf in txtfiles:
                clip_id = os.path.basename(tf).replace(".txt", "")
                metadata_index[clip_id] = {"split": split}

        return metadata_index
