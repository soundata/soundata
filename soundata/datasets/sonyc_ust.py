"""SONYC-UST 2.3 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Created by
    ----------

    Mark Cartwright (1,2,3), Jason Cramer (1), Ana Elisa Mendez Mendez (1), Yu Wang (1), Ho-Hsiang Wu (1),
    Vincent Lostanlen (1,2,4), Magdalena Fuentes (1), Graham Dove (2), Charlie Mydlarz (1,2),
    Justin Salamon (5), Oded Nov (6), Juan Pablo Bello (1,2,3)

    (1) Music and Audio Research Lab, New York University
    (2) Center for Urban Science and Progress, New York University
    (3) Department of Computer Science and Engineering, New York University
    (4) Cornell Lab of Ornithology
    (5) Adobe Research
    (6) Department of Technology Management and Innovation, New York University

    Version 2.3, September 2020

    Publication
    -----------

    If using this data in an academic work, please reference the DOI and version, as well as cite the following paper,
    which presented the data collection procedure and the first version of the dataset:

    .. code-block:: latex
        Cartwright, M., Mendez, A.E.M., Cramer, J., Lostanlen, V., Dove, G., Wu, H., Salamon, J., Nov, O., Bello, J.P.
        SONYC Urban Sound Tagging (SONYC-UST): A Multilabel Dataset from an Urban Acoustic Sensor Network.
        In Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE) , 2019.

    Description
    -----------

    SONYC Urban Sound Tagging (SONYC-UST) is a dataset for the development and evaluation of machine listening systems
    for realistic urban noise monitoring. The audio was recorded from the SONYC acoustic sensor network.
    Volunteers on the  Zooniverse citizen science platform tagged the presence of 23 classes that were chosen in
    consultation with the New York City Department of Environmental Protection. These 23 fine-grained classes can be
    grouped into 8 coarse-grained classes. The recordings are split into three sets: training, validation, and test
    (the test set will be released in July 2020). The training and validation sets are disjoint with respect to the
    sensor from which each recording came, and the test set is displaced in time. For increased reliability, three
    volunteers annotated each recording. In addition, members of the SONYC team subsequently created a subset of
    verified, ground-truth tags using a two-stage annotation procedure in which two annotators independently tagged
    and then collectively resolved any disagreements. This subset of recordings with verified annotations intersects
    with both the training and validation set. All of the recordings in the test will have annotations verified by
    the SONYC team using the same procedure. In v2 version of this dataset, we have also included coarse spatiotemporal
    context information to aid in tag prediction when time and location is known. For more details on the motivation
    and creation of this dataset see the DCASE 2020 Urban Sound Tagging with Spatiotemporal Context Task website.
    NOTE: the test set will be released shortly after the final submission date for DCASE 2020 Challenge (July 2020).


    Audio data
    ----------

    The provided audio has been acquired using the SONYC acoustic sensor network for urban noise pollution monitoring.
    Over 60 different sensors have been deployed in New York City, and these sensors have collectively gathered the
    equivalent of over 50 years of audio data, of which we provide a small subset. The data was sampled by selecting
    the nearest neighbors on VGGish features of recordings known to have classes of interest. All recordings are 10
    seconds and were recorded with identical microphones at identical gain settings. To maintain privacy, we quantized
    the spatial information to the level of a city block, and we quantized the temporal information to the level of an
    hour. We also limited the occurrence of recordings with positive human voice annotations to one per hour per sensor.


    Label taxonomy
    --------------

    The label taxonomy is as follows:

    engine
    1: small-sounding-engine
    2: medium-sounding-engine
    3: large-sounding-engine
    X: engine-of-uncertain-size
    machinery-impact
    1: rock-drill
    2: jackhammer
    3: hoe-ram
    4: pile-driver
    X: other-unknown-impact-machinery
    non-machinery-impact
    1: non-machinery-impact
    powered-saw
    1: chainsaw
    2: small-medium-rotating-saw
    3: large-rotating-saw
    X: other-unknown-powered-saw
    alert-signal
    1: car-horn
    2: car-alarm
    3: siren
    4: reverse-beeper
    X: other-unknown-alert-signal
    music
    1: stationary-music
    2: mobile-music
    3: ice-cream-truck
    X: music-from-uncertain-source
    human-voice
    1: person-or-small-group-talking
    2: person-or-small-group-shouting
    3: large-crowd
    4: amplified-speech
    X: other-unknown-human-voice
    dog
    1: dog-barking-whining
    The classes preceded by an X code indicate when an annotator was able to identify the coarse class, but couldn’t
    identify the fine class because either they were uncertain which fine class it was or the fine class was not
    included in the taxonomy. dcase-ust-taxonomy.yaml contains this taxonomy in an easily machine-readable form.


    Conditions of use
    -----------------

    The SONYC-UST dataset is offered free of charge under the terms of the Creative Commons Attribution 4.0
    International (CC BY 4.0) license: https://creativecommons.org/licenses/by/4.0/

    The dataset and its contents are made available on an “as is” basis and without warranties of any kind,
    including without limitation satisfactory quality and conformity, merchantability, fitness for a particular
    purpose, accuracy or completeness, or absence of errors. Subject to any liability that may not be excluded or
    limited by law, New York University is not liable for, and expressly excludes all liability for, loss or damage
    however and whenever caused to anyone by any use of the SONYC-UST dataset or any part of it.

    Feedback
    --------

    Please help us improve SONYC-UST by sending your feedback to:

    Mark Cartwright: mcartwright@gmail.com
    In case of a problem, please include as many details as possible.
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv

from soundata import download_utils, jams_utils, core, annotations, io

BIBTEX = """
@dataset{mark_cartwright_2020_3693077,
  author       = {Mark Cartwright and
                  Jason Cramer and
                  Ana Elisa Mendez Mendez and
                  Yu Wang and
                  Ho-Hsiang Wu and
                  Vincent Lostanlen and
                  Magdalena Fuentes and
                  Graham Dove and
                  Charlie Mydlarz and
                  Justin Salamon and
                  Oded Nov and
                  Juan Pablo Bello},
  title        = {{SONYC Urban Sound Tagging (SONYC-UST): a 
                   multilabel dataset from an urban acoustic sensor
                   network}},
  month        = mar,
  year         = 2020,
  note         = {{This work is supported by National Science 
                   Foundation award 1544753.}},
  publisher    = {Zenodo},
  version      = {2.1.0},
  doi          = {10.5281/zenodo.3693077},
  url          = {https://doi.org/10.5281/zenodo.3693077}
}
"""
REMOTES = {
    "annotations": download_utils.RemoteFileMetadata(
        filename="annotations.csv",
        url="https://zenodo.org/record/3693077/files/annotations.csv?download=1",
        checksum="70b507b15bb4cfcce4870925302f276b",
        unpack_directories=["SONYC_UST"]
    ),
    "audio_0": download_utils.RemoteFileMetadata(
        filename="audio-0.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-0.tar.gz?download=1",
        checksum="bbb4dbae7d2e58e18d24878b9ee1eb51",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_1": download_utils.RemoteFileMetadata(
        filename="audio-1.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-1.tar.gz?download=1",
        checksum="7c369ff37ac6a1fdd8493fdffe62e0a1",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_2": download_utils.RemoteFileMetadata(
        filename="audio-2.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-2.tar.gz?download=1",
        checksum="412241c063d7f196953d3dd1c44aeb5e",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_3": download_utils.RemoteFileMetadata(
            filename="audio-3.tar.gz",
            url="https://zenodo.org/record/3966543/files/audio-3.tar.gz?download=1",
            checksum="d2a872392d95993a2c238d305b940812",
            unpack_directories=["SONYC_UST", "audio"]
        ),
    "audio_4": download_utils.RemoteFileMetadata(
            filename="audio-4.tar.gz",
            url="https://zenodo.org/record/3966543/files/audio-4.tar.gz?download=1",
            checksum="b15b0e0cb3f8584259ef0d24293a9be3",
            unpack_directories=["SONYC_UST", "audio"]
        ),
    "audio_5": download_utils.RemoteFileMetadata(
            filename="audio-5.tar.gz",
            url="https://zenodo.org/record/3966543/files/audio-5.tar.gz?download=1",
            checksum="de745f887067433757a2f2c8f99f99bb",
            unpack_directories=["SONYC_UST", "audio"]
        ),
    "audio_6": download_utils.RemoteFileMetadata(
        filename="audio-6.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-6.tar.gz?download=1",
        checksum="b0286c67468369d66336f6f5ddede31f",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_7": download_utils.RemoteFileMetadata(
        filename="audio-7.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-7.tar.gz?download=1",
        checksum="ff300183ab7a9d3f2f74c3b730ffeb52",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_8": download_utils.RemoteFileMetadata(
        filename="audio-8.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-8.tar.gz?download=1",
        checksum="7be76b821fa6dbe20f6b50ca440e1024",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_9": download_utils.RemoteFileMetadata(
        filename="audio-9.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-9.tar.gz?download=1",
        checksum="959f7edfbb26eadad9865069f38aa9dd",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_10": download_utils.RemoteFileMetadata(
        filename="audio-10.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-10.tar.gz?download=1",
        checksum="84e94de273666c537a3e6b709ee88d9b",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_11": download_utils.RemoteFileMetadata(
        filename="audio-11.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-11.tar.gz?download=1",
        checksum="44fe1f43121a7d1178aa0cdf7477793b",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_12": download_utils.RemoteFileMetadata(
        filename="audio-12.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-12.tar.gz?download=1",
        checksum="a7eb21011f460b5ac289f8e285e875ab",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_13": download_utils.RemoteFileMetadata(
        filename="audio-13.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-13.tar.gz?download=1",
        checksum="27945a8b1008eec787f97213164951e6",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_14": download_utils.RemoteFileMetadata(
        filename="audio-14.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-14.tar.gz?download=1",
        checksum="fb8c7d81c3bde3a0c86c1e55e6a85a24",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_15": download_utils.RemoteFileMetadata(
        filename="audio-15.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-15.tar.gz?download=1",
        checksum="f6af9b65d876ef96d199b9e5f0473cb9",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_16": download_utils.RemoteFileMetadata(
        filename="audio-16.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-16.tar.gz?download=1",
        checksum="e8337c61a90c30989d500f950fbe443a",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_17": download_utils.RemoteFileMetadata(
        filename="audio-17.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-17.tar.gz?download=1",
        checksum="3e47e85eb4564a30fe7f442ee97ec7b9",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "audio_18": download_utils.RemoteFileMetadata(
        filename="audio-18.tar.gz",
        url="https://zenodo.org/record/3966543/files/audio-18.tar.gz?download=1",
        checksum="c6b2ef4d0d5b7269d465c469cdbbdc4b",
        unpack_directories=["SONYC_UST", "audio"]
    ),
    "taxonomy": download_utils.RemoteFileMetadata(
        filename="dcase-ust-taxonomy.yaml",
        url="https://zenodo.org/record/3693077/files/dcase-ust-taxonomy.yaml?download=1",
        checksum="6c1cca1c4c383a6ebb0cb71cb74fe3a9",
        unpack_directories=["SONYC_UST"]
    )
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International (CC BY 4.0)"


class Clip(core.Clip):
    """sonyc_ust Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        tags (soundata.annotation.Tags): tag (label) of the clip + confidence.
        audio_path (str): path to the audio file
        file_name (str): The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav.
            Please see the Dataset Info in the soundata documentation for further details.
        freesound_id (str): ID of the freesound.org recording from which this clip was taken.
        freesound_start_time (float): start time in seconds of the clip in the original freesound recording.
        freesound_end_time (float): end time in seconds of the clip in the original freesound recording.
        salience (int): annotator estimate of class sailence in the clip: 1 = foreground, 2 = background.
        fold (int): fold number (1-10) to which this clip is allocated. Use these folds for cross validation.
        class_id (int): integer representation of the class label (0-9). See Dataset Info in the documentation for mapping.
        class_label (str): string class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music.
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
    def slice_file_name(self):
        return self._clip_metadata.get("slice_file_name")

    @property
    def freesound_id(self):
        return self._clip_metadata.get("freesound_id")

    @property
    def freesound_start_time(self):
        return self._clip_metadata.get("freesound_start_time")

    @property
    def freesound_end_time(self):
        return self._clip_metadata.get("freesound_end_time")

    @property
    def salience(self):
        return self._clip_metadata.get("salience")

    @property
    def fold(self):
        return self._clip_metadata.get("fold")

    @property
    def class_id(self):
        return self._clip_metadata.get("class_id")

    @property
    def class_label(self):
        return self._clip_metadata.get("class_label")

    @property
    def tags(self):
        return annotations.Tags(
            [self._clip_metadata.get("class_label")], np.array([1.0])
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
