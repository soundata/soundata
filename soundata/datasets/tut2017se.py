"""TUT Sound events 2017 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **TUT Sound events 2017, Development and Evaluation datasets**

    `Audio Research Group,
    Tampere University of Technology <http://arg.cs.tut.fi/>`__

    *Authors*

    * `Toni Heittola <http://www.cs.tut.fi/~heittolt/>`__
    * `Annamaria Mesaros <http://www.cs.tut.fi/~mesaros/>`__
    * `Tuomas Virtanen <http://www.cs.tut.fi/~tuomasv/>`__

    *Recording and annotation*

    * Eemi Fagerlund
    * Aku Hiltunen

    *Links*

    * `Development dataset <https://zenodo.org/record/814831>`__
    * `Evaluation dataset <https://zenodo.org/record/1040179>`__

    *Dataset*

    TUT Sound Events 2017 dataset consists of two subsets: development dataset
    and evaluation dataset. Partitioning of data into these subsets was done
    based on the amount of examples available for each sound event class, while
    also taking into account recording location. Because the event instances
    belonging to different classes are distributed unevenly within the
    recordings, the partitioning of individual classes can be controlled only
    to a certain extent, but so that the majority of events are in the
    development set.

    A detailed description of the data recording and annotation procedure is available in:

    .. code-block:: latex

        Annamaria Mesaros, Toni Heittola, and Tuomas Virtanen.
        "TUT database for acoustic scene classification and sound event
        detection", In 24th European Signal Processing Conference 2016,
        Budapest, Hungary, 2016.

    TUT Sound events 2017, development and evaluation datasets consist of 24
    and 8 audio recordings from a single acoustic scene respectively:

    * Development: Street (outdoor), totaling 1:32:08
    * Evaluation: Street (outdoor), totaling 29:09

    The dataset was collected in Finland by Tampere University of Technology
    between 06/2015 - 01/2016. The data collection has received funding from
    the European Research Council under the `ERC <https://erc.europa.eu/1>`_
    Grant Agreement 637422 EVERYSOUND.


    *Preparation of the dataset*

    The recordings were captured each in a different location (different
    streets). The equipment used for recording consists of a binaural
    `Soundman OKM II Klassik/studio A3 <http://www.soundman.de/en/products/>`_
    electret in-ear microphone and a `Roland Edirol R-09
    <http://www.rolandus.com/products/r-09/>`_ wave recorder using 44.1 kHz
    sampling rate and 24 bit resolution.

    For audio material recorded in private places, written consent was
    obtained from all people involved. Material recorded in public places
    (residential area) does not require such consent.

    Individual sound events in each recording were annotated by a research
    assistant using freely chosen labels for sounds. The annotator was trained
    first on few example recordings. He was instructed to annotate all audible
    sound events, and choose event labels freely. This resulted in a large set
    of raw labels. Mapping of the raw labels was performed, merging sounds into
    classes described by their source before selecting target classes. Target
    sound event classes for the dataset were selected based on the frequency
    of the obtained labels, resulting in selection of most common sounds for
    the street acoustic scene, in sufficient numbers for learning acoustic
    models. Mapping of the raw labels was performed, merging sounds into
    classes described by their source, for example "car passing by",
    "car engine running", "car idling", etc into "car", sounds produced by
    buses and trucks into "large vehicle", "children yelling" and
    "children talking" into "children", etc.

    Due to the high level of subjectivity inherent to the annotation process,
    a verification of the reference annotation was done using these mapped
    classes. Three persons (other than the annotator) listened to each audio
    segment annotated as belonging to one of these classes, marking agreement
    about the presence of the indicated sound within the segment.
    Agreement/disagreement did not take into account the sound event onset and
    offset, only the presence of the sound event within the annotated segment.
    Event instances that were confirmed by at least one person were kept,
    resulting in elimination of about 10% of the original event instances in
    the development set.

    The original metadata file is available in the directory `non_verified`.

    The ground truth is provided as a list of the sound events present in the
    recording, with annotated onset and offset for each sound instance.
    Annotations with only targeted sound events classes are in the directory
    `meta`.

    *Event statistics*

    The sound event instance counts for the dataset are shown below.

    *Development set*

    +------------------+------------+----------------+--------------------+
    |                  |    Development dataset      | Evaluation dataset |
    +------------------+------------+----------------+--------------------+
    | Event label      |Verified set|Non-verified set| Verified set       |
    +==================+============+================+====================+
    | brakes squeaking | 52         | 59             | 23                 |
    +------------------+------------+----------------+--------------------+
    | car              | 304        | 304            | 106                |
    +------------------+------------+----------------+--------------------+
    | children         | 44         | 58             | 15                 |
    +------------------+------------+----------------+--------------------+
    | large vehicle    | 61         | 61             | 24                 |
    +------------------+------------+----------------+--------------------+
    | people speaking  | 89         | 117            | 37                 |
    +------------------+------------+----------------+--------------------+
    | people walking   | 109        | 130            | 42                 |
    +------------------+------------+----------------+--------------------+
    | **Total**        | **659**    | **729**        | **247**            |
    +------------------+------------+----------------+--------------------+


    *Usage*

    Partitioning of data into **development dataset** and **evaluation
    dataset** was done based on the amount of examples available for each event
    class, while also taking into account recording location. Ideally the
    subsets should have the same amount of data for each class, or at least the
    same relative amount, such as a 70-30% split. Because the event instances
    belonging to different classes are distributed unevenly within the
    recordings, the partitioning of individual classes can be controlled only
    to a certain extent.

    The split condition was relaxed so that 65-75% of instances of each class
    were selected into the development set.


    *Cross-validation setup*

    The setup is provided with the dataset in the directory `evaluation_setup`.

    *License*

    See file `EULA.pdf
    <https://github.com/TUT-ARG/DCASE2017-baseline-system/blob/master/EULA.pdf>`_
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv

# import jams

from soundata import download_utils, jams_utils, core, annotations, io


BIBTEX = """
@inproceedings{Mesaros:DCASE:17,
    Address = {Munich, Germany},
    Author = {Mesaros, A. and Heittola, T. and Diment, A. and Elizalde, B. and
              Shah, A. and Vincent, E. and Raj, B. and Virtanen, T.},
    Booktitle = {Proceedings of the Detection and Classification of Acoustic
                 Scenes and Events 2017 Workshop (DCASE2017)},
    Month = {November},
    Pages = {85--92},
    Title = {{DCASE} 2017 Challenge Setup: Tasks, Datasets and Baseline
             System},
    Year = {2017}}
"""

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "2.0": core.Index(
        filename="tut2017se_index_2.0.json",
        url="https://zenodo.org/records/11176916/files/tut2017se_index_2.0.json?download=1",
        checksum="26fea2fd4082f48d7e8ef8c85df88ad1",
    ),
    "sample": core.Index(filename="tut2017se_index_2.0_sample.json"),
}

REMOTES = {
    "development.audio.1": download_utils.RemoteFileMetadata(
        filename="TUT-sound-events-2017-development.audio.1.zip",
        url=(
            "https://zenodo.org/record/814831/files/TUT-sound-events-2017-"
            "development.audio.1.zip?download=1"
        ),
        checksum="6f1cd31592b8240a14be3ee513db6a23",
    ),
    "development.audio.2": download_utils.RemoteFileMetadata(
        filename="TUT-sound-events-2017-development.audio.2.zip",
        url=(
            "https://zenodo.org/record/814831/files/TUT-sound-events-2017-"
            "development.audio.2.zip?download=1"
        ),
        checksum="adcff03341b84dc8d35f035b93c1efa0",
    ),
    "development.doc": download_utils.RemoteFileMetadata(
        filename="TUT-sound-events-2017-development.doc.zip",
        url=(
            "https://zenodo.org/record/814831/files/TUT-sound-events-2017-"
            "development.doc.zip?download=1"
        ),
        checksum="aa6024e70f5bff3fe15d962b01753e23",
    ),
    "development.meta": download_utils.RemoteFileMetadata(
        filename="TUT-sound-events-2017-development.meta.zip",
        url=(
            "https://zenodo.org/record/814831/files/TUT-sound-events-2017-"
            "development.meta.zip?download=1"
        ),
        checksum="50e870b3a89ed3452e2a35b508840929",
    ),
    "evaluation.audio": download_utils.RemoteFileMetadata(
        filename="TUT-sound-events-2017-evaluation.audio.zip",
        url=(
            "https://zenodo.org/record/1040179/files/TUT-sound-events-2017-"
            "evaluation.audio.zip?download=1"
        ),
        checksum="1d3aa81896be0f142130ca9ca7a2b871",
    ),
    "evaluation.doc": download_utils.RemoteFileMetadata(
        filename="TUT-sound-events-2017-evaluation.doc.zip",
        url=(
            "https://zenodo.org/record/1040179/files/TUT-sound-events-2017-"
            "evaluation.doc.zip?download=1"
        ),
        checksum="8bbf41671949edee15d6cdc3f9e726c9",
    ),
    "evaluation.meta": download_utils.RemoteFileMetadata(
        filename="TUT-sound-events-2017-evaluation.meta.zip",
        url=(
            "https://zenodo.org/record/1040179/files/TUT-sound-events-2017-"
            "evaluation.meta.zip?download=1"
        ),
        checksum="a951598abaea87296ca409e30fb0b379",
    ),
}

LICENSE_INFO = "TUT License <https://github.com/TUT-ARG/DCASE2017-baseline-system/blob/master/EULA.pdf>"


class Clip(core.Clip):
    """TUT Sound events 2017 Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        annotations_path (str): path to the annotations file
        clip_id (str): clip id
        events (soundata.annotations.Events): sound events with start time,
            end time, label and confidence
        non_verified_annotations_path (str): path to the non-verified
            annotations file
        non_verified_events (soundata.annotations.Events): non-verified sound
            events with start time, end time, label and confidence
        split (str): subset the clip belongs to (for experiments):
            development (fold1, fold2, fold3, fold4) or evaluation
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.annotations_path = self.get_path("annotations")
        self.non_verified_annotations_path = self.get_path("non_verified_annotations")

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
        """The clip's split.

        Returns:
            * str - subset the clip belongs to (for experiments): development (fold1, fold2, fold3, fold4) or evaluation

        """
        return self._clip_metadata.get("split")

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        """The clip's events.

        Returns:
            * annotations.Events - sound events with start time, end time, label and confidence

        """
        return load_events(self.annotations_path)

    @core.cached_property
    def non_verified_events(self) -> Optional[annotations.Events]:
        """The clip's non verified events path

        Returns:
            * str - path to the non-verified annotations file

        """
        return load_events(self.non_verified_annotations_path)

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, events=self.events, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a TUT Sound events 2017 audio file

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate of 44100 without resampling.

    Returns:
        * np.ndarray - the stereo audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=False)
    return audio, sr


@io.coerce_to_string_io
def load_events(fhandle: TextIO) -> annotations.Events:
    """Load an TUT Sound events 2017 annotation file

    Args:
        fhandle (str or file-like): File-like object or path to the sound
        events annotation file

    Returns:
        Events: sound events annotation data
    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        offset = (
            0 if len(line) == 3 else 2
        )  # ann files in dev and eval have different format
        times.append([float(line[offset]), float(line[offset + 1])])
        labels.append(line[offset + 2])
        confidence.append(1.0)

    events_data = annotations.Events(
        np.array(times), "seconds", labels, "open", np.array(confidence)
    )
    return events_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The TUT Sound events 2017 dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="tut2017se",
            clip_class=Clip,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_events)
    def load_events(self, *args, **kwargs):
        return load_events(*args, **kwargs)

    @core.cached_property
    def _metadata(self):
        splits = [
            "development.fold1",
            "development.fold2",
            "development.fold3",
            "development.fold4",
            "evaluation",
        ]

        metadata_index = {}

        for split in splits:
            if split.split(".")[0] == "development":
                evaluation_setup_path = (
                    "TUT-sound-events-2017-development/evaluation_setup"
                )
                fold = split.split(".")[1]
                evaluation_setup_file = os.path.join(
                    self.data_home,
                    evaluation_setup_path,
                    "street_{}_test.txt".format(fold),
                )
            else:
                evaluation_setup_path = (
                    "TUT-sound-events-2017-evaluation/evaluation_setup"
                )
                evaluation_setup_file = os.path.join(
                    self.data_home, evaluation_setup_path, "street_test.txt"
                )

            with open(evaluation_setup_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter="\t")
                for row in csv_reader:
                    file_name = os.path.basename(row[0])
                    clip_id = os.path.basename(file_name).replace(".wav", "")
                    metadata_index[clip_id] = {"split": split}

        return metadata_index
