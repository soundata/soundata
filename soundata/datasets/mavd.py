"""MAVD-traffic Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    MAVD-traffic
    ============

    MAVD-traffic (c) by Pablo Zinemanas, Pablo Cancela, and Martín Rocamora.
    MAVD-traffic is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).
    You should have received a copy of the license along with this work. If not, see <http://creativecommons.org/licenses/by/4.0/>.

    Created By
    ----------

    Pablo Zinemanas, Pablo Cancela, and Martín Rocamora.
    Facultad de Ingeniería, Universidad de la República, Montevideo, Uruguay.

    Version 0.1.0
    -------------
    - NOTE: video files in this release are provided in low resolution (176x144) and are not downloaded by default with this loader.

    Description
    -----------

    MAVD-traffic is a dataset for sound event detection in urban environments. 

    A detailed description of the dataset is provided in the following article:

    .. code-block:: latex
        P. Zinemanas, P. Cancela, and M. Rocamora. "MAVD: A dataset for sound event detection in urban environments", 
        In Detection and Classification of Acoustic Scenes and Events (DCASE), New York, NY, USA, Oct. 2019.

    A summary is provided here:

    This is a dataset for sound event detection in urban environments that focuses on traffic noise. 

    * The recordings were produced in Montevideo, the capital city of Uruguay.
    * Audio and video files of about 15-minutes long were recorded at different times of the day in different locations.
    * Four different locations are included in this release, corresponding to different levels of traffic activity and social use characteristics.
    * The sound was captured with a SONY PCM-D50 recorder at a sampling rate of 48kHz and a resolution of 24bits.
    * The video was recorded with a GoPro Hero 3 camera at a rate of 30 frames per second and a resolution of 1920x1080 pixels.
    * The sound events in the recordings were manually annotated using the ELAN software. 
    * The sound event annotations follow an ontology for traffic sounds that is the combination of a set of two taxonomies:
        * vehicle types (e.g. car, bus) and vehicle components (e.g. engine, brakes),
        * and a set of actions related to them (e.g. idling, accelerating).
    * The dataset includes 47 recordings of about 5-minutes length each, for a total of almost 4 hours.
    * The dataset includes more than 34,000 annotated sound events.
    * MAVD-traffic comes pre-sorted into three sets: train, validate and test
        * There are 24 recordings in the training set (117 minutes), including 1,737 annotated sound events. 
        * There are 7 recordings in the validation set (33 minutes), including 556 annotated sound events. 
        * There are 16 recordings in the test set (83 minutes), including 1,134 annotated sound events. 


    Audio Files Included
    --------------------

    * 47 recordings of about 5-minutes length, in single channel (mono), 44100Hz, 24-bit, FLAC format.
    * The files are split into a training set (24), validation set (7) and test set (16).


    Annotation Files Included
    -------------------------
    The annotations list the sound events that occur in every recording and are provided as a text file for each recording (.csv file). 
    The annotations are "strong", meaning for every sound event the annotations include the start time, end time, and label of the sound event. 

    The sound event annotations follow an ontology for traffic sounds that is the combination of a set of two taxonomies:
        * vehicle types (e.g. car, bus) and vehicle components (e.g. engine, brakes),
        * and a set of actions related to them (e.g. idling, accelerating).
     Since the taxonomies follow a hierarchy it can be used with different levels of detail. 

    * Further details about the ontology for traffic noise description can be found in Section 2 of the aforementioned paper.
    * Further details about the annotation process using the ELAN software can be found in Section 3.2 of the aforementioned paper.


    Please Acknowledge MAVD-traffic in Academic Research
    ----------------------------------------------------

    When MAVD-traffic is used for academic research, we would highly appreciate it if scientific publications of works 
    partly based on the MAVD-traffic dataset cite the following publication:

    .. code-block:: latex
        P. Zinemanas, P. Cancela, and M. Rocamora. "MAVD: A dataset for sound event detection in urban environments", 
        In Detection and Classification of Acoustic Scenes and Events (DCASE), New York, NY, USA, Oct. 2019.


    Conditions of Use
    -----------------

    Dataset created by P. Zinemanas, P. Cancela, and M. Rocamora.
    
    The MAVD-traffic dataset is offered free of charge under the terms of the Creative Commons
    Attribution 4.0 International License (CC BY 4.0): http://creativecommons.org/licenses/by/4.0/
    
    The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including 
    without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or 
    completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, 
    Universidad de la República is not liable for, and expressly excludes, all liability for loss or damage however and 
    whenever caused to anyone by any use of the MAVD-traffic dataset or any part of it.


    Feedback
    --------

    Please help us improve MAVD-traffic by sending your feedback to: rocamora@fing.edu.uy
    In case of a problem report please include as many details as possible.

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
@inproceedings{ZinemanasDCASE2019,
	Address = {New York, NY, USA},
	Author = {Zinemanas, P. and Cancela, P. and Rocamora, M.},
	Booktitle = {Detection and Classification of Acoustic Scenes and Events (DCASE)},
	Month = {Oct.},
	Pages = {263--267},
        Title = {MAVD: a dataset for sound event detection in urban environments},
	Year = {2019}}
"""

REMOTES = {
    "annotations_test": download_utils.RemoteFileMetadata(
        filename="annotations_test.zip",
        url="https://zenodo.org/record/3338727/files/annotations_test.zip?download=1",
        checksum="94e609cad8702f5fb6afe52815547b67"
    ),
    "annotations_train": download_utils.RemoteFileMetadata(
        filename="annotations_train.zip",
        url="https://zenodo.org/record/3338727/files/annotations_train.zip?download=1",
        checksum="ea67a330ab695873a12a07ed494eae42"
    ),
    "annotations_validate": download_utils.RemoteFileMetadata(
        filename="annotations_validate.zip",
        url="https://zenodo.org/record/3338727/files/annotations_validate.zip?download=1",
        checksum="30edae5c398dffe52a2ee1373e89fd96"
    ),  
    "audio_test": download_utils.RemoteFileMetadata(
        filename="audio_test.zip",
        url="https://zenodo.org/record/3338727/files/audio_test.zip?download=1",
        checksum="824b035fa76dc83919e312105db4214f"
    ),
    "audio_train": download_utils.RemoteFileMetadata(
        filename="audio_train.zip",
        url="https://zenodo.org/record/3338727/files/audio_train.zip?download=1",
        checksum="c7a897623364f80c8ff3775d8dd601a2"
    ),
    "audio_validate": download_utils.RemoteFileMetadata(
        filename="audio_validate.zip",
        url="https://zenodo.org/record/3338727/files/audio_validate.zip?download=1",
        checksum="d220cef03f99b2d108d708421240f86d"
    )   
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Clip(core.Clip):
    """MAVD-traffic Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        events (soundata.annotation.Events): sound events with start time, end time, label and confidence.
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
        self.txt_path = self.get_path("txt")

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

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        return load_events(self.csv_path)

@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a MAVD-traffic audio file.

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
    """Load an MAVD-traffic sound events annotation file
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
        confidence.append(1.0) # There is no confidence annotation in MAVD-traffic v0.1.0

    events_data = annotations.Events(np.array(times), labels, np.array(confidence))
    return events_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The MAVD-traffic dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="mavd",
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

        splits = ["train", "validate", "test"]
        expected_sizes = [24, 7, 16]
        metadata_index = {}

        for split, es in zip(splits, expected_sizes):

            annotation_folder = os.path.join(self.data_home, "annotations_" + split)
            txtfiles = sorted(glob.glob(os.path.join(annotation_folder, "*.txt")))

            for tf in txtfiles:
                clip_id = os.path.basename(tf).replace(".txt", "")
                metadata_index[clip_id] = {"split": split}

        return metadata_index
