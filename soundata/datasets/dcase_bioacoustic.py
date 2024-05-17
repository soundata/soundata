"""DCASE-BIOACOUSTIC Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **DCASE-BIOACOUSTIC**

    *Development set:*

    The development set for task 5 of DCASE 2022 "Few-shot Bioacoustic Event Detection" consists of 192 audio files acquired from different bioacoustic sources. The dataset is split into training and validation sets. 

    Multi-class annotations are provided for the training set with positive (POS), negative (NEG) and unkwown (UNK) values for each class. UNK indicates uncertainty about a class. 

    Single-class (class of interest) annotations are provided for the validation set, with events marked as positive (POS) or unkwown (UNK) provided for the class of interest. 

    This version (3) fixes issues with annotations from HB set. Development_Set_Annotations.zip has the same structure but contains only the .csv files.

    
    *Annotation structure*

    Each line of the annotation csv represents an event in the audio file. The column descriptions are as follows:

    TRAINING SET: Audiofilename, Starttime, Endtime, CLASS_1, CLASS_2, ...CLASS_N

    VALIDATION SET: Audiofilename, Starttime, Endtime, Q

    

    *Classes*

    DCASE2022_task5_training_set_classes.csv and DCASE2022_task5_validation_set_classes.csv provide a table with class code correspondence to class name for all classes in the Development set.

    DCASE2022_task5_training_set_classes.csv: dataset, class_code, class_name

    DCASE2022_task5_validation_set_classes.csv: dataset, recording, class_code, class_name

    

    *Evaluation set*

    The evaluation set for task 5 of DCASE 2022 "Few-shot Bioacoustic Event Detection" consists of 46 audio files acquired from different bioacoustic sources. 

    The first 5 annotations are provided for each file, with events marked as positive (POS) for the class of interest. 

    This dataset is to be used for evaluation purposes during the task and the rest of the annotations will be released after the end of the DCASE 2022 challenge (July 1st).    

    Evaluation_Set_5shots.zip has the same structure but contains only the .wav files.

    Evaluation_Set_5shots_annotations_only.zip has the same structure but contains only the .csv files

    The subfolders denote different recording sources and there may or may not be overlap between classes of interest from different wav files.

    Annotation structure

    Each line of the annotation csv represents an event in the audio file. The column descriptions are as follows:
    [ Audiofilename, Starttime, Endtime, Q ]


    *Open Access:*

    This dataset is available under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
    

    *Contact info:*

    Please send any feedback or questions to:

    Ines Nolasco -  i.dealmeidanolasco@qmul.ac.uk
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import jams
import glob
import json

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
@dataset{nolasco_ines_2022_6482837,
  author       = {Nolasco, Ines and
                  Singh, Shubhr and
                  Strandburg-Peshkin, Ariana and
                  Gill, Lisa and
                  Pamula, Hanna and
                  Morford, Joe and
                  Emmerson, Michael and
                  Jensen, Frants and
                  Whitehead, Helen and
                  Kiskin, Ivan and
                  Vidaña-Vila, Ester and
                  Lostanlen, Vincent and
                  Morfi, Veronica and
                  Stowell, Dan},
  title        = {{DCASE 2022 Task 5: Few-shot Bioacoustic Event 
                   Detection Development Set}},
  month        = mar,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6482837},
  url          = {https://doi.org/10.5281/zenodo.6482837}
}
"""

INDEXES = {
    "default": "3.0",
    "test": "sample",
    "3.0": core.Index(
        filename="dcase_bioacoustic_index_3.0.json",
        url="https://zenodo.org/records/11176645/files/dcase_bioacoustic_index_3.0.json?download=1",
        checksum="e4c4eca3f1b9224e178f283bd2835f8f",
    ),
    "sample": core.Index(filename="dcase_bioacoustic_index_3.0_sample.json"),
}

REMOTES = {
    "dev": download_utils.RemoteFileMetadata(
        filename="Development_Set.zip",
        url="https://zenodo.org/record/6482837/files/Development_Set.zip?download=1",
        checksum="cf4d3540c6c78ac2b3df2026c4f1f7ea",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "train-classes": download_utils.RemoteFileMetadata(
        filename="DCASE2022_task5_Training_set_classes.csv",
        url="https://zenodo.org/record/6482837/files/DCASE2022_task5_Training_set_classes.csv?download=1",
        checksum="abce1818ba10436971bad0b6a3464aa6",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "validation-classes": download_utils.RemoteFileMetadata(
        filename="DCASE2022_task5_Validation_set_classes.csv",
        url="https://zenodo.org/record/6482837/files/DCASE2022_task5_Validation_set_classes.csv?download=1",
        checksum="0c05ff0c9e1662ff8958c4c812abffdb",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "eval": download_utils.RemoteFileMetadata(
        filename="Evaluation_set_5shots.zip",
        url="https://zenodo.org/record/6517414/files/Evaluation_set_5shots.zip?download=1",
        checksum="5212c0e133874bba1ee25c81ced0de99",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Clip(core.Clip):
    """DCASE bioacoustic Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        csv_path (str): path to the csv file
        clip_id (str): clip id
        split (str): subset the clip belongs to (for experiments): train, validate, or test

    Cached properties:
        events_classes (list): list of classes annotated for the file
        events (soundata.annotations.Events): sound events with start time, end time, labels (list for all classes) and confidence
        POSevents (soundata.annotations.Events): sound events for the positive class with start time, end time, label and confidence

    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.csv_path = self.get_path("csv")

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
        """The data splits (e.g. train)

        Returns
            * str - split

        """
        return self._clip_metadata.get("split")

    @property
    def subdataset(self):
        """The (sub)dataset

        Returns
            * str - subdataset

        """
        return self._clip_metadata.get("subdataset")

    @core.cached_property
    def events_classes(self) -> Optional[list]:
        """The audio events

        Returns
            * list - list of the annotated events

        """
        return load_events_classes(self.csv_path)

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        """The audio events

        Returns
            * annotations.Events - audio event object

        """
        return load_events(self.csv_path)

    @core.cached_property
    def POSevents(self) -> Optional[annotations.Events]:
        """The audio events for POS (positive) class

        Returns
            * annotations.Events - audio event object

        """
        return load_POSevents(self.csv_path)

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            events=self.events,
            metadata={
                "split": self._clip_metadata.get("split"),
                "subdataset": self._clip_metadata.get("subdataset"),
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a DCASE bioacoustic audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate without resampling.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@io.coerce_to_string_io
def load_events(fhandle: TextIO) -> annotations.Events:
    """Load an DCASE bioacoustic sound events annotation file

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        Events: sound events annotation data

    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    for line in reader:
        times.append([float(line[1]), float(line[2])])
        classes = [class_ids[i] for i, l in enumerate(line[3:])]
        labels.append(",".join(classes))
        confidence.append(1.0)
    events_data = annotations.Events(
        intervals=np.array(times),
        intervals_unit="seconds",
        labels=labels,
        labels_unit="open",
        confidence=np.array(confidence),
    )
    return events_data


@io.coerce_to_string_io
def load_POSevents(fhandle: TextIO) -> annotations.Events:
    """Load an DCASE bioacoustic sound events annotation file, just for POS labels

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        Events: sound events annotation data

    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    for line in reader:
        times.append([float(line[1]), float(line[2])])
        classes = [class_ids[i] for i, l in enumerate(line[3:]) if l == "POS"]
        labels.append(",".join(classes))
        confidence.append(1.0)
    events_data = annotations.Events(
        intervals=np.array(times),
        intervals_unit="seconds",
        labels=labels,
        labels_unit="open",
        confidence=np.array(confidence),
    )
    return events_data


@io.coerce_to_string_io
def load_events_classes(fhandle: TextIO) -> list:
    """Load an DCASE bioacoustic sound events annotation file

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file
        positive (bool): False get all labels, True get just POS labels

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        class_ids: list of events classes

    """
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    return class_ids


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The DCASE bioacoustic dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="dcase_bioacoustic",
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
        metadata_index = {
            clip_id: {
                "subdataset": os.path.normpath(v["csv"][0])
                .split(clip_id)[0]
                .split(os.path.sep)[-2],
                "split": (
                    "train"
                    if "Training" in os.path.normpath(v["csv"][0]).split(clip_id)[0]
                    else (
                        "validation"
                        if "Validation"
                        in os.path.normpath(v["csv"][0]).split(clip_id)[0]
                        else "evaluation"
                    )
                ),
            }
            for clip_id, v in self._index["clips"].items()
        }

        metadata_paths = {
            "train": os.path.join(
                self.data_home, "DCASE2022_task5_Training_set_classes.csv"
            ),
            "validation": os.path.join(
                self.data_home, "DCASE2022_task5_Validation_set_classes.csv"
            ),
        }

        metadata_index["class_codes"] = {}
        metadata_index["subdatasets"] = {}

        for split, metadata_path in metadata_paths.items():
            metadata_path = os.path.normpath(metadata_path)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError("Metadata not found. Did you run .download()?")

            with open(metadata_path, "r") as fhandle:
                reader = csv.reader(fhandle, delimiter=",")

                headers = next(reader)
                class_code_id = headers.index("class_code")
                class_name_id = headers.index("class_name")
                dataset_id = headers.index("dataset")

                for line in reader:
                    metadata_index["class_codes"][line[class_code_id]] = {
                        "subdataset": line[dataset_id],
                        "class_name": line[class_name_id],
                        "split": split,
                    }
                    if line[dataset_id] not in metadata_index["subdatasets"]:
                        metadata_index["subdatasets"][line[dataset_id]] = [
                            line[class_code_id]
                        ]
                    else:
                        metadata_index["subdatasets"][line[dataset_id]].append(
                            line[class_code_id]
                        )

        return metadata_index
