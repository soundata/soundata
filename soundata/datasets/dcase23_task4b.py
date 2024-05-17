"""DCASE23 Task 4B Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **DCASE 2023 Task-4B**

    *Created By:* 

        | Annamaria Mesaros, Tuomas Heittola, and Tuomas Virtanen.
        | Tampere University of Technology. 
    
    Version 1.0

    *Description:*
        MAESTRO real development contains 49 real-life audio files from 5 different acoustic scenes, each of them from 3 to 5 minutes long. 
        The other 26 files are kept for evaluation purposes on the DCASE task 4 B. The distribution of files per scene is the following: 
        cafe restaurant 10 files, city center 10 files, residential_area 11 files, metro station 9 files and grocery store 9 files. 
        The total duration of the development dataset is 97 minutes and 4 seconds.

        The audio files contain sounds from the following classes:

        - announcement
        - birds singing
        - breakes squeaking
        - car
        - cash register
        - children voices
        - coffee machine
        - cutlery/dishes
        - door opens/closes
        - footsteps
        - furniture dragging

        The real life-recordings used in this study include a subset of the TUT Sound Events 2016 and a subset of TUT Sound Events 2017.

    *Please Acknowledge TUT Acoustic Scenes Strong Label Dataset in Academic Research:*
    If you use this dataset, please cite the following paper:

    .. code-block:: latex

        A. Mesaros, T. Heittola, and T. Virtanen, "TUT database for acoustic scene classification and sound event detection," in 2016 24th European Signal Processing Conference (EUSIPCO), 2016, pp. 1128-1132.

    *License:*
        License permits free academic usage. Any commercial use is strictly prohibited. For commercial use, contact dataset authors.
        Copyright (c) 2020 Tampere University and its licensors. All rights reserved.

        Permission is hereby granted, without written agreement and without license or royalty
        fees, to use and copy the MAESTRO Real - Multi Annotator Estimated Strong Labels (“Work”) described in this document
        and composed of audio and metadata. This grant is only for experimental and non-commercial
        purposes, provided that the copyright notice in its entirety appear in all copies of this Work,
        and the original source of this Work, (MAchine Listening Group at Tampere University),
        is acknowledged in any publication that reports research using this Work.
        Any commercial use of the Work or any part thereof is strictly prohibited.
        Commercial use include, but is not limited to:

        - selling or reproducing the Work
        - selling or distributing the results or content achieved by use of the Work
        - providing services by using the Work.

    *Feedback:*
        For questions or feedback, please contact irene.martinmorato@tuni.fi.
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple
import glob
import librosa
import csv
import numpy as np
from soundata import download_utils, jams_utils, core, annotations, io


BIBTEX = """
@article{Martinmorato2023,
    author = "Martín-Morató, Irene and Mesaros, Annamaria",
    journal = "IEEE/ACM Transactions on Audio, Speech, and Language Processing",
    title = "Strong Labeling of Sound Events Using Crowdsourced Weak Labels and Annotator Competence Estimation",
    year = "2023",
    volume = "31",
    number = "",
    pages = "902-914",
    doi = "10.1109/TASLP.2022.3233468",
    abstract = "Crowdsourcing is a popular tool for collecting large amounts of annotated data, but the specific format of the strong labels necessary for sound event detection is not easily obtainable through crowdsourcing. In this work, we propose a novel annotation workflow that leverages the efficiency of crowdsourcing weak labels, and uses a high number of annotators to produce reliable and objective strong labels. The weak labels are collected in a highly redundant setup, to allow reconstruction of the temporal information. To obtain reliable labels, the annotators' competence is estimated using MACE (Multi-Annotator Competence Estimation) and incorporated into the strong labels estimation through weighing of individual opinions. We show that the proposed method produces consistently reliable strong annotations not only for synthetic audio mixtures, but also for audio recordings of real everyday environments. While only a maximum 80\% coincidence with the complete and correct reference annotations was obtained for synthetic data, these results are explained by an extended study of how polyphony and SNR levels affect the identification rate of the sound events by the annotators. On real data, even though the estimated annotators' competence is significantly lower and the coincidence with reference labels is under 69\%, the proposed majority opinion approach produces reliable aggregated strong labels in comparison with the more difficult task of crowdsourcing directly strong labels."
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="dcase23_task4b_index_1.0.json",
        url="https://zenodo.org/records/11176783/files/dcase23_task4b_index_1.0.json?download=1",
        checksum="fc6bf79b17b2ce713e5389668174966a",
    ),
    "sample": core.Index(filename="dcase23_task4b_index_1.0_sample.json"),
}

REMOTES = {
    "development_audio": download_utils.RemoteFileMetadata(
        filename="development_audio.zip",
        url="https://zenodo.org/records/7244360/files/development_audio.zip?download=1",
        checksum="3de7cb4f92a115a6f5cc077a41ca07b3",
    ),
    "development_annotation": download_utils.RemoteFileMetadata(
        filename="development_annotation.zip",
        url="https://zenodo.org/records/7244360/files/development_annotation.zip?download=1",
        checksum="e6a3f84f8020725d559b38ccb494ef3d",
    ),
    "evaluation_audio": download_utils.RemoteFileMetadata(
        filename="Evaluation_audio.zip",
        url="https://zenodo.org/records/7870026/files/Evaluation_audio.zip?download=1",
        checksum="f18a036f5d8d3b5d4e810cae75da308e",
    ),
}

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Clip(core.Clip):
    """DCASE23_Task4B Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        annotations_path (str): path to the annotations file
        clip_id (str): clip id
        events (soundata.annotations.Events): sound events with start time,
            end time, label and confidence
        split (str): subset the clip belongs to:
            development or evaluation
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.annotations_path = self.get_path("annotations")

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
            * str - subset the clip belongs to: development or evaluation

        """
        return self._clip_metadata.get("split")

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        """The clip's events.

        Returns:
            * annotations.Events - sound events with start time, end time, label and confidence

        """
        return load_events(self.annotations_path)

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
    """Load a DCASE23_Task4B audio file.

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
    """Load a DCASE23_Task4B annotation file

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file

    Returns:
        Events: sound events annotation data
    """

    times = []
    labels = []
    confidence = []
    default_headers = ["start", "end", "label", "confidence"]
    reader = csv.DictReader(fhandle, delimiter="\t", fieldnames=default_headers)
    for line in reader:
        times.append([float(line["start"]), float(line["end"])])
        labels.append(line["label"])
        confidence.append(min(float(line["confidence"]), 1.0))

    events_data = annotations.Events(
        np.array(times), "seconds", labels, "open", np.array(confidence)
    )
    return events_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The DCASE23_Task4B dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="dcase23_task4b",
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
        # Define the base paths for annotations and audio files
        annotations_base_path = os.path.join(self.data_home, "development_annotation")

        metadata_index = {}

        environments = [
            "cafe_restaurant",
            "city_center",
            "grocery_store",
            "metro_station",
            "residential_area",
        ]
        for env in environments:
            env_annotations_path = os.path.join(
                annotations_base_path, "soft_labels_" + env
            )
            annotation_files = glob.glob(os.path.join(env_annotations_path, "*.txt"))
            for annotation_file in annotation_files:
                file_id = os.path.basename(annotation_file).replace(".txt", "")
                annotations = []
                with open(annotation_file, "r") as f:
                    for line in f:
                        start, end, label, confidence = line.strip().split("\t")
                        annotations.append(
                            {
                                "start": float(start),
                                "end": float(end),
                                "label": label,
                                "confidence": float(confidence),
                            }
                        )
                metadata_index[file_id] = {
                    "split": "development",
                    "environment": env,
                    "annotations": annotations,
                }
        return metadata_index
