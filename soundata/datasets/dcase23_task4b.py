"""DCASE23 Task 4B Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    *Created By:*
        | Annamaria Mesaros, Tuomas Heittola, and Tuomas Virtanen.
        | Tampere University of Technology. 
    
    Version 1.0

    *Description:*
        The TUT Acoustic Scenes Strong Label Dataset was created to study the estimation of strong labels for sound events using crowdsourcing. It features 49 real-life audio files from 5 different acoustic scenes, complemented by the results of annotation performed via Amazon Mechanical Turk. The dataset spans a total duration of 189 minutes and 52 seconds.

        Audio files are sourced from a subset of the TUT Acoustic Scenes 2016 dataset, covering five acoustic scenes: cafe/restaurant, city center, grocery store, metro station, and residential area. Each scene comprises 6 sound event classes, some of which are common across multiple scenes, resulting in a total of 17 unique classes.

        The dataset includes:
            * Audio: 49 real-life recordings, each ranging from 3 to 5 minutes in length.
            * Soft labels: Crowdsourced estimated strong labels, with values between 0 and 1 indicating the uncertainty of the annotators regarding the presence of sound events.

        For a detailed understanding of the recordings, refer to the paper:
            A. Mesaros, T. Heittola, and T. Virtanen, "TUT database for acoustic scene classification and sound event detection," in 2016 24th European Signal Processing Conference (EUSIPCO), 2016, pp. 1128-1132.

    *Relevant Links:*
        * TUT Acoustic Scenes 2016 dataset: [Insert link to dataset]
        * EUSIPCO 2016 paper: [Insert link to paper]

    *Please Acknowledge TUT Acoustic Scenes Strong Label Dataset in Academic Research:*
        If you use this dataset, please cite the following paper:

            * A. Mesaros, T. Heittola, and T. Virtanen, "TUT database for acoustic scene classification and sound event detection," in 2016 24th European Signal Processing Conference (EUSIPCO), 2016, pp. 1128-1132.

        The dataset's creation was supported by [Insert supporting grants or acknowledgements].

    *License:*
        The TUT Acoustic Scenes Strong Label Dataset is released under [Insert appropriate license here], which governs its use and distribution.

    *Feedback:*
        For questions or feedback, please contact [Insert contact information] or join the [Insert relevant community or forum].
"""

import os
from typing import BinaryIO, Optional, Tuple

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
REMOTES = {
    "development_audio": download_utils.RemoteFileMetadata(
        filename="development_audio.zip",
        url="https://zenodo.org/records/7244360/files/development_audio.zip?download=1",
        checksum="34dc1d34ca44622af5bf439ceb6f0d55",
    ),
    "development_annotation": download_utils.RemoteFileMetadata(
        filename="development_annotation.zip",
        url="https://zenodo.org/records/7244360/files/development_annotation.zip?download=1",
        checksum="1ac73d70b4ef3f81900d98c261a832de",
    ),
    "evaluation_audio": download_utils.RemoteFileMetadata(
        filename="Evaluation_audio.zip",
        url="https://zenodo.org/records/7870026/files/Evaluation_audio.zip?download=1",
        checksum="093a1ca185ec341ca4eac14215e7f96b",
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
        aso_id (str): the id of the corresponding category as per the AudioSet Ontology
        audio_path (str): path to the audio file
        clip_id (str): clip id
        manually_verified (int): flag to indicate whether the clip belongs to the clean portion (1), or to the noisy portion (0) of the train set
        noisy_small (int): flag to indicate whether the clip belongs to the noisy_small portion (1) of the train set
        split (str): flag to indicate whether the clip belongs the train or test split
        tag (soundata.annotations.Tags): tag (label) of the clip + confidence
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

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
    def tags(self):
        """The clip's tags.

        Returns:
            * annotations.Tags - tag (label) of the clip + confidence

        """
        return annotations.Tags(
            [self._clip_metadata.get("tag")], "open", np.array([1.0])
        )

    @property
    def aso_id(self):
        """The clip's Audioset ontology ID.

        Returns:
            * str - the id of the corresponding category as per the AudioSet Ontology

        """
        return self._clip_metadata.get("aso_id")

    @property
    def manually_verified(self):
        """The clip's manually annotated flag.

        Returns:
            * int - flag to indicate whether the clip belongs to the clean portion (1), or to the noisy portion (0) of the train set

        """
        return self._clip_metadata.get("manually_verified")

    @property
    def noisy_small(self):
        """The clip's noisy flag.

        Returns:
            * int - flag to indicate whether the clip belongs to the noisy_small portion (1) of the train set

        """
        return self._clip_metadata.get("noisy_small")

    @property
    def split(self):
        """The clip's split.

        Returns:
            * str - flag to indicate whether the clip belongs the train or test split

        """
        return self._clip_metadata.get("split")

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, tags=self.tags, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a FSDnoisy18K audio file.

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
    The FSDnoisy18K dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="fsdnoisy18k",
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
        metadata_train_path = os.path.join(
            self.data_home, "FSDnoisy18k.meta", "train.csv"
        )
        metadata_test_path = os.path.join(
            self.data_home, "FSDnoisy18k.meta", "test.csv"
        )

        if not os.path.exists(metadata_train_path):
            raise FileNotFoundError(
                "Train metadata not found. Did you run .download()?"
            )
        if not os.path.exists(metadata_test_path):
            raise FileNotFoundError("Test metadata not found. Did you run .download()?")

        metadata_index = {}

        with open(metadata_train_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            for row in reader:
                metadata_index[row[0].replace(".wav", "")] = {
                    "split": "train",
                    "tag": row[1],
                    "aso_id": str(row[2]),
                    "manually_verified": int(row[3]),
                    "noisy_small": int(row[4]),
                }

        with open(metadata_test_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            for row in reader:
                metadata_index[row[0].replace(".wav", "")] = {
                    "split": "test",
                    "tag": row[1],
                    "aso_id": str(row[2]),
                }

        return metadata_index
