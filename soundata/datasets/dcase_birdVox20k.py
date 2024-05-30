"""BirdVox20k Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **BirdVox20k**

    *Created By*  

        | Vincent Lostanlen*^#, Justin Salamon^#, Andrew Farnsworth*, Steve Kelling*, and Juan Pablo Bello^#.
        | * Cornell Lab of Ornithology (CLO).
        | ^ Center for Urban Science and Progress, New York University.
        | # Music and Audio Research Lab, New York University.

    Version 1.0

    *Description*
        The BirdVox-DCASE-20k dataset contains 20,000 ten-second audio recordings. These recordings come from ROBIN autonomous recording units, placed near Ithaca, NY, USA during the fall 2015. They were captured on the night of September 23rd, 2015, by six different sensors, originally numbered 1, 2, 3, 5, 7, and 10.
        Out of these 20,000 recording, 10,017 (50.09%) contain at least one bird vocalization (either song, call, or chatter).
        The dataset is a derivative work of the BirdVox-full-night dataset [1], containing almost as much data but formatted into ten-second excerpts rather than ten-hour full night recordings.
        In addition, the BirdVox-DCASE-20k dataset is provided as a development set in the context of the "Bird Audio Detection" challenge, organized by DCASE (Detection and Classification of Acoustic Scenes and Events) and the IEEE Signal Processing Society.
        The dataset can be used, among other things, for the development and evaluation of bioacoustic classification models.

    *Audio Files Included*
        20,000 ten-second audio recordings (see description above) in WAV format. The wav folder contains the recordings as WAV files, sampled at 44,1 kHz, with a single channel (mono). The original sample rate was 24 kHz.

    *Meta-data Files Included*
        A table containing a binary label "hasbird" associated to every recording in BirdVox-DCASE-20k is available on the website of the DCASE "Bird Audio Detection" challenge: http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/
        These labels were automatically derived from the annotations of avian flight call events in the BirdVox-full-night dataset.

    *Please Acknowledge UrbanSound8K in Academic Research*
        When BirdVox-70k is used for academic research, we would highly appreciate it if  scientific publications of works partly based on this dataset cite the  following publication:

        .. code-block:: latex

            V. Lostanlen, J. Salamon, A. Farnsworth, S. Kelling, J. Bello. "BirdVox-full-night: a dataset and benchmark for avian flight call detection", Proc. IEEE ICASSP, 2018.
        
        The creation of this dataset was supported by NSF grants 1125098 (BIRDCAST) and 1633259 (BIRDVOX), a Google Faculty Award, the Leon Levy Foundation, and two anonymous donors.

    *Conditions of Use*
        Dataset created by Vincent Lostanlen, Justin Salamon, Andrew Farnsworth, Steve Kelling, and Juan Pablo Bello.

        The BirdVox-DCASE-20k dataset is offered free of charge under the terms of the Creative  Commons Attribution 4.0 International (CC BY 4.0) license:
        https://creativecommons.org/licenses/by/4.0/

        The dataset and its contents are made available on an "as is" basis and without  warranties of any kind, including without limitation satisfactory quality and  conformity, merchantability, fitness for a particular purpose, accuracy or  completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, Cornell Lab of Ornithology is not liable for, and expressly excludes all liability for, loss or damage however and whenever caused to anyone by any use of the BirdVox-DCASE-20k dataset or any part of it.

    *Feedback*
        Please help us improve BirdVox-DCASE-20k by sending your feedback to:  
        * Vincent Lostanlen: vincent.lostanlen@gmail.com for feedback regarding data pre-processing,
        * Andrew Farnsworth: af27@cornell.edu for feedback regarding data collection and ornithology, or
        * Dan Stowell: dan.stowell@qmul.ac.uk for feedback regarding the DCASE "Bird Audio Detection" challenge.

        In case of a problem, please include as many details as possible.

"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
@inproceedings{lostanlen2018icassp,
  title = {BirdVox-full-night: a dataset and benchmark for avian flight call detection},
  author = {Lostanlen, Vincent and Salamon, Justin and Farnsworth, Andrew and Kelling, Steve and Bello, Juan Pablo},
  booktitle = {Proc. IEEE ICASSP},
  year = {2018},
  published = {IEEE},
  venue = {Calgary, Canada},
  month = {April},
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="dcase_birdVox20k_index_1.0.json",
        url="https://zenodo.org/records/11176775/files/dcase_birdVox20k_index_1.0.json?download=1",
        checksum="d68016f669df15b67b5af1c4043593b9",
    ),
    "sample": core.Index(filename="dcase_birdVox20k_index_1.0_sample.json"),
}

REMOTES = {
    "dataset": download_utils.RemoteFileMetadata(
        filename="BirdVox-DCASE-20k.zip",
        url="https://zenodo.org/record/1208080/files/BirdVox-DCASE-20k.zip?download=1",
        checksum="2f4e7e194ccbd3de86e997af8f2a0405",
        unpack_directories=["wav"],
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="BirdVoxDCASE20k_csvpublic.csv",
        url="https://ndownloader.figshare.com/files/10853300",
        checksum="2f4e7e194ccbd3de86e997af8f2a0405",
    ),
}

LICENSE_INFO = "Creative Commons Attribution Non Commercial 4.0 International"


class Clip(core.Clip):
    """BirdVox20k Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        itemid (str): clip id
        datasetid (str): the dataset to which the clip belongs to
        hasbird (str): indication of whether the clips contains bird sounds (0/1)
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
    def item_id(self):
        """The clip's item ID.

        Returns:
            * str - ID of the clip

        """
        return self._clip_metadata.get("itemid")

    @property
    def dataset_id(self):
        """The clip's dataset ID.

        Returns:
            * str - ID of the dataset from where this clip is extracted

        """
        return self._clip_metadata.get("datasetid")

    @property
    def has_bird(self):
        """The flag to tell whether the clip has bird sound or not.

        Returns:
            * str - 1/0 depending on whether the clip contains bird sound

        """
        return self._clip_metadata.get("hasbird")

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=44100) -> Tuple[np.ndarray, float]:
    """Load a BirdVox20k audio file.

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
    The BirdVox20k dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="dcase_birdVox20k",
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
        metadata_path = os.path.join(self.data_home, "BirdVoxDCASE20k_csvpublic.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. Did you run .download()?"
            )

        with open(metadata_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            raw_data = [line for line in reader if line[0] != "slice_file_name"]

        metadata_index = {}
        for line in raw_data:
            clip_id = line[0].replace(".wav", "")

            metadata_index[clip_id] = {
                "itemid": line[0],
                "datasetid": line[1],
                "hasbird": line[2],
            }

        return metadata_index
