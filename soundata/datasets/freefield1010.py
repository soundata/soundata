"""freefield1010 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Created By
    ----------

   Dan Stowell*, Mark D. Plumbley*
    * Centre for Digital Music, Queen Mary University of London

    Version 1.0

    Description
    -----------

    Freefield1010 dataset, a freely accessible collection of 7,690 field recording excerpts from various global locations. These recordings have been standardized for research, and both the audio clips and their metadata are under the Creative Commons license. 
    The dataset boasts a wide range of environments and locales. For the BAD Challenge, these recordings have been annotated to indicate whether birds are present or absent. 
    Furthermore, the freefield1010 dataset serves as a development set for the "Bird Audio Detection" challenge, a joint venture by DCASE (Detection and Classification of Acoustic Scenes and Events) and the IEEE Signal Processing Society. 
    This dataset is especially valuable for tasks such as creating and testing bioacoustic classification models.

    Audio Files Included
    --------------------

    The collection contains 7,690 audio clips, sourced from the field-recording tag in the Freesound audio archive (as detailed above). All acquired sounds have been transformed into standard CD-quality mono WAV format. 
    Inside the 'wav' folder, the recordings are stored as WAV files with a 16-bit 44.1 kHz sampling rate. 
    Since amplitude levels in the crowdsourced Freesound archive can vary, which might be an issue for listening tests, they decided to normalize the amplitude of each excerpt in our dataset.

    Meta-data Files Included
    ------------------------

    A table containing a binary label "hasbird" associated to every recording in freefield1010 is available on the website of the DCASE "Bird Audio Detection" challenge: http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/
    

    Please Acknowledge freefield1010 in Academic Research
    ----------------------------------------------------

    When freefield1010 is used for academic research, we would highly appreciate it if scientific publications of works partly based on this dataset cite the following publication:

    .. code-block:: latex
        D. Stowell, M. Plumbley. "An open dataset for research on audio field recording archives: Freefield1010.", Proc. Audio Engineering Society 53rd Conference on Semantic Audio (AES53), 2014.

    Conditions of Use
    -----------------

    Dataset created by Dan Stowell and Mark D. Plumbley

    The freefield1010 dataset is offered free of charge under the terms of the Creative  Commons Attribution 4.0 International (CC BY 4.0) license:
    https://creativecommons.org/licenses/by/4.0/
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
@inproceedings{Stowell:2014f,
  title = {freefield1010 - an open dataset for research on audio field recording archives},
  booktitle={Proceedings of the Audio Engineering Society 53rd Conference on Semantic Audio (AES53)},
  author={Stowell, D. and Plumbley, M. D.},
  year={2014},
  publisher={Audio Engineering Society}}
"""
REMOTES = {
    "dataset": download_utils.RemoteFileMetadata(
        filename="ff1010bird_wav.zip",
        url="https://archive.org/download/ff1010bird/ff1010bird_wav.zip",
        checksum="2f4e7e194ccbd3de86e997af8f2a0405",
        unpack_directories=["wav"],
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="ff1010bird_metadata_2018.csv",
        url="https://ndownloader.figshare.com/files/10853303",
        checksum="2f4e7e194ccbd3de86e997af8f2a0405",
    ),
}

LICENSE_INFO = "Creative Commons Attribution Non Commercial 4.0 International"


class Clip(core.Clip):
    """freefield1010 Clip class

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
    """Load a freefield1010 audio file.

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
    The freefield1010 dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="freefield1010",
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
        metadata_path = os.path.join(self.data_home, "ff1010bird_metadata_2018.csv")

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
