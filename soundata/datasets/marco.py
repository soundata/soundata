"""3D-MARCo Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown
    
    **3D-MARCo: database of 3D sound recordings of musical performances and room impulse responses**
    
    *Created By:*

        | Hyunkook Lee, Dale Johnson, Bogdan Bacila.
        | Centre for Audio and Psychoacoustic Engineering, University of Huddersfield. 
        
    Version 1.0.1
	
    *Description:*
        3D-MARCo is an open-access database of 3D sound recordings of musical performances and room impulse responses. 
        The recordings were made in the St. Paul's concert hall in Huddersfield, UK 
        A total of 71 microphone capsules were used simultaneously. 
        The main microphone arrays included in the database comprise PCMA-3D, OCT-3D, 2L-Cube, Decca Cubioid, First-order Ambisonics (FOA), Higher-order Ambisonics (HOA) and Hamasaki Square with height. 
        In addition, ORTF, side/height, Voice of God and floor channels as well as a dummy head and spot microphones are included. 
        The sound sources recorded are string quartet, piano trio, piano solo, organ, a cappella group, various single sources and room impulse responses of a virtual ensemble with 13 source positions captured by all of the microphones. 
        3D-MARCo would be useful for spatial audio research, recording education, critical ear training, etc.

    *Audio Files Included:*
        * For each musical performance sound source (Acappella, Organ, Piano Solo 1, Piano solo 2, Quartet, Trio), there are 65 wav files that correspond to: 
            * 64 individual capsules (24-bit / 96kHz resolution) 
            * one 32-channel EigenMike file in A-format (24-bit / 48kHz resolution). 
        * The piano recordings contain two more channels (left and right) that correspond to spot microphones placed just outside the piano pointing toward the hammers.
        * The quartet recordings contain four more channels corresponding to spot microphones placed above the instruments (violin 1, violin 2, cello, viola) pointing toward the F hole.
        * The trio recordins contain four more channels corresponding to spot microphones, two placed above the string instruments (violin, cello) pointing toward the F hole, and two placed just outside the piano pointing toward the hammers.
        * The single sources were recorded at 7 different azimuth angles. For each angle there are also 65 wav files.
        * The impulse responses were recorded at 13 different azimuth angles. For each angle there are 66 wav files. The extra one is the EigenMike 4th-order B-format ambisonics (ACN SN3D; 24-bit / 48kHz resolution). 
	
    *Annotations Included:*
        * No event labels associated with this dataset
        * No predefined training, validation, or testing splits. 
        * Angular orientation for "impulse responses" and "single sources" (follows the ITU-R convention where positive angles in the left-hand side and negative angles in the right-hand side, e.g. +30° for Front Left and -30° for Front Right).
	
    *Please Acknowledge 3D-MARCo in Academic Research:*
    If you use this dataset please cite its original publication:
    
    .. code-block:: latex
    
        Lee H, Johnson D. An open-access database of 3D microphone array recordings. InAudio Engineering Society Convention 147 2019 Oct 8. Audio Engineering Society.
	    
    *License:*
        * CC-BY NC 3.0 license (free to share and adapt the material, but not permitted to use it for commercial purposes)
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
@inproceedings{lee2019open,
  title={An open-access database of 3D microphone array recordings},
  author={Lee, Hyunkook and Johnson, Dale},
  booktitle={Audio Engineering Society Convention 147},
  year={2019},
  organization={Audio Engineering Society}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="marco_index_1.0.1.json",
        url="https://zenodo.org/records/11176835/files/marco_index_1.0.1.json?download=1",
        checksum="caf2a5c17bbe75ff6c26c450cb24bcb7",
    ),
    "sample": core.Index(filename="marco_index_1.0.1_sample.json"),
}

REMOTES = {
    "ImpulseResponses": download_utils.RemoteFileMetadata(
        filename="03 3D-MARCo Impulse Responses.zip",
        url="https://zenodo.org/record/3477602/files/03%203D-MARCo%20Impulse%20Responses.zip?download=1",
        checksum="d328425ee2d1e847e225d78b676cd81e",
    ),
    "Quartet": download_utils.RemoteFileMetadata(
        filename="04 3D-MARCo Samples_Quartet.zip",
        url="https://zenodo.org/record/3477602/files/04%203D-MARCo%20Samples_Quartet.zip?download=1",
        checksum="cce3442ae5a11ea869412c2e6a4cadcd",
    ),
    "Trio": download_utils.RemoteFileMetadata(
        filename="05 3D-MARCo Samples_Trio.zip",
        url="https://zenodo.org/record/3477602/files/05%203D-MARCo%20Samples_Trio.zip?download=1",
        checksum="48262496ecb6a32843e4b69393eeeec1",
    ),
    "Organ": download_utils.RemoteFileMetadata(
        filename="06 3D-MARCo Samples_Organ.zip",
        url="https://zenodo.org/record/3477602/files/06%203D-MARCo%20Samples_Organ.zip?download=1",
        checksum="cd015829e0a2bfc0aac239adc2b86321",
    ),
    "PianoSolo1": download_utils.RemoteFileMetadata(
        filename="07 3D-MARCo Samples_Piano solo 1.zip",
        url="https://zenodo.org/record/3477602/files/07%203D-MARCo%20Samples_Piano%20solo%201.zip?download=1",
        checksum="4a27da19a0bc857967e47b0044abf128",
    ),
    "PianoSolo2": download_utils.RemoteFileMetadata(
        filename="08 3D-MARCo Samples_Piano solo 2.zip",
        url="https://zenodo.org/record/3477602/files/08%203D-MARCo%20Samples_Piano%20solo%202.zip?download=1",
        checksum="7372b3a1273bcf10ade09472c3a92eed",
    ),
    "Acappella": download_utils.RemoteFileMetadata(
        filename="09 3D-MARCo Samples_Acappella.zip",
        url="https://zenodo.org/record/3477602/files/09%203D-MARCo%20Samples_Acappella.zip?download=1",
        checksum="9ce5a1e973fa04c084495f509f855225",
    ),
    "SingleSources": download_utils.RemoteFileMetadata(
        filename="10 3D-MARCo Samples_Single sources.zip",
        url="https://zenodo.org/record/3477602/files/10%203D-MARCo%20Samples_Single%20sources.zip?download=1",
        checksum="389e774c829a0729047bd8802021b239",
    ),
}

LICENSE_INFO = """
CC-BY NC 3.0 license
"""


class Clip(core.Clip):
    """3D-MARCo Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        source_label (str): label of the source being recorded
        source_angle (str): angle of the source being recorded
        audio_path (str): path to the audio file
        clip_id (str): clip id
        microphone_info (list): list of strings with all relevant microphone metadata
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

        source_label = self._clip_metadata.get("source_label")
        self.source_label = source_label

        source_angle = self._clip_metadata.get("source_angle")
        if source_angle is None:
            self.source_angle = None
        else:
            self.source_angle = source_angle

        self.microphone_info = self._clip_metadata.get("microphone_info")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, tags=None, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=48000) -> Tuple[np.ndarray, float]:
    """Load a 3D-MARCo audio file

    Args:
        fhandle (str or file-like): file-like object or path to audio file
        sr (int or None): sample rate for loaded audio, 48000 by default, which re-samples all files except the EigenMike ones, resulting in constant sampling rate between all clips in the dataset.

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file
    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=False)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The 3D-MARCo dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="marco",
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
        # parsing the data from the filenames due to lack of metadata file
        metadata_index = {}

        with open(self.index_path) as f:
            marco_index = json.load(f)
            all_paths_filenames = list(marco_index["clips"].keys())

        for path_filename in all_paths_filenames:
            clip_id = path_filename
            path, filename = path_filename.split("/")
            source_label = path
            clip_metadata = filename.split("_")

            # remove arbitrary clip numbering used by dataset authors
            clip_metadata = [
                data for data in clip_metadata if data != "" and data[0] != "0"
            ]
            microphone_info = clip_metadata[1:]
            if "deg" in clip_metadata[0]:
                source_angle = "".join(clip_metadata[0].partition("deg")[:2])
            else:
                source_angle = None
            metadata_index[clip_id] = {
                "source_label": source_label,
                "source_angle": source_angle,
                "microphone_info": microphone_info,
            }

        return metadata_index
