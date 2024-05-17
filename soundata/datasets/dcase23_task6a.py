"""DCASE 2023 Task-6A Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **DCASE 2023 Task-6A**

        | Clotho (c) by K. Drossos, S. Lipping, and T. Virtanen.
        | Clotho is licensed under the terms set by Tampere University and Creative Commons licenses for the audio files as per their origin from the Freesound platform.
        | You should have received a copy of the license along with this work.
        | https://github.com/audio-captioning/clotho-dataset. Paper: "Clotho: an Audio Captioning Dataset," ICASSP 2020

    *Created By:*

        | K. Drossos, S. Lipping, and T. Virtanen.
        | Tampere University, Finland
        
    *Version 2.1.0*
        Fixes for corrupted files and illegal characters.
        More details on version changes are available in the dataset repository.

    *Description*
        Clotho is an audio captioning dataset, consisting of 6974 audio samples, each accompanied by five captions, totaling 34,870 captions. 
        
        * Audio samples are 15 to 30 seconds in duration.
        * Captions are 8 to 20 words long.
        * Dataset splits: development, validation, and evaluation.
        * Detailed description and usage guidelines in the ICASSP 2020 paper and dataset repository.

    *Audio Files Included*
        * Development split: 3840 audio files (including 947 new files in version 2)
        * Validation split: 1046 new audio files
        * Evaluation split: No changes from version 1
        * File format: Single channel (mono), various bitrates and sample rates, WAV format.

    *Caption Files Included*
        * Clotho captions in CSV format for each dataset split.
        * Captions follow consistent word usage, no named entities or speech transcription.
        * Unique vocabulary across splits to prevent data leakage.

    *Metadata Files Included*
        * Accompanying metadata for each audio file, including file name, keywords, original URL, excerpt samples, uploader, and license link.

    *Conditions of Use*
        Dataset created by K. Drossos, S. Lipping, and T. Virtanen.
        Audio files under various Creative Commons licenses as per Freesound platform terms.
        Captions under Tampere University license, primarily non-commercial with attribution.
        Full details in the LICENSE file included with the dataset.

    *Acknowledgment in Academic Research*
    When using Clotho for academic research, please cite:
        
    .. code-block:: latex

        K. Drossos, S. Lipping, and T. Virtanen, "Clotho: an Audio Captioning Dataset," ICASSP 2020.

    *Feedback and Contributions*
        Feedback and contributions are welcome.
        Please contact the creators through the GitHub repository.

"""

from collections import defaultdict
import os
from typing import BinaryIO, Optional, TextIO, Tuple
import numpy as np
import csv
import librosa
from soundata import download_utils, jams_utils, core, annotations, io

BIBTEX = """
@inproceedings{Drossos:ICASSP:20,
    Address = {Barcelona, Spain},
    Author = {Drossos, K. and Lipping, S. and Virtanen, T.},
    Booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    Pages = {736--740},
    Title = {Clotho: an Audio Captioning Dataset},
    Year = {2020},
    DOI = {10.1109/ICASSP40776.2020.9052990}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="dcase23_task6a_index_1.0.json",
        url="https://zenodo.org/records/11176789/files/dcase23_task6a_index_1.0.json?download=1",
        checksum="67415822f11bdb4400e422f0136a22f8",
    ),
    "sample": core.Index(filename="dcase23_task6a_index_1.0_sample.json"),
}

REMOTES = {
    "clotho_audio_development": download_utils.RemoteFileMetadata(
        filename="clotho_audio_development.7z",
        url="https://zenodo.org/record/4783391/files/clotho_audio_development.7z?download=1",
        checksum="c8b05bc7acdb13895bb3c6a29608667e",
    ),
    "clotho_audio_evaluation": download_utils.RemoteFileMetadata(
        filename="clotho_audio_evaluation.7z",
        url="https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z?download=1",
        checksum="4569624ccadf96223f19cb59fe4f849f",
    ),
    "clotho_audio_validation": download_utils.RemoteFileMetadata(
        filename="clotho_audio_validation.7z",
        url="https://zenodo.org/record/4783391/files/clotho_audio_validation.7z?download=1",
        checksum="7dba730be08bada48bd15dc4e668df59",
    ),
    "clotho_captions_development": download_utils.RemoteFileMetadata(
        filename="clotho_captions_development.csv",
        url="https://zenodo.org/record/4783391/files/clotho_captions_development.csv?download=1",
        checksum="d4090b39ce9f2491908eebf4d5b09bae",
    ),
    "clotho_captions_evaluation": download_utils.RemoteFileMetadata(
        filename="clotho_captions_evaluation.csv",
        url="https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv?download=1",
        checksum="1b16b9e57cf7bdb7f13a13802aeb57e2",
    ),
    "clotho_captions_validation": download_utils.RemoteFileMetadata(
        filename="clotho_captions_validation.csv",
        url="https://zenodo.org/record/4783391/files/clotho_captions_validation.csv?download=1",
        checksum="5879e023032b22a2c930aaa0528bead4",
    ),
    "clotho_metadata_development": download_utils.RemoteFileMetadata(
        filename="clotho_metadata_development.csv",
        url="https://zenodo.org/record/4783391/files/clotho_metadata_development.csv?download=1",
        checksum="170d20935ecfdf161ce1bb154118cda5",
    ),
    "clotho_metadata_evaluation": download_utils.RemoteFileMetadata(
        filename="clotho_metadata_evaluation.csv",
        url="https://zenodo.org/record/4783391/files/clotho_metadata_evaluation.csv?download=1",
        checksum="13946f054d4e1bf48079813aac61bf77",
    ),
    "clotho_metadata_validation": download_utils.RemoteFileMetadata(
        filename="clotho_metadata_validation.csv",
        url="https://zenodo.org/record/4783391/files/clotho_metadata_validation.csv?download=1",
        checksum="2e010427c56b1ce6008b0f03f41048ce",
    ),
    "clotho_audio_test": download_utils.RemoteFileMetadata(
        filename="clotho_audio_test.7z",
        url="https://zenodo.org/records/3865658/files/clotho_audio_test.7z?download=1",
        checksum="9b3fe72560a621641ff4351ba1154349",
    ),
    "clotho_metadata_test": download_utils.RemoteFileMetadata(
        filename="clotho_metadata_test.csv",
        url="https://zenodo.org/records/3865658/files/clotho_metadata_test.csv?download=1",
        checksum="52f8ad01c229a310a0ff8043df480e21",
    ),
}

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Clip(core.Clip):
    """DCASE'23 Task 6A Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): Audio signal and sample rate.
        file_name (str): Name of the file.
        keywords (str): Associated keywords.
        sound_id (str): Unique identifier for the sound.
        sound_link (str): Link to the sound.
        start_end_samples (tuple): Start and end samples in the audio file.
        manufacturer (str): Manufacturer of the recording equipment.
        license (str): License of the clip.
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
    def file_name(self):
        """The name of the audio file.

        Returns:
            * str - Name of the file.
        """
        return self._clip_metadata.get("file_name")

    @property
    def keywords(self):
        """Keywords associated with the clip.

        Returns:
            * str - Keywords for the clip.
        """
        return self._clip_metadata.get("keywords")

    @property
    def sound_id(self):
        """Unique identifier for the sound.

        Returns:
            * str - Sound ID.
        """
        return self._clip_metadata.get("sound_id")

    @property
    def sound_link(self):
        """Link to the sound.

        Returns:
            * str - URL of the sound.
        """
        return self._clip_metadata.get("sound_link")

    @property
    def start_end_samples(self):
        """Start and end samples in the audio file.

        Returns:
            * tuple - Start and end samples.
        """
        return self._clip_metadata.get("start_end_samples")

    @property
    def manufacturer(self):
        """Manufacturer of the recording equipment.

        Returns:
            * str - Manufacturer name.
        """
        return self._clip_metadata.get("manufacturer")

    @property
    def license(self):
        """License of the clip.

        Returns:
            * str - License information.
        """
        return self._clip_metadata.get("license")

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a  DCASE'23 Task 6A audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate of 44100 without resampling.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=False)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The DCASE'23 Task 6A dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="dcase23_task6a",
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
        # Define all the metadata and caption files for both datasets
        files = {
            "clotho_metadata_development.csv": "metadata",
            "clotho_metadata_evaluation.csv": "metadata",
            "clotho_metadata_validation.csv": "metadata",
            "clotho_captions_development.csv": "captions",
            "clotho_captions_evaluation.csv": "captions",
            "clotho_captions_validation.csv": "captions",
            "clotho_metadata_test.csv": "test_metadata",  # Differentiate the test metadata
        }
        combined_data = {}

        # Process each file
        for file_name, file_type in files.items():
            file_path = os.path.join(self.data_home, file_name)
            delimiter = ";" if file_type == "test_metadata" else ","
            with open(file_path, encoding="ISO-8859-1") as csv_file:
                csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
                for row in csv_reader:
                    file_key = row["file_name"].replace(
                        ".wav", ""
                    )  # Assuming 'file_name' is the header for the first column
                    dataset_type = file_name.split("_")[2].split(".")[
                        0
                    ]  # development, validation, evaluation, test
                    file_key = f"{dataset_type}/{file_key}"
                    if file_key not in combined_data:
                        combined_data[file_key] = {
                            "file_name": "",
                            "keywords": "",
                            "sound_id": "",
                            "sound_link": "",
                            "start_end_samples": "",
                            "manufacturer": "",
                            "license": "",
                            "captions": [],
                        }
                    if file_type == "metadata":
                        combined_data[file_key].update(
                            {
                                "file_name": file_key,
                                "keywords": row[
                                    "keywords"
                                ],  # Replace with actual header names
                                "sound_id": row["sound_id"],
                                "sound_link": row["sound_link"],
                                "start_end_samples": row["start_end_samples"],
                                "manufacturer": row["manufacturer"],
                                "license": row["license"],
                            }
                        )
                    elif file_type == "test_metadata":
                        combined_data[file_key].update(
                            {
                                "file_name": file_key,
                                "start_end_samples": row["start_end_samples"],
                                "manufacturer": row["manufacturer"],
                                "license": row["license"],
                            }
                        )
                    elif file_type == "captions":
                        combined_data[file_key]["captions"] = [
                            row[key] for key in row if key != "file_name"
                        ]  # Assuming rest of the keys are captions

        return combined_data
