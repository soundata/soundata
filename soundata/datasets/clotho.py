"""Clotho Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **Clotho**

        | Clotho (c) by Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen.
        | Clotho is licensed under the terms set by Tampere University.
        | You should have received a copy of the license along with this work. If not, see https://github.com/audio-captioning/clotho-dataset.

    *Created By:*

        | K. Drossos, S. Lipping, and T. Virtanen.
        | Tampere University, Finland

    *Version 2.1*
        In version 2.1 of Clotho, we fixed some files that were corrupted from the compression and transferring processes (around 150 files) and we also replaced some characters that were illegal for most filesystems, e.g. ":" (around 10 files).


    *Description*

        Clotho is an audio captioning dataset, now reached version 2.1. Clotho consists of 5929 audio samples, and each audio sample has five captions (a total of 34 870 captions). Audio samples are of 15 to 30 s duration and captions are eight to 20 words long.
        Clotho is thoroughly described in our paper:
        K. Drossos, S. Lipping and T. Virtanen, "Clotho: an Audio Captioning Dataset," IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 736-740, doi: 10.1109/ICASSP40776.2020.9052990.
        available online at: https://arxiv.org/abs/1910.09387 and at: https://ieeexplore.ieee.org/document/9052990


    *Audio Files Included*
        Development split: 3839 audio files (Version 2.1)
        Validation split 1045 audio files (Version 2.1)
        Evaluation split: 1045 audio files (Version 2.1)
        File format: Single Channel (MONO), 16 bits, 44.1 kHz sample rates, .WAV format


    *Caption Files Included*
        Captions in CSV format for each dataset split.
        Captions includes 5 different captions (caption_1 ~ caption_5)

    *Metadata Files Included*
        Metadata in CSV format for each dataset split.
        Metadata includes keywords, sound_id, sound_link, start_end_samples, manufacturer, license.

    *Please Acknowledge URBAN-SED in Academic Research*
        For using Cloto for academic research, please cite the paper. The paper can be found from :https://ieeexplore.ieee.org/document/9052990


    *Conditions of Use*
        Dataset created by Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen.
        Audio files under various Creative Commons licenses as per Freesound platform terms.
        Captions under Tampere University license, primarily non-commercial with attribution.
        Full details in the LICENSE file included with the dataset.


    *Feedback*
        Feedback and contributions are welcome.
        Please contact the creators through the GitHub repository.

"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import glob

import pandas as pd

from soundata import download_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
@INPROCEEDINGS{9052990,
  author={Drossos, Konstantinos and Lipping, Samuel and Virtanen, Tuomas},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Clotho: an Audio Captioning Dataset}, 
  year={2020},
  volume={},
  number={},
  pages={736-740},
  keywords={Training;Conferences;Employment;Signal processing;Task analysis;Speech processing;Tuning;audio captioning;dataset;Clotho},
  doi={10.1109/ICASSP40776.2020.9052990}}
"""

INDEXES = {
    "default": "2.1",
    "test": "sample",
    "2.1": core.Index(
        filename="clotho_index.json",
        url="https://zenodo.org/records/15208093/files/clotho_index.json?download=1&preview=1",  # NOT PUBLISHED YET
        checksum="da0e7fbffcd18a59e2da830e62340343",
    ),
    "sample": core.Index(filename="clotho_index_2.1_sample.json"),
}

REMOTES = {
    "clotho_audio_development": download_utils.RemoteFileMetadata(
        filename="clotho_audio_development.7z",
        url="https://zenodo.org/records/4783391/files/clotho_audio_development.7z?download=1",
        checksum="c8b05bc7acdb13895bb3c6a29608667e",
    ),
    "clotho_audio_evaluation": download_utils.RemoteFileMetadata(
        filename="clotho_audio_evaluation.7z",
        url="https://zenodo.org/records/4783391/files/clotho_audio_evaluation.7z?download=1",
        checksum="4569624ccadf96223f19cb59fe4f849f",
    ),
    "clotho_audio_validation": download_utils.RemoteFileMetadata(
        filename="clotho_audio_validation.7z",
        url="https://zenodo.org/records/4783391/files/clotho_audio_validation.7z?download=1",
        checksum="7dba730be08bada48bd15dc4e668df59",
    ),
    "clotho_captions_development": download_utils.RemoteFileMetadata(
        filename="clotho_captions_development.csv",
        url="https://zenodo.org/records/4783391/files/clotho_captions_development.csv?download=1",
        checksum="d4090b39ce9f2491908eebf4d5b09bae",
    ),
    "clotho_captions_evaluation": download_utils.RemoteFileMetadata(
        filename="clotho_captions_evaluation.csv",
        url="https://zenodo.org/records/4783391/files/clotho_captions_evaluation.csv?download=1",
        checksum="1b16b9e57cf7bdb7f13a13802aeb57e2",
    ),
    "clotho_captions_validation": download_utils.RemoteFileMetadata(
        filename="clotho_captions_validation.csv",
        url="https://zenodo.org/records/4783391/files/clotho_captions_validation.csv?download=1",
        checksum="5879e023032b22a2c930aaa0528bead4",
    ),
    "clotho_metadata_development": download_utils.RemoteFileMetadata(
        filename="clotho_metadata_development.csv",
        url="https://zenodo.org/records/4783391/files/clotho_metadata_development.csv?download=1",
        checksum="170d20935ecfdf161ce1bb154118cda5",
    ),
    "clotho_metadata_evaluation": download_utils.RemoteFileMetadata(
        filename="clotho_metadata_evaluation.csv",
        url="https://zenodo.org/records/4783391/files/clotho_metadata_evaluation.csv?download=1",
        checksum="13946f054d4e1bf48079813aac61bf77",
    ),
    "clotho_metadata_validation": download_utils.RemoteFileMetadata(
        filename="clotho_metadata_validation.csv",
        url="https://zenodo.org/records/4783391/files/clotho_metadata_validation.csv?download=1",
        checksum="2e010427c56b1ce6008b0f03f41048ce",
    ),
}
#######
LICENSE_INFO = "Creative Commons Attribution 4.0 International"
#######


class Clip(core.Clip):
    """Clotho Clip class

    Args:
        clip_id (str):id of the clip

    Attributes:
        audio (np.ndarray, float): Audio signal and sample rate.
        file_name (str): Name of the file.
        keywords (str): Associated keywords.
        sound_id (str): Unique identifier for the sound.
        sound_link (str): Link to the sound.
        start_end_samples (tuple): Start and end samples in the audio file.
        manufacturer (str): Manufacturer of the recording equipment.
        captions (list): Captions for the clip
        license (str): License of the clip.
        split (str): split
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
    def captions(self):
        """Captions for the clip.

        Returns:
            * list - Captions.
        """
        return self._clip_metadata.get("captions")

    @property
    def license(self):
        """License of the clip.

        Returns:
            * str - License information.
        """
        return self._clip_metadata.get("license")

    @property
    def split(self):
        """Split of the clip.

        Returns:
            * str - split name
        """

        return self._clip_metadata.get("split")


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a Clotho audio file.

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


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The Clotho dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="clotho",
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
        # Name of each splits
        splits = ["development", "validation", "evaluation"]

        # Create empty index dictionary
        metadata_index = {}

        # Process through each split
        for split in splits:

            metadata_path = os.path.join(self.data_home, f"clotho_metadata_{split}.csv")
            captions_path = os.path.join(self.data_home, f"clotho_captions_{split}.csv")

            metadata_df = pd.read_csv(metadata_path, encoding="ISO-8859-1")
            captions_df = pd.read_csv(captions_path, encoding="ISO-8859-1")

            # Create clip_id in df by removing .wav from the file_name
            captions_df["clip_id"] = captions_df["file_name"].apply(
                lambda x: x.replace(".wav", "")
            )
            metadata_df["clip_id"] = metadata_df["file_name"].apply(
                lambda x: x.replace(".wav", "")
            )

            for _, row in metadata_df.iterrows():
                clip_id = row["clip_id"]

                # find matching row in captions_df
                caption_row = captions_df[captions_df["clip_id"] == clip_id].iloc[0]

                metadata_index[clip_id] = {
                    "clip_id": str(clip_id),
                    "file_name": str(row["file_name"]),
                    "keywords": str(row.get("keywords", "")),
                    "sound_id": str(row.get("sound_id", "")),
                    "sound_link": str(row.get("sound_link", "")),
                    "start_end_samples": str(row.get("start_end_samples", "")),
                    "manufacturer": str(row.get("manufacturer", "")),
                    "license": str(row.get("license", "")),
                    "captions": [
                        caption_row.get("caption_1", ""),
                        caption_row.get("caption_2", ""),
                        caption_row.get("caption_3", ""),
                        caption_row.get("caption_4", ""),
                        caption_row.get("caption_5", ""),
                    ],
                    "split": split,
                }

            return metadata_index
