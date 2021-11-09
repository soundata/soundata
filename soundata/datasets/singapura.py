"""SINGA:PURA Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown
    
    SINGA:PURA (SINGApore: Polyphonic URban Audio) v1.0
    
    The SINGA:PURA (SINGApore: Polyphonic URban Audio) dataset is a strongly-labelled polyphonic urban sound dataset with spatiotemporal context, created by the Digital Signal Processing Laboratory, School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore.
    
    The dataset contains 6547 strongly-labelled and 72406 unlabelled recordings from a wireless acoustic sensor network deployed in Singapore to identify and mitigate noise sources in Singapore. The strongly-labelled and unlabelled recordings are disjoint, so there are a total of 78953 unique recordings. The recordings are all 10 seconds in length, and may have 1 or 7 channels, depending on the recording device used to record them. Total duration for the labelled subset provided here is 18.2 hours.
    
    For full details regarding the sensor units used, the recording conditions, and annotation methodology, please refer to our conference paper below:
    
    ```
    K. Ooi, K. N. Watcharasupat, S. Peksi, F. A. Karnapi, Z.-T. Ong, D. Chua, H.-W. Leow, L.-L. Kwok, X.-L. Ng, Z.-A. Loh, W.-S. Gan, "A Strongly-Labelled Polyphonic Dataset of Urban Sounds with Spatiotemporal Context," in Proceedings of the 13th Asia Pacific Signal and Information Processing Association Annual Summit and Conference, 2021.
    ```
    
    Our label taxonomy is derived from the taxonomy used in the SONYC-UST datasets, but has been adapted to fit the local (Singapore) context while retaining compatibility with the SONYC-UST ontonology. We chose this taxonomy to allow the SINGA:PURA dataset to be used in conjunction with the SONYC-UST datasets when training urban sound tagging models by simply omitting the labels that are absent in the SONYC-UST taxonomy from the recordings in the SINGA:PURA dataset. For more information regarding the SONYC-UST datasets, please refer to the following paper published by the SONYC team:
    ```
    M. Cartwright, J. Cramer, A. E. M. Mendez, Y. Wang, H. Wu, V. Lostanlen, M. Fuentes, G. Dove, C. Mydlarz, J. Salamon, O. Nov, J. P. Bello, "SONYC-UST-V2: An Urban Sound Tagging Dataset with Spatiotemporal Context," in Proceedings of the Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2020.
    ```

    Specifically, our label taxonomy consists of 14 coarse-grained classes and 40 fine-grained classes. Their organisation is as follows:
    
    ─┬─ 1. Engine ───────────────┬─ 1. Small engine
     │                           ├─ 2. Medium engine
     │                           └─ 3. Large engine
     ├─ 2. Machinery impact ─────┬─ 1. Rock drill
     │                           ├─ 2. Jackhammer
     │                           ├─ 3. Hoe ram
     │                           └─ 4. Pile driver
     ├─ 3. Non-machinery impact ─┬─ 1. Glass breaking*
     │                           ├─ 2. Car crash*
     │                           └─ 3. Explosion*
     ├─ 4. Powered saw ──────────┬─ 1. Chainsaw
     │                           ├─ 2. Small/medium rotating saw
     │                           └─ 3. Large rotating saw
     ├─ 5. Alert signal ─────────┬─ 1. Car horn
     │                           ├─ 2. Car alarm
     │                           ├─ 3. Siren
     │                           └─ 4. Reverse beeper
     ├─ 6. Music ────────────────┬─ 1. Stationary music
     │                           └─ 2. Mobile music
     ├─ 7. Human voice ──────────┬─ 1. Talking
     │                           ├─ 2. Shouting
     │                           ├─ 3. Large crowd
     │                           ├─ 4. Amplified speech
     │                           └─ 5. Singing*
     ├─ 8. Human movement* ──────┬─ 1. Footsteps*
     │                           └─ 2. Clapping*
     ├─ 9. Animal* ──────────────┬─ 1. Dog barking
     │                           ├─ 2. Bird chirping*
     │                           └─ 3. Insect chirping*
     ├─ 10. Water* ──────────────── 1. Hose pump*
     ├─ 11. Weather* ────────────┬─ 1. Rain*
     │                           ├─ 2. Thunder*
     │                           └─ 3. Wind*
     ├─ 12. Brake* ──────────────┬─ 1. Friction brake*
     │                           └─ 2. Exhaust brake*
     ├─ 13. Train* ──────────────── 1. Electric train*
     └─ X. Others* ──────────────┬─ 1. Screeching*
                                 ├─ 2. Plastic crinkling*
                                 ├─ 3. Cleaning*
                                 └─ 4. Gear*

    Classes marked with an asterisk (`*`) are present in the SINGA:PURA taxonomy but not the SONYC taxonomy. The "Ice cream truck" class from the SONYC taxonomy has been excluded from the SINGA:PURA taxonomy because this class does not exist in the local context.
    
    This dataset is also accessible via
    - Zenodo (labelled subset only): https://zenodo.org/record/5645825
    - DR-NTU (all): https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/Y8UQ6F
    
    This dataset is licensed under the Creative Commons Attribution-ShareAlike 4.0 International license (a human-readable summary is available at https://creativecommons.org/licenses/by-sa/4.0/ and the legal document for the license is available at https://creativecommons.org/licenses/by-sa/4.0/legalcode).

"""
import os


import librosa
import numpy as np
import pandas as pd

from soundata import download_utils, jams_utils, core, annotations, io

# -- Add any relevant citations here
BIBTEX = """
@inproceedings{ooi2021singapura,
    author    = "K. Ooi and K. N. Watcharasupat and S. Peksi and F. A. Karnapi and Z.-T. Ong and D. Chua and H.-W. Leow and L.-L. Kwok and X.-L. Ng and Z.-A. Loh and W.-S. Gan",
    title     = "A Strongly-Labelled Polyphonic Dataset of Urban Sounds with Spatiotemporal Context",
    booktitle = "Proceedings of the 13th Asia Pacific Signal and Information Processing Association Annual Summit and Conference",
    location  = "Tokyo, Japan"
    year      = 2021
}
"""

# -- REMOTES is a dictionary containing all files that need to be downloaded.
# -- The keys should be descriptive (e.g. 'annotations', 'audio').
# -- When having data that can be partially downloaded, remember to set up
# -- correctly destination_dir to download the files following the correct structure.

meta_files = [
    ("metadata", "labelled_metadata_public.csv", "c5beb6374e55abfe7cd50f4f498c8376"),
    (
        "labels",
        "labels_public.zip",
        "535242cf1094d95d086fc574874e9ddf",
    ),  # the base zip should be the last file
]

audio_files = [
    ("labelled.zip", "bdd46cc5e9187e97c37989b3b73e786e"),
    ("labelled.z01", "98477daca861c6950cc8b620cecc286d"),
    ("labelled.z02", "873b26cfe25bb3084e39e5af0dfebcad"),
    ("labelled.z03", "71322a5c3ba33badfdd8e25c9ebf559a"),
    ("labelled.z04", "a6d087babea1f797af99b81cb7c7ea4a"),
]

meta_remotes = {
    k: download_utils.RemoteFileMetadata(
        filename=f,
        url=f"https://zenodo.org/record/5645825/files/{f}?download=1",
        checksum=m,  # -- the md5 checksum
        destination_dir=None,  # -- relative path for where to unzip the data, or None
    )
    for k, f, m in meta_files
}

# put as list for multipart zip
audio_remotes = {
    "audio": [
        download_utils.RemoteFileMetadata(
            filename=f,
            url=f"https://zenodo.org/record/5645825/files/{f}?download=1",
            checksum=m,  # -- the md5 checksum
            destination_dir=None,  # -- relative path for where to unzip the data, or None
        )
        for f, m in audio_files
    ]
}

REMOTES = {**meta_remotes, **audio_remotes}

# -- Include any information that should be printed when downloading
# -- remove this variable if you don't need to print anything during download
DOWNLOAD_INFO = """
SINGA:PURA (SINGApore: Polyphonic URban Audio) v1.0a

Labelled data subset downloaded from https://zenodo.org/record/5645825.
"""

# -- Include the dataset's license information
LICENSE_INFO = "Creative Commons Attribution-ShareAlike 4.0 International"


class Clip(core.Clip):
    """
    Args:
        clip_id (str): clip id of the clip

    Attributes:
        clip_id (str): clip id
        audio_path (str): path to the audio file
        audio (np.ndarray, float): audio data
        annotation (annotations.Events): sound events with start time, end time, label and confidence
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(
            clip_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_path = self.get_path("audio")
        self.annotation_path = self.get_path("annotation")

    @core.cached_property
    def annotation(self):
        """
        The clip's event annotations
        Returns:
            * annotations.Events - sound events with start time, end time, label and confidence
        """
        return load_annotation(self.annotation_path)

    @property
    def audio(self):
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            audio_path=self.audio_path, events=self.annotation
        )


@io.coerce_to_string_io
def load_annotation(fhandle):
    """Load an annotation file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an annotation file

    Returns:
        * annotations.Events - sound events with start time, end time, label and confidence
    """

    df = pd.read_csv(fhandle)
    intervals = df[["onset", "offset"]].values
    label = df["event_label"].tolist()

    annotation_data = annotations.Events(
        intervals=intervals,
        intervals_unit="seconds",
        labels=label,
        labels_unit="open",
        confidence=np.array([np.nan for _ in label]),
    )

    return annotation_data


@io.coerce_to_bytes_io
def load_audio(fhandle):
    """Load a Example audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal at 44.1 kHz
    """
    data, _ = librosa.load(fhandle, sr=44100, mono=False)
    return data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """SINGA:PURA v1.0 dataset"""

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="singapura",
            clip_class=Clip,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_annotation)
    def load_annotation(self, *args, **kwargs):
        return load_annotation(*args, **kwargs)

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "labelled_metadata_public.csv")

        df = pd.read_csv(metadata_path)
        df["filename"] = df["filename"].apply(lambda x: x.replace(".flac", ""))
        df = df.set_index("filename")

        metadata = df.to_dict(orient="index")

        return metadata
