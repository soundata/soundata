"""
SINGA:PURA Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown
    
    **SINGA:PURA (SINGApore: Polyphonic URban Audio) v1.0a**
    
    *Created by:*

        | Kenneth Ooi, Karn N. Watcharasupat, Santi Peksi, Furi Andi Karnapi, Zhen-Ting Ong, Danny Chua, Hui-Wen Leow, Li-Long Kwok, Xin-Lei Ng, Zhen-Ann Loh, Woon-Seng Gan
        | Digital Signal Processing Laboratory, School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore.
        
    *Description:*
        The SINGA:PURA (SINGApore: Polyphonic URban Audio) dataset is a strongly-labelled polyphonic urban sound dataset with spatiotemporal context. The dataset contains 6547 strongly-labelled and 72406 unlabelled recordings from a wireless acoustic sensor network deployed in Singapore to identify and mitigate noise sources in Singapore. The strongly-labelled and unlabelled recordings are disjoint, so there are a total of 78953 unique recordings. The recordings are all 10 seconds in length, and may have 1 or 7 channels, depending on the recording device used to record them. Total duration for the labelled subset provided here is 18.2 hours.
    
        For full details regarding the sensor units used, the recording conditions, and annotation methodology, please refer to our conference paper.
    
    *Annotations:*
        Our label taxonomy is derived from the taxonomy used in the SONYC-UST datasets, but has been adapted to fit the local (Singapore) context while retaining compatibility with the SONYC-UST ontonology. We chose this taxonomy to allow the SINGA:PURA dataset to be used in conjunction with the SONYC-UST datasets when training urban sound tagging models by simply omitting the labels that are absent in the SONYC-UST taxonomy from the recordings in the SINGA:PURA dataset. 
        
        Specifically, our label taxonomy consists of 14 coarse-grained classes and 40 fine-grained classes. Their organisation is as follows:
        
        1. Engine
            1. Small engine
            2. Medium engine
            3. Large engine
            
        2. Machinery impact  
            1. Rock drill
            2. Jackhammer
            3. Hoe ram
            4. Pile driver
            
        3. Non-machinery impact  
            1. Glass breaking (*)
            2. Car crash (*)
            3. Explosion (*)
            
        4. Powered saw  
            1. Chainsaw
            2. Small/medium rotating saw
            3. Large rotating saw
            
        5. Alert signal  
            1. Car horn
            2. Car alarm
            3. Siren
            4. Reverse beeper
            
        6. Music  
            1. Stationary music
            2. Mobile music
            
        7. Human voice  
            1. Talking
            2. Shouting
            3. Large crowd
            4. Amplified speech
            5. Singing (*)
            
        8. Human movement (*)  
            1. Footsteps (*)
            2. Clapping (*)
        
        9. Animal (*) 
            1. Dog barking
            2. Bird chirping (*)
            3. Insect chirping (*)
            
        #. Water (*)
            1. Hose pump (*)
            
        #. Weather (*)
            1. Rain (*)
            2. Thunder (*)
            3. Wind (*)
            
        #. Brake (*)
            1. Friction brake (*)
            2. Exhaust brake (*)
            
        #. Train (*)
            1. Electric train (*)
            
        X. Others (*)
            1. Screeching (*)
            2. Plastic crinkling (*)
            3. Cleaning (*)
            4. Gear (*)
            
        Classes marked with an asterisk (*) are present in the SINGA:PURA taxonomy but not the SONYC taxonomy. The "Ice cream truck" class from the SONYC taxonomy has been excluded from the SINGA:PURA taxonomy because this class does not exist in the local context.

        In addition, note that the label for the coarse-grained class "Others" in the soundata loader is "0", which is different from the label "X" that is used in the full version of the SINGA:PURA dataset.

    *This dataset is also accessible via:*
        - Zenodo (labelled subset only): https://zenodo.org/record/5645825
        - DR-NTU (all): https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/Y8UQ6F
        
    *Please Acknowledge SINGA:PURA in Academic Research:*
    If you use this dataset please cite its original publication:
        
        .. code-block:: latex
        
            K. Ooi, K. N. Watcharasupat, S. Peksi, F. A. Karnapi, Z.-T. Ong, D. Chua, H.-W. Leow, L.-L. Kwok, X.-L. Ng, Z.-A. Loh, W.-S. Gan, "A Strongly-Labelled Polyphonic Dataset of Urban Sounds with Spatiotemporal Context," in Proceedings of the 13th Asia Pacific Signal and Information Processing Association Annual Summit and Conference, 2021.
        
    *License:*
        Creative Commons Attribution-ShareAlike 4.0 International.

"""

import os
from typing import Dict, List, Optional, TextIO, Union

import librosa
import numpy as np
import pandas as pd
from soundata import annotations, core, download_utils, io, jams_utils

BIBTEX = """
@inproceedings{ooi2021singapura,
    author    = "K. Ooi and K. N. Watcharasupat and S. Peksi and F. A. Karnapi and Z.-T. Ong and D. Chua and H.-W. Leow and L.-L. Kwok and X.-L. Ng and Z.-A. Loh and W.-S. Gan",
    title     = "A Strongly-Labelled Polyphonic Dataset of Urban Sounds with Spatiotemporal Context",
    booktitle = "Proceedings of the 13th Asia Pacific Signal and Information Processing Association Annual Summit and Conference",
    location  = "Tokyo, Japan"
    year      = 2021
}
"""

INDEXES = {
    "default": "1.0a",
    "test": "sample",
    "1.0a": core.Index(
        filename="singapura_index_1.0a.json",
        url="https://zenodo.org/records/11176844/files/singapura_index_1.0a.json?download=1",
        checksum="404d2057835cc97ef4dcef1b78e1a946",
    ),
    "sample": core.Index(filename="singapura_index_1.0a_sample.json"),
}

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
        checksum=m,
        destination_dir=None,
    )
    for k, f, m in meta_files
}

# put as list for multipart zip
audio_remotes = {
    "audio": [
        download_utils.RemoteFileMetadata(
            filename=f,
            url=f"https://zenodo.org/record/5645825/files/{f}?download=1",
            checksum=m,
            destination_dir=None,
        )
        for f, m in audio_files
    ]
}


RemoteDictType = Dict[
    str,
    Union[List[download_utils.RemoteFileMetadata], download_utils.RemoteFileMetadata],
]

REMOTES: RemoteDictType = {
    **audio_remotes,
    **meta_remotes,
}

DOWNLOAD_INFO = """
SINGA:PURA (SINGApore: Polyphonic URban Audio) v1.0a

Labelled data subset downloaded from https://zenodo.org/record/5645825.
"""

LICENSE_INFO = "Creative Commons Attribution-ShareAlike 4.0 International"


class Clip(core.Clip):
    """
    Args:
        clip_id (str): clip id of the clip

    Attributes:
        clip_id (str): clip id
        audio (np.ndarray, float): audio data
        audio_path (str): path to the audio file
        events (annotations.MultiAnnotator): sound events with start time, end time, label and confidence
        annotation_path (str): path to the annotation file
        sensor_id (str): sensor_id of the device used to record the data
        town (str): town in Singapore where the sensor is located
        timestamp (np.datetime): timestamp of the recording
        dotw (int): day of the week when the clip was recorded, starting from 0 for Sunday
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
    def events(self) -> Optional[annotations.MultiAnnotator]:
        """
        The clip's event annotations

        Returns:
            * annotations.MultiAnnotator - sound events with start time, end time, label and confidence
        """
        return load_annotation(self.annotation_path)

    @property
    def audio(self):
        """
        The clip's audio

        Returns:
            * np.ndarray - audio signal
        """
        return load_audio(self.audio_path)

    @property
    def sensor_id(self) -> str:
        """
        The clip's sensor ID

        Returns:
            * str - sensor_id of the device used to record the data
        """
        return self._clip_metadata["sensor_id"]

    @property
    def town(self) -> str:
        """
        The clip's location

        Returns:
            * str - location of the sensor, one of {'East 1', 'East 2', 'West 1', 'West 2'}
        """
        return self._clip_metadata["town"]

    @property
    def timestamp(self) -> np.datetime64:
        """
        The clip's timestamp

        Returns:
            * np.datetime64 - timestamp of the clip
        """

        return np.datetime64(
            f"{self._clip_metadata['year']}-{self._clip_metadata['month']:02d}-{self._clip_metadata['date']:02d}"
            + f"T{self._clip_metadata['hour']:02d}:{self._clip_metadata['minute']:02d}:{self._clip_metadata['second']:02d}"
        )

    @property
    def dotw(self) -> int:
        """
        The clip's day of the week

        Returns:
            * int - day of the week when the clip was recorded, starting from 0 for Sunday
        """
        return self._clip_metadata["day"]

    def to_jams(self):
        """
        Jams: the clip's data in jams format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, events=self.events, metadata=self._clip_metadata
        )


@io.coerce_to_string_io
def load_annotation(fhandle: TextIO) -> annotations.MultiAnnotator:
    """
    Load an annotation file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an annotation file

    Returns:
        * annotations.MultiAnnotator - sound events with start time, end time, label and confidence
    """

    df = pd.read_csv(fhandle)

    annotators = []
    annotations_ = []

    for id, dfa in df.groupby("annotator"):
        intervals = dfa[["onset", "offset"]].values
        label = dfa["event_label"].tolist()

        events = annotations.Events(
            intervals=intervals,
            intervals_unit="seconds",
            labels=label,
            labels_unit="open",
            confidence=np.ones((len(label),)),
        )

        annotators.append(f"{id:02d}")
        annotations_.append(events)

    return annotations.MultiAnnotator(annotators=annotators, annotations=annotations_)


@io.coerce_to_bytes_io
def load_audio(fhandle):
    """
    Load a Example audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal at 44.1 kHz
    """
    data, _ = librosa.load(fhandle, sr=44100, mono=False)
    return data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    SINGA:PURA v1.0 dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="singapura",
            clip_class=Clip,
            bibtex=BIBTEX,
            indexes=INDEXES,
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
