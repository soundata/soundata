"""EigenScape Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown
    
    **EigenScape: a database of spatial acoustic scene recordings**
    
    *Created By:*

        | Marc Ciufo Green, Damian Murphy.
        | Audio Lab, Department of Electronic Engineering, University of York. 
        
    Version 2.0
	
    *Description:*
        EigenScape is a database of acoustic scenes recorded spatially using the mh Acoustics EigenMike. 
        All scenes were recorded in 4th-order Ambisonics
        The database contains recordings of eight different location classes: Beach, Busy Street, Park, Pedestrian Zone, Quiet Street, Shopping Centre, Train Station, Woodland.
        The recordings were made in May 2017 at sites across the North of England. 
	
    *Audio Files Included:*
	* 8 different examples of each location class were recorded over a duration of 10 minutes 
        * 64 recordings in total. 
        * ACN channel ordering with SN3D normalisation at 24-bit / 48 kHz resolution. 
	
    *Annotations Included:*
        * No event labels associated with this dataset
        * The metadata file gives more tempogeographic detail on each recording
        * the EigenScape [recording map](http://bit.ly/EigenSMap) shows the locations and classes of all the recordings.
        * No predefined training, validation, or testing splits. 
	
    *Please Acknowledge EigenScape in Academic Research:*
    If you use this dataset please cite its original publication:

    .. code-block:: latex

        Green MC, Murphy D. EigenScape: A database of spatial acoustic scene recordings. Applied Sciences. 2017 Nov;7(11):1204.
	    
    *License:*
        * Creative Commons Attribution 4.0 International

    *Important:*
        * Use with caution. This loader "Engineers" a solution to obtain the correct files after Park6 and Park8 got mixed-up at the `eigenscape` and `eigenscape_raw` remotes. See the REMOTES and index if you want to understand how this engineered solution works. Also see the discussion about this engineered solution with the dataset author https://github.com/micarraylib/micarraylib/issues/8#issuecomment-1105357329
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import jams
import glob
import numbers
from itertools import cycle

from soundata import download_utils, jams_utils, core, annotations, io

BIBTEX = """
@article{green2017eigenscape,
  title={EigenScape: A database of spatial acoustic scene recordings},
  author={Green, Marc Ciufo and Murphy, Damian},
  journal={Applied Sciences},
  volume={7},
  number={11},
  pages={1204},
  year={2017},
  publisher={Multidisciplinary Digital Publishing Institute}
}
"""

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "2.0": core.Index(
        filename="eigenscape_index_2.0.json",
        url="https://zenodo.org/records/11176800/files/eigenscape_index_2.0.json?download=1",
        checksum="3ea0322ee5e5174a1e265155c9de9be1",
    ),
    "sample": core.Index(filename="eigenscape_index_2.0_sample.json"),
}

REMOTES = {
    "Beach": download_utils.RemoteFileMetadata(
        filename="Beach.zip",
        url="https://zenodo.org/record/1284156/files/Beach.zip?download=1",
        checksum="3dd3920c3a5e56f534760fa2dac86359",
    ),
    "BusyStreet": download_utils.RemoteFileMetadata(
        filename="BusyStreet.zip",
        url="https://zenodo.org/record/1284156/files/BusyStreet.zip?download=1",
        checksum="532b45f5d941d66506c42321a3e062ab",
    ),
    "Park": download_utils.RemoteFileMetadata(
        filename="Park.zip",
        url="https://zenodo.org/record/2628463/files/Park.zip?download=1",
        checksum="c5d638518b4f7d597dd410ee5bb48b67",
    ),
    "PedestrianZone": download_utils.RemoteFileMetadata(
        filename="PedestrianZone.zip",
        url="https://zenodo.org/record/1284156/files/PedestrianZone.zip?download=1",
        checksum="799eb3fccdc628785b3fb69d01e9a7e4",
    ),
    "QuietStreet": download_utils.RemoteFileMetadata(
        filename="QuietStreet.zip",
        url="https://zenodo.org/record/1284156/files/QuietStreet.zip?download=1",
        checksum="f3ead0a54b322886b78ca49c7374a987",
    ),
    "ShoppingCentre": download_utils.RemoteFileMetadata(
        filename="ShoppingCentre.zip",
        url="https://zenodo.org/record/1284156/files/ShoppingCentre.zip?download=1",
        checksum="3f7541ab39d8b00a5898dd3a35412531",
    ),
    "TrainStation": download_utils.RemoteFileMetadata(
        filename="TrainStation.zip",
        url="https://zenodo.org/record/1284156/files/TrainStation.zip?download=1",
        checksum="63fc5406485d5b876ef3805193d63841",
    ),
    "Woodland": download_utils.RemoteFileMetadata(
        filename="Woodland.zip",
        url="https://zenodo.org/record/1284156/files/Woodland.zip?download=1",
        checksum="dadcf83c711ef0cf72f7a4d8585eddad",
    ),
    "Metadata-EigenScape": download_utils.RemoteFileMetadata(
        filename="Metadata-EigenScape.csv",
        url="https://zenodo.org/record/1284156/files/Metadata-EigenScape.csv?download=1",
        checksum="cbed105fb56604c4b763788690089d55",
    ),
}

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Clip(core.Clip):
    """Eigenscape Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        tags (soundata.annotation.Tags): tag (scene label) of the clip + confidence.
        audio_path (str): path to the audio file
        clip_id (str): clip id
        location (str): city were the audio signal was recorded
        time (str): time when the audio signal was recorded
        date (str): date when the audio signal was recorded
        additional information (str): notes included by the dataset
            authors with other details relevant to the specific clip
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
        """The clip's tags

        Returns:
            * annotations.Tags - Tags (scene label) of the clip + confidence.

        """
        scene_label = self._clip_metadata.get("scene_label")
        if scene_label is None:
            return None
        else:
            return annotations.Tags([scene_label], "open", np.array([1.0]))

    @property
    def location(self):
        """The clip's location.

        Returns:
            * str - Tags annotation object
        """
        return self._clip_metadata.get("location")

    @property
    def time(self):
        """The clip's time.

        Returns:
            * str - time when the audio signal was recorded
        """
        return self._clip_metadata.get("time")

    @property
    def date(self):
        """The clip's date.

        Returns:
            * str - date when the audio signal was recorded
        """
        return self._clip_metadata.get("date")

    @property
    def additional_information(self):
        """The clip's additional information

        Returns:
            * str - notes included by the dataset authors with other details relevant to the specific clip
        """
        return self._clip_metadata.get("additional information")

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
    """Load an EigenScape audio file

    Args:
        fhandle (str or file-like): file-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sampling rate of 48000 without resampling.

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file
    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=False)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The EigenScape dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="eigenscape",
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
        metadata_path = os.path.join(self.data_home, "Metadata-EigenScape.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        metadata_index = {}

        with open(metadata_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            next(csv_reader)
            for row in csv_reader:
                file_name = os.path.basename(row[0])
                clip_id = (
                    os.path.basename(file_name).replace(".wav", "").replace("-0", ".")
                )
                scene_label = row[1]
                location = row[2]
                time = row[3]
                date = row[4]
                additional_information = row[5]
                metadata_index[clip_id] = {
                    "scene_label": scene_label,
                    "location": location,
                    "time": time,
                    "date": date,
                    "additional information": additional_information,
                }

        return metadata_index
