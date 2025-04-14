"""EigenScape Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown
    
    **EigenScape: a database of spatial acoustic scene recordings**
    
    *Created By:*

        | Marc Ciufo Green, Damian Murphy.
        | Audio Lab, Department of Electronic Engineering, University of York. 
        
    Version raw
	
    *Description:*
        EigenScape is a database of acoustic scenes recorded spatially using the mh Acoustics EigenMike. 
        All scenes in this format are in Raw format (A-format) with 32 channels
        The database contains recordings of eight different location classes: Beach, Busy Street, Park, Pedestrian Zone, Quiet Street, Shopping Centre, Train Station, Woodland.
        The recordings were made in May 2017 at sites across the North of England. 
	
    *Audio Files Included:*
	* 8 different examples of each location class were recorded over a duration of 10 minutes 
        * 64 recordings in total. 
        * EigenMike channel ordering (32 total) with calibration and PGA level (captured with firewire interface and EigenStudio). 24-bit / 48 kHz resolution. 
	
    *Annotations Included:*
        * No event labels associated with this dataset
        * The metadata file gives more tempogeographic detail on each recording
        * the EigenScape `recording map <http://bit.ly/EigenSMap>`_ shows the locations and classes of all the recordings.
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
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="eigenscape_raw_index_1.0.json",
        url="https://zenodo.org/records/11176807/files/eigenscape_raw_index_1.0.json?download=1",
        checksum="619fa16f7e58aa247b4da43ff5c36a03",
    ),
    "sample": core.Index(filename="eigenscape_raw_index_1.0_sample.json"),
}

REMOTES = {
    "Beach": download_utils.RemoteFileMetadata(
        filename="Beach.zip",
        url="https://webfiles.york.ac.uk/INFODATA/eaeaac50-483e-408f-a391-01b02d4ff9c4/Beach.zip",
        checksum="3f03a527291b5aedffb436fe4f12774b",
    ),
    "BusyStreet": download_utils.RemoteFileMetadata(
        filename="BusyStreet.zip",
        url="https://webfiles.york.ac.uk/INFODATA/eaeaac50-483e-408f-a391-01b02d4ff9c4/BusyStreet.zip",
        checksum="7561ef835b4f43c5f0fa637a9747bc41",
    ),
    "Park_york": download_utils.RemoteFileMetadata(
        filename="Park.zip",
        url="https://webfiles.york.ac.uk/INFODATA/eaeaac50-483e-408f-a391-01b02d4ff9c4/Park.zip",
        checksum="081417d792a31c7cdada5442227c796c",
    ),
    "Park_zenodo": download_utils.RemoteFileMetadata(
        filename="Park.zip",
        url="https://zenodo.org/record/1284156/files/Park.zip?download=1",
        checksum="1268c7d8057529672d3cc17bec8ae302",
    ),
    "PedestrianZone": download_utils.RemoteFileMetadata(
        filename="PedestrianZone.zip",
        url="https://webfiles.york.ac.uk/INFODATA/eaeaac50-483e-408f-a391-01b02d4ff9c4/PedestrianZone.zip",
        checksum="9b6cf5afefa7779eb69d6764d680801c",
    ),
    "Metadata-EigenScape": download_utils.RemoteFileMetadata(
        filename="Metadata-EigenScape.csv",
        url="https://zenodo.org/record/1284156/files/Metadata-EigenScape.csv?download=1",
        checksum="cbed105fb56604c4b763788690089d55",
    ),
}

wav_md5_dict = {
    "QuietStreet-01-Raw": "4b92a30ca73edb12f9190fde53cadcba",
    "QuietStreet-02-Raw": "15b7742bacadb0417b227de9550af725",
    "QuietStreet-03-Raw": "b4f6dc1cb65f9b2051273e496764219d",
    "QuietStreet-04-Raw": "7c45aaca071c2b9cee3849f73be97998",
    "QuietStreet-05-Raw": "74fc68ce1d552b3c76162809bf343750",
    "QuietStreet-06-Raw": "bb6f09247cf62fecc29349aca10500bd",
    "QuietStreet-07-Raw": "a1026a16ba0f4ebcbae0bff6f78c3c5c",
    "QuietStreet-08-Raw": "69c5376a43357262c950393a5766698d",
    "ShoppingCentre-01-Raw": "c02ca658b0cf85561f8c5f728db5bee9",
    "ShoppingCentre-02-Raw": "951c77f0867a09d18d5d14c52932b5d2",
    "ShoppingCentre-03-Raw": "a3ef52bc54d0c7738be537f3b28be7c6",
    "ShoppingCentre-04-Raw": "bb040c0dce13eb553458f014644061f4",
    "ShoppingCentre-05-Raw": "eaf0b9c5313d5bceb0355ab989d04a03",
    "ShoppingCentre-06-Raw": "7e402f7412a95d295d40fa05cb6096b6",
    "ShoppingCentre-07-Raw": "0c10d361a1e2b1a150f48d869580cc00",
    "ShoppingCentre-08-Raw": "38f58d35833aab2a2bbf62c1388c2fb3",
    "TrainStation-01-Raw": "b7c9ab43daa0f7ba940c44e0b9591e67",
    "TrainStation-02-Raw": "104723e307c61a501d24033fac0ea205",
    "TrainStation-03-Raw": "651a9c536d5ee12f955998ba9fc0d63c",
    "TrainStation-04-Raw": "6326b1ab69a2ff306978741937ce2b7a",
    "TrainStation-05-Raw": "fe137aefe8d769b084bd0ff1949d0400",
    "TrainStation-06-Raw": "60932c5d02c5907550a47924e349f47e",
    "TrainStation-07-Raw": "03a56d202ab8a3851799fe616add38b6",
    "TrainStation-08-Raw": "de418fdd958ed9875630e1fa65417b1e",
    "Woodland-01-Raw": "b451fd7e167268d7308aeee95e632829",
    "Woodland-02-Raw": "cd7f344fc6085c8385f1323c92049a79",
    "Woodland-03-Raw": "78c7661bf21225308183c67e903a229f",
    "Woodland-04-Raw": "91afe627730952d75401277d851a67a3",
    "Woodland-05-Raw": "f22e3a06cef4e9dd81b44b2f418b0ef7",
    "Woodland-06-Raw": "84dec0b311d153a3757ca7582f9d12fb",
    "Woodland-07-Raw": "eef91d59e41d41a7e0e7c035893dcb55",
    "Woodland-08-Raw": "0541c709206b1cb4971969cb6d8dab0a",
}
nrecs = 8
wav_files = ["QuietStreet", "ShoppingCentre", "TrainStation", "Woodland"]

for f in wav_files:
    for i in range(nrecs):
        remote_id = f + "-0" + str(i + 1) + "-Raw"
        REMOTES[remote_id] = download_utils.RemoteFileMetadata(
            filename=f + "-0" + str(i + 1) + "-Raw.wav",
            url="https://webfiles.york.ac.uk/INFODATA/eaeaac50-483e-408f-a391-01b02d4ff9c4/"
            + f
            + "-0"
            + str(i + 1)
            + "-Raw.wav",
            checksum=wav_md5_dict[remote_id],
        )

LICENSE_INFO = """
Creative Commons Attribution 4.0 International
"""


class Clip(core.Clip):
    """Eigenscape Raw Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:

        audio_path (str): path to the audio file
        additional information (str): notes included by the dataset
            authors with other details relevant to the specific clip
        clip_id (str): clip id
        date (str): date when the audio signal was recorded
        location (str): city were the audio signal was recorded
        tags (soundata.annotation.Tags): tag (scene label) of the clip + confidence.
        time (str): time when the audio signal was recorded

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
        """The clip's time (00:00-23:59).

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
        """The clip's additional information.

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
    """Load an EigenScape Raw audio file

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
    """The EigenScape Raw dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="eigenscape_raw",
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
                clip_id = (os.path.basename(file_name).replace(".wav", "")) + "-Raw"
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
