"""ESC-50 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    ESC-50: Dataset for Environmental Sound Classification

    The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.
    The total duration of the dataset is 2.8 hours (2000 x 5 seconds).

    The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories:

    Animals	        Natural soundscapes & water sounds	    Human, non-speech sounds	Interior/domestic sounds	Exterior/urban noises
    Dog	            Rain	                                Crying baby	                Door knock	                Helicopter
    Rooster	        Sea waves	                            Sneezing	                Mouse click             	Chainsaw
    Pig	            Crackling fire	                        Clapping                	Keyboard typing	            Siren
    Cow	            Crickets	                            Breathing	                Door, wood creaks	        Car horn
    Frog	        Chirping birds	                        Coughing	                Can opening             	Engine
    Cat     	    Water drops                     	    Footsteps               	Washing machine	            Train
    Hen     	    Wind                                	Laughing	                Vacuum cleaner	            Church bells
    Insects (flying)Pouring water	                        Brushing teeth	            Clock alarm             	Airplane
    Sheep	        Toilet flush                        	Snoring	                    Clock tick              	Fireworks
    Crow	        Thunderstorm	                        Drinking, sipping	        Glass breaking	            Hand saw
    
    Clips in this dataset have been manually extracted from public field recordings gathered by the Freesound.org project. 
    The dataset has been prearranged into 5 folds for comparable cross-validation, making sure that fragments from the same original source file are contained in a single fold.

    A more thorough description of the dataset is available in the original paper with some supplementary materials on GitHub: 
    
    .. code-block:: latex
        K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015.

    https://github.com/karolpiczak/ESC-50

    Repository content
    audio/*.wav

    2000 audio recordings in WAV format (5 seconds, 44.1 kHz, mono) with the following naming convention:

    {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav

    {FOLD} - index of the cross-validation fold,
    {CLIP_ID} - ID of the original Freesound clip,
    {TAKE} - letter disambiguating between different fragments from the same Freesound clip,
    {TARGET} - class in numeric format [0, 49].
    meta/esc50.csv

    CSV file with the following structure:

    filename	fold	target	category	esc10	src_file	take
    
    The esc10 column indicates if a given file belongs to the ESC-10 subset (10 selected classes, CC BY license).

    https://github.com/karolpiczak/ESC-50/blob/master/meta/esc50-human.xlsx

    Additional data pertaining to the crowdsourcing experiment (human classification accuracy).

"""

import os
from typing import BinaryIO, Optional, Tuple

import librosa
import numpy as np
import csv

from soundata import download_utils, jams_utils, core, annotations, io


BIBTEX = """
@inproceedings{piczak2015dataset,
  title = {{ESC}: {Dataset} for {Environmental Sound Classification}},
  author = {Piczak, Karol J.},
  booktitle = {Proceedings of the 23rd {Annual ACM Conference} on {Multimedia}},
  date = {2015-10-13},
  url = {http://dl.acm.org/citation.cfm?doid=2733373.2806390},
  doi = {10.1145/2733373.2806390},
  location = {{Brisbane, Australia}},
  isbn = {978-1-4503-3459-4},
  publisher = {{ACM Press}},
  pages = {1015--1018}
}
"""
REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="ESC-50-master.zip",
        url="https://github.com/karoldvl/ESC-50/archive/master.zip",
        checksum="70aba3bada37d2674b8f6cd5afd5f065",
        unpack_directories=["ESC-50-master"],
    )
}

LICENSE_INFO = "Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)"


class Clip(core.Clip):
    """ESC-50 Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        tags (soundata.annotation.Tags): tag (label) of the clip + confidence. In ESC-50 every clip has one tag.
        audio_path (str): path to the audio file
        filename (str): clip filename
        fold (int): index of the cross-validation fold the clip belongs to
        target (int): clip class in numeric format
        category (str): clip class in string format, i.e., label
        esc10 (bool): True if the clip belongs to the ESC-10 subset (10 selected classes, CC BY license)
        src_file (str): freesound ID of the original file from which the clip was taken
        take (str): letter disambiguating between different fragments from the same Freesound clip (e.g., "A", "B", etc.)
        clip_id (str): clip id

    """

    def __init__(
        self,
        clip_id,
        data_home,
        dataset_name,
        index,
        metadata,
    ):
        super().__init__(
            clip_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

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
    def filename(self):
        return self._clip_metadata.get("filename")

    @property
    def fold(self):
        return self._clip_metadata.get("fold")

    @property
    def target(self):
        return self._clip_metadata.get("target")

    @property
    def category(self):
        return self._clip_metadata.get("category")

    @property
    def esc10(self):
        return self._clip_metadata.get("esc10")

    @property
    def src_file(self):
        return self._clip_metadata.get("src_file")

    @property
    def take(self):
        return self._clip_metadata.get("take")

    @property
    def tags(self):
        return annotations.Tags(
            [self._clip_metadata.get("category")], "open", np.array([1.0])
        )

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
    """Load an ESC-50 audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default,
            which loads the file using its original sample rate of 44100.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The ESC-50 dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="esc50",
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

        metadata_path = os.path.join(self.data_home, "meta", "esc50.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        with open(metadata_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            raw_data = []
            for line in reader:
                if line[0] != "filename":
                    raw_data.append(line)

        metadata_index = {}
        for line in raw_data:

            clip_id = line[0].replace(".wav", "")

            metadata_index[clip_id] = {
                "filename": line[0],
                "fold": int(line[1]),
                "target": int(line[2]),
                "category": line[3],
                "esc10": True if line[4] == "True" else False,
                "src_file": line[5],
                "take": line[6],
            }

        return metadata_index
