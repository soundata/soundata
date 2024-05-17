"""FSDnoisy18K Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **FSDnoisy18K**

    *Created By:*

        | Eduardo Fonseca, Mercedes Collado, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, Xavier Serra.
        | Music Technology Group, Universitat Pompeu Fabra (Barcelona). 
    
    Version 1.0

    *Description:*
        FSDnoisy18k is an audio dataset collected with the aim of fostering the investigation of label noise in sound
        event classification. It contains 42.5 hours of audio across 20 sound classes, including a small amount of
        manually-labeled data and a larger quantity of real-world noisy data.

    What follows is a summary of the most basic aspects of FSDnoisy18k. For a complete description of FSDnoisy18k,
    make sure to check:

        * The FSDnoisy18k companion site: http://www.eduardofonseca.net/FSDnoisy18k/
        * The description provided in Section 2 of our ICASSP 2019 paper

    FSDnoisy18k is an audio dataset collected with the aim of fostering the investigation of label noise in sound
    event classification. It contains 42.5 hours of audio across 20 sound classes, including a small amount of
    manually-labeled data and a larger quantity of real-world noisy data.

    The source of audio content is Freesound—a sound sharing site created an maintained by the Music Technology Group
    hosting over 400,000 clips uploaded by its community of users, who additionally provide some basic metadata
    (e.g., tags, and title). The 20 classes of FSDnoisy18k are drawn from the AudioSet Ontology and are selected based
    on data availability as well as on their suitability to allow the study of label noise.
    The 20 classes are: "Acoustic guitar", "Bass guitar", "Clapping", "Coin (dropping)", "Crash cymbal", "Dishes,
    pots, and pans", "Engine", "Fart", "Fire", "Fireworks", "Glass", "Hi-hat", "Piano", "Rain", "Slam", "Squeak",
    "Tearing", "Walk, footsteps", "Wind", and "Writing". FSDnoisy18k was created with the Freesound Annotator,
    which is a platform for the collaborative creation of open audio datasets.

    We defined a clean portion of the dataset consisting of correct and complete labels. The remaining portion is
    referred to as the noisy portion. Each clip in the dataset has a single ground truth label (singly-labeled data).

    The clean portion of the data consists of audio clips whose labels are rated as present in the clip and
    predominant (almost all with full inter-annotator agreement), meaning that the label is correct and, in most
    cases, there is no additional acoustic material other than the labeled class. A few clips may contain some
    additional sound events, but they occur in the background and do not belong to any of the 20 target classes.
    This is more common for some classes that rarely occur alone, e.g., “Fire”, “Glass”, “Wind” or “Walk, footsteps”.

    The noisy portion of the data consists of audio clips that received no human validation. In this case, they are
    categorized on the basis of the user-provided tags in Freesound. Hence, the noisy portion features a certain
    amount of label noise.
    
    *Included files and statistics:*
        * FSDnoisy18k contains 18,532 audio clips (42.5h) unequally distributed in the 20 aforementioned classes drawn from the AudioSet Ontology.
        * The audio clips are provided as uncompressed PCM 16 bit, 44.1 kHz, mono audio files.
        * The audio clips are of variable length ranging from 300ms to 30s, and each clip has a single ground truth label (singly-labeled data).
        * The dataset is split into a test set and a train set. The test set is drawn entirely from the clean portion, while the remainder of data forms the train set.
        * The train set is composed of 17,585 clips (41.1h) unequally distributed among the 20 classes. It features a clean subset and a noisy subset. In terms of number of clips their proportion is 10%/90%, whereas in terms of duration the proportion is slightly more extreme (6%/94%). The per-class percentage of clean data within the train set is also imbalanced, ranging from 6.1% to 22.4%. The number of audio clips per class ranges from 51 to 170, and from 250 to 1000 in the clean and noisy subsets, respectively. Further, a noisy small subset is defined, which includes an amount of (noisy) data comparable (in terms of duration) to that of the clean subset.
        * The test set is composed of 947 clips (1.4h) that belong to the clean portion of the data. Its class distribution is similar to that of the clean subset of the train set. The number of per-class audio clips in the test set ranges from 30 to 72. The test set enables a multi-class classification problem.
        * FSDnoisy18k is an expandable dataset that features a per-class varying degree of types and amount of label noise. The dataset allows investigation of label noise as well as other approaches, from semi-supervised learning, e.g., self-training to learning with minimal supervision.


    *Additional code:*
        We've released the code for our ICASSP 2019 paper at https://github.com/edufonseca/icassp19. The framework
        comprises all the basic stages: feature extraction, training, inference and evaluation. After loading the
        FSDnoisy18k dataset, log-mel energies are computed and a CNN baseline is trained and evaluated. The code also
        allows to test four noise-robust loss functions. Please check our paper for more details.

    *Label noise characteristics:*
        FSDnoisy18k features real label noise that is representative of audio data retrieved from the web,
        particularly from Freesound. The analysis of a per-class, random, 15% of the noisy portion of FSDnoisy18k
        revealed that roughly 40% of the analyzed labels are correct and complete, whereas 60% of the labels show
        some type of label noise. Please check the FSDnoisy18k companion site for a detailed characterization of
        the label noise in the dataset, including a taxonomy of label noise for singly-labeled data as well as a
        per-class description of the label noise.
        
    *Relevant links:*
        * Source code for our preprint: https://github.com/edufonseca/icassp19
        * Freesound Annotator: https://annotator.freesound.org/
        * Freesound: https://freesound.org
        * Eduardo Fonseca’s personal website: http://www.eduardofonseca.net/


    *Please Acknowledge FSDnoisy18K in Academic Research:*
        If you use the FSDnoisy18K Dataset please cite the following paper:

        .. code-block:: latex
        
            Eduardo Fonseca, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, and Xavier Serra, “Learning Sound Event Classifiers from Web Audio with Noisy Labels”, arXiv preprint arXiv:1901.01189, 2019
            
        This work is partially supported by the European Union’s Horizon 2020 research and innovation programme
        under grant agreement No 688382 AudioCommons. Eduardo Fonseca is also sponsored by a Google Faculty Research
        Award 2017. We thank everyone who contributed to FSDnoisy18k with annotations.


    *License:*
        FSDnoisy18k has licenses at two different levels, as explained next. All sounds in Freesound are released
        under Creative Commons (CC) licenses, and each audio clip has its own license as defined by the audio clip
        uploader in Freesound. In particular, all Freesound clips included in FSDnoisy18k are released under either
        CC-BY or CC0. For attribution purposes and to facilitate attribution of these files to third parties, we
        include a relation of audio clips and their corresponding license in the LICENSE-INDIVIDUAL-CLIPS file
        downloaded with the dataset.

        In addition, FSDnoisy18k as a whole is the result of a curation process and it has an additional license.
        FSDnoisy18k is released under CC-BY. This license is specified in the LICENSE-DATASET file downloaded with
        the dataset.

    *Feedback:*
        For further questions, please contact eduardo.fonseca@upf.edu, or join the freesound-annotator Google Group.

"""

import os
from typing import BinaryIO, Optional, Tuple

import librosa
import csv
import numpy as np

from soundata import download_utils, jams_utils, core, annotations, io


BIBTEX = """
@misc{fonseca2019learning,
      title={Learning Sound Event Classifiers from Web Audio with Noisy Labels},
      author={Eduardo Fonseca and Manoj Plakal and Daniel P. W. Ellis and Frederic Font and Xavier Favory and Xavier Serra},
      year={2019},
      eprint={1901.01189},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="fsdnoisy18k_index_1.0.json",
        url="https://zenodo.org/records/11176823/files/fsdnoisy18k_index_1.0.json?download=1",
        checksum="09b7c6156156b9ccef2200c37c9b2791",
    ),
    "sample": core.Index(filename="fsdnoisy18k_index_1.0_sample.json"),
}

REMOTES = {
    "audio_train": download_utils.RemoteFileMetadata(
        filename="FSDnoisy18k.audio_train.zip",
        url="https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip?download=1",
        checksum="34dc1d34ca44622af5bf439ceb6f0d55",
    ),
    "audio_test": download_utils.RemoteFileMetadata(
        filename="FSDnoisy18k.audio_test.zip",
        url="https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_test.zip?download=1",
        checksum="1ac73d70b4ef3f81900d98c261a832de",
    ),
    "docs": download_utils.RemoteFileMetadata(
        filename="FSDnoisy18k.doc.zip",
        url="https://zenodo.org/record/2529934/files/FSDnoisy18k.doc.zip?download=1",
        checksum="093a1ca185ec341ca4eac14215e7f96b",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="FSDnoisy18k.meta.zip",
        url="https://zenodo.org/record/2529934/files/FSDnoisy18k.meta.zip?download=1",
        checksum="96e27a4a63b7a2870522ddcedb5d8296",
    ),
}

LICENSE_INFO = """
Please note that FSDnoisy18k has licenses at two different levels. All sounds in Freesound are released
under Creative Commons (CC) licenses, and each audio clip has its own license as defined by the audio clip
uploader in Freesound. In particular, all Freesound clips included in FSDnoisy18k are released under either
CC-BY or CC0. For attribution purposes and to facilitate attribution of these files to third parties, we
include a relation of audio clips and their corresponding license in the LICENSE-INDIVIDUAL-CLIPS file
downloaded with the dataset.
"""


class Clip(core.Clip):
    """FSDnoisy18K Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        aso_id (str): the id of the corresponding category as per the AudioSet Ontology
        audio_path (str): path to the audio file
        clip_id (str): clip id
        manually_verified (int): flag to indicate whether the clip belongs to the clean portion (1), or to the noisy portion (0) of the train set
        noisy_small (int): flag to indicate whether the clip belongs to the noisy_small portion (1) of the train set
        split (str): flag to indicate whether the clip belongs the train or test split
        tag (soundata.annotations.Tags): tag (label) of the clip + confidence
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
        """The clip's tags.

        Returns:
            * annotations.Tags - tag (label) of the clip + confidence

        """
        return annotations.Tags(
            [self._clip_metadata.get("tag")], "open", np.array([1.0])
        )

    @property
    def aso_id(self):
        """The clip's Audioset ontology ID.

        Returns:
            * str - the id of the corresponding category as per the AudioSet Ontology

        """
        return self._clip_metadata.get("aso_id")

    @property
    def manually_verified(self):
        """The clip's manually annotated flag.

        Returns:
            * int - flag to indicate whether the clip belongs to the clean portion (1), or to the noisy portion (0) of the train set

        """
        return self._clip_metadata.get("manually_verified")

    @property
    def noisy_small(self):
        """The clip's noisy flag.

        Returns:
            * int - flag to indicate whether the clip belongs to the noisy_small portion (1) of the train set

        """
        return self._clip_metadata.get("noisy_small")

    @property
    def split(self):
        """The clip's split.

        Returns:
            * str - flag to indicate whether the clip belongs the train or test split

        """
        return self._clip_metadata.get("split")

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
    """Load a FSDnoisy18K audio file.

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
    The FSDnoisy18K dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="fsdnoisy18k",
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
        metadata_train_path = os.path.join(
            self.data_home, "FSDnoisy18k.meta", "train.csv"
        )
        metadata_test_path = os.path.join(
            self.data_home, "FSDnoisy18k.meta", "test.csv"
        )

        if not os.path.exists(metadata_train_path):
            raise FileNotFoundError(
                "Train metadata not found. Did you run .download()?"
            )
        if not os.path.exists(metadata_test_path):
            raise FileNotFoundError("Test metadata not found. Did you run .download()?")

        metadata_index = {}

        with open(metadata_train_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            for row in reader:
                metadata_index[row[0].replace(".wav", "")] = {
                    "split": "train",
                    "tag": row[1],
                    "aso_id": str(row[2]),
                    "manually_verified": int(row[3]),
                    "noisy_small": int(row[4]),
                }

        with open(metadata_test_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            for row in reader:
                metadata_index[row[0].replace(".wav", "")] = {
                    "split": "test",
                    "tag": row[1],
                    "aso_id": str(row[2]),
                }

        return metadata_index
