"""FSD50K Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **FSD50K: an Open Dataset of Human-Labeled Sound Events**

    *Created By:*

        | Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra.
        | Music Technology Group, Universitat Pompeu Fabra (Barcelona). 
        
    Version 1.0

    *Description:*
        FSD50K is an open dataset of human-labeled sound events containing 51,197 Freesound clips unequally
        distributed in 200 classes drawn from the AudioSet Ontology. FSD50K has been created at the Music Technology Group
        of Universitat Pompeu Fabra.

    *Audio Files Included:*
        * FSD50K contains 51,197 audio clips from Freesound, totalling 108.3 hours of multi-labeled audio.
        * The audio content is composed mainly of sound events produced by physical sound sources and production mechanisms,
          including human sounds, sounds of things, animals, natural sounds, musical instruments and more. The vocabulary
          can be inspected in vocabulary.csv.
        * Clips are of variable length from 0.3 to 30s, due to the diversity of the sound classes and the preferences of
          Freesound users when recording sounds.
        * All clips are provided as uncompressed PCM 16 bit 44.1 kHz mono audio files.

    *Annotations Included:*
        * The dataset encompasses 200 sound classes (144 leaf nodes and 56 intermediate nodes) hierarchically organized
          with a subset of the AudioSet Ontology. Please refer to the included vocabulary.csv file for a complete list of
          considered classes.
        * The acoustic material has been manually labeled by humans following a data labeling
          process using the Freesound Annotator platform.
        * Ground truth labels are provided at the clip-level (i.e., weak labels).
        * Note: All classes in FSD50K are represented in AudioSet, except Crash cymbal, Human group actions, Human voice,
          Respiratory sounds, and Domestic sounds, home sounds.
        * Note: We use a slightly different format than AudioSet for the naming of class labels in order to avoid potential
          problems with spaces, commas, etc. Example: we use Accelerating_and_revving_and_vroom instead of the original
          Accelerating, revving, vroom. You can go back to the original AudioSet naming using the information provided in
          vocabulary.csv (class label and mid for the 200 classes of FSD50K) and the AudioSet Ontology specification.


    *Organization:*
        FSD50K is split in two subsets: the developement (dev) and the evaluation (eval) sets.
        Especifications of both subsets is detailed below:

    * Dev set:
        * 40,966 audio clips totalling 80.4 hours of audio
        * Avg duration/clip: 7.1s
        * 114,271 smeared labels (i.e., labels propagated in the upwards direction to the root of the ontology)
        * Labels are correct but could be occasionally incomplete
        * A train/validation split is provided. If a different split is used, it should be specified for reproducibility
          and fair comparability of results

    * Eval set:
        * 10,231 audio clips totalling 27.9 hours of audio
        * Avg duration/clip: 9.8s
        * 38,596 smeared labels
        * Eval set is labeled exhaustively (labels are correct and complete for the considered vocabulary)

    *Ground-truth Files Included:*
        FSD50K ground-truth is represented through the following file structure:

    * dev.csv:
        Each row (i.e. audio clip) of dev.csv contains the following information:

        * fname:
            The file name without the .wav extension, e.g., the fname 64760 corresponds to the file 64760.wav
            in disk. This number is the Freesound id. We always use Freesound ids as filenames.

        * labels:
            The class labels (i.e., the ground truth). Note these class labels are smeared, i.e., the labels
            have been propagated in the upwards direction to the root of the ontology. More details about the label
            smearing process can be found in Appendix D of our paper.

        * mids:
            The Freebase identifiers corresponding to the class labels, as defined in the AudioSet Ontology
            specification.

        * split:
            Whether the clip belongs to train or val (see paper for details on the proposed split)


    * eval.csv:
        Rows in eval.csv follow the same format as dev.csv, except that there is no split column.

    *Metadata Files Included:*
        To allow a variety of analysis and approaches with FSD50K, we provide the following metadata:

    * class_info_FSD50K.json:
        Python dictionary where each entry corresponds to one sound class and  contains: FAQs
        utilized during the annotation of the class, examples (representative audio clips), and verification_examples
        (audio clips presented to raters during annotation as a quality control mechanism). Audio clips are described by
        the Freesound id. Note: It may be that some of these examples are not included in the FSD50K release.


    * dev_clips_info_FSD50K.json:
        Python dictionary where each entry corresponds to one dev clip and contains: title,
        description, tags, clip license, and the uploader name. All these metadata are provided by the uploader.


    * eval_clips_info_FSD50K.json:
        Same as above, but with eval clips.


    * pp_pnp_ratings.json:
        Python dictionary where each entry corresponds to one clip in the dataset and contains
        the PP/PNP ratings for the labels associated with the clip. More specifically, these ratings are gathered for the
        labels validated in the validation task. This file includes 59,485 labels for the 51,197 clips in FSD50K.
        Out of these labels:

            * 56,095 labels have inter-annotator agreement (PP twice, or PNP twice). Each of these combinations can be
              occasionally accompanied by other (non-positive) ratings.

            * 3390 labels feature other rating configurations such as i) only one PP rating and one PNP rating (and nothing
              else). This can be considered inter-annotator agreement at the "Present" level; ii) only one PP rating (and
              nothing else); iii) only one PNP rating (and nothing else).

        Ratings' legend: PP=1; PNP=0.5; U=0; NP=-1.

        Note: The PP/PNP ratings have been provided in the validation task. Subsequently, a subset of these clips
        corresponding to the eval set was exhaustively labeled in the refinement task, hence receiving additional labels
        in many cases. For these eval clips, you might want to check their labels in eval.csv in order to have more info
        about their audio content.


    *collection folder:*
        This folder contains metadata for what we call the sound collection format. This format consists of
        the raw annotations gathered, featuring all generated class labels without any restriction.
        We provide the collection format to make available some annotations that do not appear in the FSD50K ground truth
        release. This typically happens in the case of classes for which we gathered human-provided annotations, but that
        were discarded in the FSD50K release due to data scarcity (more specifically, they were merged with their parents).
        In other words, the main purpose of the collection format is to make available annotations for tiny classes.
        The format of these files in analogous to that of the files in FSD50K.ground_truth/. A couple of examples show the
        differences between collection and ground truth formats:

        * clip:  labels_in_collection - labels_in_ground_truth

            * 51690:  Owl - Bird,Wild_Animal,Animal

            * 190579:  Toothbrush,Electric_toothbrush - Domestic_sounds_and_home_sounds

        In the first example, raters provided the label Owl. However, due to data scarcity, Owl labels were merged into
        their parent Bird. Then, labels Wild_Animal,Animal were added via label propagation (smearing). The second example
        shows one of the most extreme cases, where raters provided the labels Electric_toothbrush,Toothbrush, which both
        had few data. Hence, they were merged into Toothbrush's parent, which unfortunately is Domestic_sounds_and_home_sounds
        (a rather vague class containing a variety of children sound classes).

        Note: Labels in the collection format are not smeared.

        Note: While in FSD50K's ground truth the vocabulary encompasses 200 classes (common for dev and eval), since the
        collection format is composed of raw annotations, the vocabulary here is much larger (over 350 classes), and it is
        slightly different in dev and eval.

    *Please Acknowledge FSD50K in Academic Research:*
        If you use the FSD50K Dataset please cite the following paper:

        .. code-block:: latex
        
            Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font, Xavier Serra. "FSD50K: an Open Dataset of Human-Labeled Sound Events", arXiv:2010.00475, 2020.

        The authors would like to thank everyone who contributed to FSD50K with annotations, and especially Mercedes
        Collado, Ceren Can, Rachit Gupta, Javier Arredondo, Gary Avendano and Sara Fernandez for their commitment and
        perseverance. The authors would also like to thank Daniel P.W. Ellis and Manoj Plakal from Google Research for
        valuable discussions. This work is partially supported by the European Union’s Horizon 2020 research and innovation
        programme under grant agreement No 688382 AudioCommons, and two Google Faculty Research Awards 2017 and 2018, and
        the Maria de Maeztu Units of Excellence Programme (MDM-2015-0502).


    *License:*
        All audio clips in FSD50K are released under Creative Commons (CC) licenses. Each clip has its own license as
        defined by the clip uploader in Freesound, some of them requiring attribution to their original authors and some
        forbidding further commercial reuse. For attribution purposes and to facilitate attribution of these files to third
        parties, we include a mapping from the audio clips to their corresponding licenses. The licenses are specified in
        the files dev_clips_info_FSD50K.json and eval_clips_info_FSD50K.json. These licenses are CC0, CC-BY, CC-BY-NC and
        CC Sampling+.

        In addition, FSD50K as a whole is the result of a curation process and it has an additional license: FSD50K is
        released under CC-BY. This license is specified in the LICENSE-DATASET file downloaded with the FSD50K.doc zip file.

        Usage of FSD50K for commercial purposes: If you'd like to use FSD50K for commercial purposes, please contact Eduardo
        Fonseca and Frederic Font at eduardo.fonseca@upf.edu and frederic.font@upf.edu.


    *Feedback:*
        For further questions, please contact eduardo.fonseca@upf.edu, or join the freesound-annotator Google Group.

"""

import os
from typing import BinaryIO, Optional, Tuple

import librosa
import csv
import json
import logging
import subprocess
import numpy as np

from soundata import download_utils, jams_utils, core, annotations, io


BIBTEX = """
@dataset{fonseca2020fsd50k,
    title={FSD50K: an Open Dataset of Human-Labeled Sound Events}, 
    author={Eduardo Fonseca and Xavier Favory and Jordi Pons and Frederic Font and Xavier Serra},
    year={2020},
    eprint={2010.00475},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="fsd50k_index_1.0.json",
        url="https://zenodo.org/records/11176815/files/fsd50k_index_1.0.json?download=1",
        checksum="3317c25426cb3f539eea2b94651c14ba",
    ),
    "sample": core.Index(filename="fsd50k_index_1.0_sample.json"),
}
# a dictionary key that has a list of RemoteFileMetadata implies a multi-part zip
# and will be processed as such using the zip subprocess (see soundata.download_utils)
REMOTES = {
    "FSD50K.dev_audio": [
        download_utils.RemoteFileMetadata(
            filename="FSD50K.dev_audio.zip",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip?download=1",
            checksum="c480d119b8f7a7e32fdb58f3ea4d6c5a",
        ),
        download_utils.RemoteFileMetadata(
            filename="FSD50K.dev_audio.z01",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01?download=1",
            checksum="faa7cf4cc076fc34a44a479a5ed862a3",
        ),
        download_utils.RemoteFileMetadata(
            filename="FSD50K.dev_audio.z02",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02?download=1",
            checksum="8f9b66153e68571164fb1315d00bc7bc",
        ),
        download_utils.RemoteFileMetadata(
            filename="FSD50K.dev_audio.z03",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03?download=1",
            checksum="1196ef47d267a993d30fa98af54b7159",
        ),
        download_utils.RemoteFileMetadata(
            filename="FSD50K.dev_audio.z04",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04?download=1",
            checksum="d088ac4e11ba53daf9f7574c11cccac9",
        ),
        download_utils.RemoteFileMetadata(
            filename="FSD50K.dev_audio.z05",
            url="https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05?download=1",
            checksum="81356521aa159accd3c35de22da28c7f",
        ),
    ],
    "FSD50K.eval_audio": [
        download_utils.RemoteFileMetadata(
            filename="FSD50K.eval_audio.zip",
            url="https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip?download=1",
            checksum="6fa47636c3a3ad5c7dfeba99f2637982",
        ),
        download_utils.RemoteFileMetadata(
            filename="FSD50K.eval_audio.z01",
            url="https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01?download=1",
            checksum="3090670eaeecc013ca1ff84fe4442aeb",
        ),
    ],
    "ground_truth": download_utils.RemoteFileMetadata(
        filename="FSD50K.ground_truth.zip",
        url="https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip?download=1",
        checksum="ca27382c195e37d2269c4c866dd73485",
    ),
    "metadata": download_utils.RemoteFileMetadata(
        filename="FSD50K.metadata.zip",
        url="https://zenodo.org/record/4060432/files/FSD50K.metadata.zip?download=1",
        checksum="b9ea0c829a411c1d42adb9da539ed237",
    ),
    "documentation": download_utils.RemoteFileMetadata(
        filename="FSD50K.doc.zip",
        url="https://zenodo.org/record/4060432/files/FSD50K.doc.zip?download=1",
        checksum="3516162b82dc2945d3e7feba0904e800",
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Clip(core.Clip):
    """FSD50K Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        clip_id (str): clip id
        description (str): description of the sound provided by the Freesound uploader
        mids (soundata.annotations.Tags): tag (labels) encoded in Audioset formatting
        pp_pnp_ratings (dict): PP/PNP ratings given to the main label of the clip
        split (str): flag to identify if clip belongs to developement, evaluation or validation splits
        tags (soundata.annotations.Tags): tag (label) of the clip + confidence
        title (str): the title of the uploaded file in Freesound
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio.

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
            self._clip_metadata["ground_truth"].get("tags"),
            "open",
            np.array([1.0] * len(self._clip_metadata["ground_truth"].get("tags"))),
        )

    @property
    def mids(self):
        """The clip's mids.

        Returns:
            * annotations.Tags - tag (labels) encoded in Audioset formatting

        """
        return annotations.Tags(
            self._clip_metadata["ground_truth"].get("mids"),
            "open",
            np.array([1.0] * len(self._clip_metadata["ground_truth"].get("tags"))),
        )

    @property
    def split(self):
        """The clip's split.

        Returns:
            * str - flag to identify if clip belongs to developement, evaluation or validation splits

        """
        return self._clip_metadata["ground_truth"].get("split")

    @property
    def title(self):
        """The clip's title.

        Returns:
            * str - the title of the uploaded file in Freesound

        """
        return self._clip_metadata["clip_info"].get("title")

    @property
    def description(self):
        """The clip's description.

        Returns:
            * str - description of the sound provided by the Freesound uploader

        """
        return self._clip_metadata["clip_info"].get("description")

    @property
    def pp_pnp_ratings(self):
        """The clip's PP/PNP ratings.

        Returns:
            * dict - PP/PNP ratings given to the main label of the clip
        """
        return self._clip_metadata.get("pp_pnp_ratings")

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            tags=self.tags,
            metadata={
                "split": self._clip_metadata["ground_truth"].get("split"),
                "mids": self._clip_metadata["ground_truth"].get("mids"),
                "pp_pnp_ratings": self._clip_metadata.get("pp_pnp_ratings"),
                "title": self._clip_metadata["clip_info"].get("title"),
                "description": self._clip_metadata["clip_info"].get("description"),
                "freesound_tags": self._clip_metadata["clip_info"].get("tags"),
                "license": self._clip_metadata["clip_info"].get("license"),
                "uploader": self._clip_metadata["clip_info"].get("uploader"),
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a FSD50K audio file

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


def load_ground_truth(data_path):
    """Load ground truth files of FSD50K

    Args:
        data_path (str): Path to the ground truth file

    Returns:
        * ground_truth_dict (dict): ground truth dict of the clips in the input split
        * clip_ids (list): list of clip ids of the input split
    """
    ground_truth_dict = {}
    clip_ids = []
    with open(data_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        next(reader)
        for line in reader:
            if len(line) == 3:
                if "collection" not in data_path:
                    ground_truth_dict[line[0]] = {
                        "tags": (
                            list(line[1].split(","))
                            if "," in line[1]
                            else list([line[1]])
                        ),
                        "mids": (
                            list(line[2].split(","))
                            if "," in line[2]
                            else list([line[2]])
                        ),
                        "split": "test",
                    }
                else:
                    ground_truth_dict[line[0]] = {
                        "tags": (
                            list(line[1].split(","))
                            if "," in line[1]
                            else list([line[1]])
                        ),
                        "mids": (
                            list(line[2].split(","))
                            if "," in line[2]
                            else list([line[2]])
                        ),
                    }
                clip_ids.append(line[0])
            if len(line) == 4:
                ground_truth_dict[line[0]] = {
                    "tags": (
                        list(line[1].split(",")) if "," in line[1] else list([line[1]])
                    ),
                    "mids": (
                        list(line[2].split(",")) if "," in line[2] else list([line[2]])
                    ),
                    "split": "train" if line[3] == "train" else "validation",
                }
                clip_ids.append(line[0])

    return ground_truth_dict, clip_ids


def load_fsd50k_vocabulary(data_path):
    """Load vocabulary of FSD50K to relate FSD50K labels with AudioSet onthology

    Args:
        data_path (str): Path to the vocabulary file

    Returns:
        * fsd50k_to_audioset (dict): vocabulary to convert FSD50K to AudioSet
        * audioset_to_fsd50k (dict): vocabulary to convert from AudioSet to FSD50K
    """
    fsd50k_to_audioset = {}
    audioset_to_fsd50k = {}
    with open(data_path, "r") as fhandle:
        reader = csv.reader(fhandle, delimiter=",")
        for line in reader:
            fsd50k_to_audioset[line[1]] = line[2]
            audioset_to_fsd50k[line[2]] = line[1]

    return fsd50k_to_audioset, audioset_to_fsd50k


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The FSD50K dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="fsd50k",
            clip_class=Clip,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

        # Ground_truth paths
        self.ground_truth_dev_path = os.path.join(
            self.data_home, "FSD50K.ground_truth", "dev.csv"
        )
        self.ground_truth_eval_path = os.path.join(
            self.data_home, "FSD50K.ground_truth", "eval.csv"
        )

        # Sound collection format labels paths
        self.collection_dev_path = os.path.join(
            self.data_home, "FSD50K.metadata", "collection", "collection_dev.csv"
        )
        self.collection_eval_path = os.path.join(
            self.data_home, "FSD50K.metadata", "collection", "collection_eval.csv"
        )

        # Clip metadata paths
        self.clips_info_dev_path = os.path.join(
            self.data_home, "FSD50K.metadata", "dev_clips_info_FSD50K.json"
        )
        self.clips_info_eval_path = os.path.join(
            self.data_home, "FSD50K.metadata", "eval_clips_info_FSD50K.json"
        )

        # Class info path
        self.label_info_path = os.path.join(
            self.data_home, "FSD50K.metadata", "class_info_FSD50K.json"
        )

        # PP/PNP ratings path
        self.pp_pnp_ratings_path = os.path.join(
            self.data_home, "FSD50K.metadata", "pp_pnp_ratings_FSD50K.json"
        )

        # Vocabulary paths
        self.vocabulary_path = os.path.join(
            self.data_home, "FSD50K.ground_truth", "vocabulary.csv"
        )
        self.collection_vocabulary_dev_path = os.path.join(
            self.data_home,
            "FSD50K.metadata",
            "collection",
            "vocabulary_collection_dev.csv",
        )
        self.collection_vocabulary_eval_path = (
            self.collection_vocabulary_dev_path.replace("_dev", "_eval")
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_ground_truth)
    def load_ground_truth(self, *args, **kwargs):
        return load_ground_truth(*args, **kwargs)

    @core.copy_docs(load_fsd50k_vocabulary)
    def load_fsd50k_vocabulary(self, *args, **kwargs):
        return load_fsd50k_vocabulary(*args, **kwargs)

    @property
    def fsd50k_to_audioset(self):
        return load_fsd50k_vocabulary(self.vocabulary_path)[0]

    @property
    def audioset_to_fsd50k(self):
        return load_fsd50k_vocabulary(self.vocabulary_path)[1]

    @property
    def label_info(self):
        return (
            json.load(open(self.label_info_path, "r"))
            if os.path.exists(self.label_info_path)
            else None
        )

    @property
    def collection_fsd50k_to_audioset(self):
        collection_fsd50k_to_audioset = {
            "dev": load_fsd50k_vocabulary(self.collection_vocabulary_dev_path)[0],
            "eval": load_fsd50k_vocabulary(self.collection_vocabulary_eval_path)[0],
        }
        return collection_fsd50k_to_audioset

    @property
    def collection_audioset_to_fsd50k(self):
        collection_audioset_to_fsd50k = {
            "dev": load_fsd50k_vocabulary(self.collection_vocabulary_dev_path)[1],
            "eval": load_fsd50k_vocabulary(self.collection_vocabulary_eval_path)[1],
        }
        return collection_audioset_to_fsd50k

    @core.cached_property
    def _metadata(self):
        metadata_index = {}

        ground_truth_dev, clip_ids_dev = load_ground_truth(self.ground_truth_dev_path)
        ground_truth_eval, clip_ids_eval = load_ground_truth(
            self.ground_truth_eval_path
        )

        collection_dev, _ = load_ground_truth(self.collection_dev_path)
        collection_eval, _ = load_ground_truth(self.collection_eval_path)

        clips_info_dev = (
            json.load(open(self.clips_info_dev_path, "r"))
            if os.path.exists(self.clips_info_dev_path)
            else None
        )
        clips_info_eval = (
            json.load(open(self.clips_info_eval_path, "r"))
            if os.path.exists(self.clips_info_eval_path)
            else None
        )

        pp_pnp_ratings = (
            json.load(open(self.pp_pnp_ratings_path, "r"))
            if os.path.exists(self.pp_pnp_ratings_path)
            else None
        )

        for clip_id in self.clip_ids:
            if clip_id in clip_ids_dev:
                metadata_index[clip_id] = {
                    "ground_truth": ground_truth_dev[clip_id],
                    "clip_info": clips_info_dev[clip_id],
                    "pp_pnp_ratings": pp_pnp_ratings[clip_id],
                    "collection_labels": collection_dev[clip_id],
                }
            if clip_id in clip_ids_eval:
                metadata_index[clip_id] = {
                    "ground_truth": ground_truth_eval[clip_id],
                    "clip_info": clips_info_eval[clip_id],
                    "pp_pnp_ratings": pp_pnp_ratings[clip_id],
                    "collection_labels": collection_eval[clip_id],
                }

        return metadata_index
