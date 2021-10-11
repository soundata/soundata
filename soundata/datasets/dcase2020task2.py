"""DCASE Challenge 2020 Task 2 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **DCASE Challenge 2020 Task 2, Development, Additional training and
    Evaluation datasets**

    **Description**

    This dataset is the *evaluation dataset* for the DCASE 2020 Challenge
    Task 2 `"Unsupervised Detection of Anomalous Sounds for Machine Condition
    Monitoring" <http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds>`_

    In the task, three datasets have been released: "development dataset",
    "additional training dataset", and "evaluation dataset". This evaluation
    dataset was the last of the three released. This dataset includes around
    400 samples for each Machine Type and Machine ID used in the evaluation
    dataset, none of which have a condition label (i.e., normal or anomaly).

    The recording procedure and data format are the same as the development
    dataset and additional training dataset. The Machine IDs in this dataset
    are the same as those in the additional training dataset. For more
    information, please see the pages of the development dataset and the
    task description. 

    The data used for this task comprises parts of **ToyADMOS** and the 
    **MIMII Dataset** consisting of the normal/anomalous operating sounds of
    six types of toy/real machines. Each recording is a single-channel
    (approximately) 10-sec length audio that includes both a target machine's
    operating sound and environmental noise. The following six types of
    toy/real machines are used in this task:

    * Toy-car (ToyADMOS)
    * Toy-conveyor (ToyADMOS)
    * Valve (MIMII Dataset)
    * Pump (MIMII Dataset)
    * Fan (MIMII Dataset)
    * Slide rail (MIMII Dataset)


    **Recording procedure**

    The ToyADMOS consists of normal/anomalous operating sounds of miniature
    machines (toys) collected with four microphones, and the MIMII dataset
    consists of those of real-machines collected with eight microphones.
    Anomalous sounds in these datasets were collected by deliberately damaging 
    target machines. For simplifying the task, we used only the first channel
    of multi-channel recordings; all recordings are regarded as single-channel
    recordings of a fixed microphone. The sampling rate of all signals has been
    downsampled to 16 kHz. From ToyADMOS, we used only IND-type data that
    contain the operating sounds of the entire operation (i.e., from start to
    stop) in a recording. We mixed a target machine sound with environmental
    noise, and only noisy recordings are provided as training/test data. The
    environmental noise samples were recorded in several real factory
    environments. For the details of the recording procedure, please refer to
    the papers of `ToyADMOS <https://ieeexplore.ieee.org/document/8937164>`_ 
    and `MIMII <http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Purohit_21.pdf>`_
    Dataset.


    **Development and evaluation datasets**

    We first define two important terms in this task: Machine Type and Machine
    ID. Machine Type means the kind of machine, which in this task can be one
    of six: toy-car, toy-conveyor, valve, pump, fan, and slide rail. Machine ID
    is the identifier of each individual of the same type of machine, which in
    the training dataset can be of three or four.

    * **Development dataset**: Each Machine Type has three or four Machine IDs. Each
      machine ID's dataset consists of (i) around 1,000 samples of normal sounds
      for training and (ii) 100-200 samples each of normal and anomalous sounds
      for the test. The normal and anomalous sound samples in (ii) are only for
      checking performance therefore the sound samples in (ii) shall not be used
      for training.
        
    * **Evaluation dataset**: This dataset consists of the same Machine Types' test
      samples as the development dataset. The number of test samples for each
      Machine ID is around 400, none of which have a condition label (i.e.,
      normal or anomaly). Note that the Machine IDs of the evaluation dataset are
      different from those of the development dataset.
    
    * **Additional training dataset**: This dataset includes around 1,000 normal
      samples for each Machine Type and Machine ID used in the evaluation
      dataset. The participants can also use this dataset for training. The
      additional training dataset will be open on April 1st.


    **Baseline system**

    A simple baseline system is available on the `Github repository 
    <https://github.com/y-kawagu/dcase2020_task2_baseline>`_. The 
    baseline system provides a simple entry-level approach that gives a
    reasonable performance in the dataset of Task 2. It is a good starting
    point, especially for entry-level researchers who want to get familiar
    with the anomalous-sound-detection task.

    **Conditions of use**

    This dataset was created jointly by NTT Corporation and Hitachi, Ltd. and
    is available under a Creative Commons Attribution-NonCommercial-ShareAlike
    4.0 International (CC BY-NC-SA 4.0) license.

 
    **Publication**

    If you use this dataset, please cite all the following three papers:

    .. code-block:: latex

        Yuma Koizumi, Shoichiro Saito, Noboru Harada, Hisashi Uematsu, andKeisuke Imoto,
        "ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection,"
        Proc. of IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2019.

    .. code-block:: latex

        Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi,
        "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection",
        in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.


    .. code-block:: latex

        Yuma Koizumi, Yohei Kawaguchi, Keisuke Imoto, Toshiki Nakamura, Yuki Nikaido, Ryo Tanabe, Harsh Purohit, Kaori Suefusa, Takashi Endo, Masahiro Yasuda, and Noboru Harada,
        "Description and Discussion on DCASE2020 Challenge Task2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring,",
        in arXiv e-prints: 2006.05822, 2020.


    **Feedback**

    If there is any problem, please contact us:

    * Yuma Koizumi, koizumi.yuma@ieee.org
    * Yohei Kawaguchi, yohei.kawaguchi.xk@hitachi.com
    * Keisuke Imoto, keisuke.imoto@ieee.org

"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import glob

from soundata import download_utils, jams_utils, core, annotations, io


BIBTEX = """

"""
REMOTES = {
    "dev_data_fan": download_utils.RemoteFileMetadata(
        filename="dev_data_fan.zip",
        url="https://zenodo.org/record/3678171/files/dev_data_fan.zip?download=1",
        checksum="649bdfc06263ae7a838963f43b6641e6",
        destination_dir="./development",
    ),
    "dev_data_pump": download_utils.RemoteFileMetadata(
        filename="dev_data_pump.zip",
        url="https://zenodo.org/record/3678171/files/dev_data_pump.zip?download=1",
        checksum="90e7091ef722b7238a7f1009365779cd",
        destination_dir="./development",
    ),
    "dev_data_slider": download_utils.RemoteFileMetadata(
        filename="dev_data_slider.zip",
        url="https://zenodo.org/record/3678171/files/dev_data_slider.zip?download=1",
        checksum="da24a757719f0d94d5aa2d646bbfdc86",
        destination_dir="./development",
    ),
    "dev_data_ToyCar": download_utils.RemoteFileMetadata(
        filename="dev_data_ToyCar.zip",
        url="https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1",
        checksum="4dec75ca8d9f666aa9e4c1894a740501",
        destination_dir="./development",
    ),
    "dev_data_ToyConveyor": download_utils.RemoteFileMetadata(
        filename="dev_data_ToyConveyor.zip",
        url="https://zenodo.org/record/3678171/files/dev_data_ToyConveyor.zip?download=1",
        checksum="03b6aa1bfd09a39d53af0ba39f71fa91",
        destination_dir="./development",
    ),
    "dev_data_valve": download_utils.RemoteFileMetadata(
        filename="dev_data_valve.zip",
        url="https://zenodo.org/record/3678171/files/dev_data_valve.zip?download=1",
        checksum="34d2672f55bb041589ef79c10dc89934",
        destination_dir="./development",
    ),
    "eval_data_train_fan": download_utils.RemoteFileMetadata(
        filename="eval_data_train_fan.zip",
        url="https://zenodo.org/record/3727685/files/eval_data_train_fan.zip?download=1",
        checksum="567798854130c8019df8b664c095ed1e",
        destination_dir="./additional_training",
    ),
    "eval_data_train_pump": download_utils.RemoteFileMetadata(
        filename="eval_data_train_pump.zip",
        url="https://zenodo.org/record/3727685/files/eval_data_train_pump.zip?download=1",
        checksum="4e33ae9c0db5cc88437675f5e317caee",
        destination_dir="./additional_training",
    ),
    "eval_data_train_slider": download_utils.RemoteFileMetadata(
        filename="eval_data_train_slider.zip",
        url="https://zenodo.org/record/3727685/files/eval_data_train_slider.zip?download=1",
        checksum="f84f186f4295fc23754abcdd1b0580a7",
        destination_dir="./additional_training",
    ),
    "eval_data_train_ToyCar": download_utils.RemoteFileMetadata(
        filename="eval_data_train_ToyCar.zip",
        url="https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip?download=1",
        checksum="a5d4ccd498af70b04dd74b644b820006",
        destination_dir="./additional_training",
    ),
    "eval_data_train_ToyConveyor": download_utils.RemoteFileMetadata(
        filename="eval_data_train_ToyConveyor.zip",
        url="https://zenodo.org/record/3727685/files/eval_data_train_ToyConveyor.zip?download=1",
        checksum="0e808ac0b761e9fb13d5e0c65caff00d",
        destination_dir="./additional_training",
    ),
    "eval_data_train_valve": download_utils.RemoteFileMetadata(
        filename="eval_data_train_valve.zip",
        url="https://zenodo.org/record/3727685/files/eval_data_train_valve.zip?download=1",
        checksum="057cdd560caffb8305449541afd28c6d",
        destination_dir="./additional_training",
    ),
    "eval_data_test_fan": download_utils.RemoteFileMetadata(
        filename="eval_data_test_fan.zip",
        url="https://zenodo.org/record/3841772/files/eval_data_test_fan.zip?download=1",
        checksum="1eb9356a768cadfd0f2e59a5c57e578b",
        destination_dir="./evaluation",
    ),
    "eval_data_test_pump": download_utils.RemoteFileMetadata(
        filename="eval_data_test_pump.zip",
        url="https://zenodo.org/record/3841772/files/eval_data_test_pump.zip?download=1",
        checksum="23a8f8f924be218c69df67fe07360348",
        destination_dir="./evaluation",
    ),
    "eval_data_test_slider": download_utils.RemoteFileMetadata(
        filename="eval_data_test_slider.zip",
        url="https://zenodo.org/record/3841772/files/eval_data_test_slider.zip?download=1",
        checksum="0193c769073840332ce3aad84b1ccaa2",
        destination_dir="./evaluation",
    ),
    "eval_data_test_ToyCar": download_utils.RemoteFileMetadata(
        filename="eval_data_test_ToyCar.zip",
        url="https://zenodo.org/record/3841772/files/eval_data_test_ToyCar.zip?download=1",
        checksum="bea2bdd612f616be8f2b8eb087b32c7c",
        destination_dir="./evaluation",
    ),
    "eval_data_test_ToyConveyor": download_utils.RemoteFileMetadata(
        filename="eval_data_test_ToyConveyor.zip",
        url="https://zenodo.org/record/3841772/files/eval_data_test_ToyConveyor.zip?download=1",
        checksum="ccfcc9847c7d3404d7ee9d6d9e0b2ba2",
        destination_dir="./evaluation",
    ),
    "eval_data_test_valve": download_utils.RemoteFileMetadata(
        filename="eval_data_test_valve.zip",
        url="https://zenodo.org/record/3841772/files/eval_data_test_valve.zip?download=1",
        checksum="f69c551d2088691050d20cccca9d631c",
        destination_dir="./evaluation",
    ),
}

LICENSE_INFO = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)"


class Clip(core.Clip):
    """DCASE 2020 Task 2 Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio_path (str): path to the audio file
        split (str): subset the clip belongs to (for experiments):
            development (train, test), additional_training (train) or
            evaluation (test)
        tags (soundata.annotation.Tags): normal or anomaly
        machine_type (str): machine type
        machine_id (int): machine id

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
    def split(self):
        return self._clip_metadata.get("split")

    @property
    def tags(self) -> Optional[annotations.Tags]:
        if not os.path.isfile(self.audio_path):
            raise FileNotFoundError(
                "The audio file of this clip is not available locally. You may need to run .downlad()"
            )

        tag_name = self.clip_id.split("/")[-1].split("id_")[0]

        if tag_name is "":
            return None
        else:
            return annotations.Tags([tag_name[:-1]], np.array([1.0]))

    @property
    def machine_type(self):
        return self._clip_metadata.get("machine_type")

    @property
    def machine_id(self):
        return self._clip_metadata.get("machine_id")

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            tags=self.tags,
            metadata=self._clip_metadata,
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a  DCASE 2020 Task 2 audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate of 16000 without resampling.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The  DCASE 2020 Task 2 dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="dcase2020task2",
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

        subsets = ["development", "additional_training", "evaluation"]

        splits = [
            "development.train",
            "development.test",
            "additional_training.train",
            "evaluation.test",
        ]

        machine_types = ["fan", "pump", "slider", "ToyCar", "ToyConveyor", "valve"]

        metadata_index = {}

        for split in splits:

            subset, fold = split.split(".")

            for mt in machine_types:

                audio_path = os.path.join(self.data_home, subset, mt, fold)

                wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))

                for wf in wavfiles:

                    filename = os.path.basename(wf).replace(".wav", "")

                    clip_id = "{}.{}/{}/{}".format(subset, fold, mt, filename)

                    machine_id = filename.split("id_")[-1]

                    metadata_index[clip_id] = {
                        "split": split,
                        "machine_type": mt,
                        "machine_id": machine_id,
                    }

        return metadata_index
