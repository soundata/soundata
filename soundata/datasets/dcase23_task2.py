"""DCASE23_Task2 Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **DCASE 2023 Task-2**

    *Created By*  

        | Noboru Harada, Daisuke Niizumi, Yasunori Ohishi, Daiki Takeuchi, and Masahiro Yasuda.
        | Hitachi, Ltd.
        | NTT Corporation.

    *Version*  
        1.0 

    *Description*  
        The DCASE 2023 Task 2 "First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring" dataset provides the operating sounds of seven real/toy machines: ToyCar, ToyTrain, Fan, Gearbox, Bearing, Slide rail, and Valve. Each recording is a single-channel, 10-second audio that includes both a machine's operating sound and environmental noise. The dataset contains training clips containing normal sounds in the source and target domain and test clips of both normal and anomalous sounds. 

    *Audio Files Included*  
        10,000 ten-second audio recordings for each machine type in WAV format. The `raw` directory contains recordings as WAV files, with the source/target domain and attributes provided in the file name.

    *Meta-data Files Included*  
        Attribute csv files accompany the audio files for easy access to attributes that cause domain shifts. Each file lists the file names, domain shift parameters, and the value or type of these parameters.

    *Please Acknowledge DCASE 2023 Task 2 in Academic Research*  
        When the DCASE 2023 Task 2 dataset is used for academic research, we would highly appreciate it if scientific publications of works partly based on this dataset cite the following publications:

        .. code-block:: latex

            Noboru Harada, Daisuke Niizumi, Yasunori Ohishi, Daiki Takeuchi, and Masahiro Yasuda. "First-shot anomaly detection for machine condition monitoring: A domain generalization baseline", arXiv e-prints: 2303.00455, 2023.
            Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi. "MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task", Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022), 31-35. Nancy, France, November 2022.
            Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito. "ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions", Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021), 1–5. Barcelona, Spain, November 2021.

    *Conditions of Use*  
        The DCASE 2023 Task 2 dataset was created jointly by Hitachi, Ltd. and NTT Corporation. It is available under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

    *Feedback*  
        For any issues or feedback regarding the dataset, please reach out to:  
        * Kota Dohi: kota.dohi.gr@hitachi.com  
        * Keisuke Imoto: keisuke.imoto@ieee.org  
        * Noboru Harada: noboru@ieee.org  
        * Daisuke Niizumi: daisuke.niizumi.dt@hco.ntt.co.jp  
        * Yohei Kawaguchi: yohei.kawaguchi.xk@hitachi.com.  
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
@article{harada2023firstshot,
  title={First-shot anomaly detection for machine condition monitoring: A domain generalization baseline},
  author={Harada, Noboru and Niizumi, Daisuke and Ohishi, Yasunori and Takeuchi, Daiki and Yasuda, Masahiro},
  journal={arXiv e-prints},
  volume={2303.00455},
  year={2023},
}

@inproceedings{dohi2022mimii,
  title={MIMII DG: sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task},
  author={Dohi, Kota and Nishida, Tomoya and Purohit, Harsh and Tanabe, Ryo and Endo, Takashi and Yamamoto, Masaaki and Nikaido, Yuki and Kawaguchi, Yohei},
  booktitle={Proceedings of the 7th Detection and Classification of Acoustic Scenes and Events 2022 Workshop (DCASE2022)},
  pages={31-35},
  year={2022},
  address={Nancy, France},
  month={November},
}

@inproceedings{harada2021toyadmos2,
  title={ToyADMOS2: another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions},
  author={Harada, Noboru and Niizumi, Daisuke and Takeuchi, Daiki and Ohishi, Yasunori and Yasuda, Masahiro and Saito, Shoichiro},
  booktitle={Proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)},
  pages={1–5},
  year={2021},
  address={Barcelona, Spain},
  month={November},
}

"""

INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="dcase23_task2_index_1.0.json",
        url="https://zenodo.org/records/11176781/files/dcase23_task2_index_1.0.json?download=1",
        checksum="d026ec551fad229ffd8c5e5339100e54",
    ),
    "sample": core.Index(filename="dcase23_task2_index_1.0_sample.json"),
}

REMOTES = {
    "dev_bearing": download_utils.RemoteFileMetadata(
        filename="dev_bearing.zip",
        url="https://zenodo.org/records/7882613/files/dev_bearing.zip?download=1",
        checksum="8a813bc8d8f156b5395bfccdfac7673c",
        destination_dir="7882613",
    ),
    "dev_fan": download_utils.RemoteFileMetadata(
        filename="dev_fan.zip",
        url="https://zenodo.org/records/7882613/files/dev_fan.zip?download=1",
        checksum="9348591e96fb0ad499a1e33b082562fc",
        destination_dir="7882613",
    ),
    "dev_gearbox": download_utils.RemoteFileMetadata(
        filename="dev_gearbox.zip",
        url="https://zenodo.org/records/7882613/files/dev_gearbox.zip?download=1",
        checksum="b6e55f6a31faa0fc8569ec0afdd53ccf",
        destination_dir="7882613",
    ),
    "dev_slider": download_utils.RemoteFileMetadata(
        filename="dev_slider.zip",
        url="https://zenodo.org/records/7882613/files/dev_slider.zip?download=1",
        checksum="b3f8dee36b4718c36d659a4fd1c4afe0",
        destination_dir="7882613",
    ),
    "dev_ToyCar": download_utils.RemoteFileMetadata(
        filename="dev_ToyCar.zip",
        url="https://zenodo.org/records/7882613/files/dev_ToyCar.zip?download=1",
        checksum="4e3bf15f4101ed4ed4f1fecde2e2b2a3",
        destination_dir="7882613",
    ),
    "dev_ToyTrain": download_utils.RemoteFileMetadata(
        filename="dev_ToyTrain.zip",
        url="https://zenodo.org/records/7882613/files/dev_ToyTrain.zip?download=1",
        checksum="6b02a6c65eebb3b8b1ae59a6b25bb897",
        destination_dir="7882613",
    ),
    "dev_valve": download_utils.RemoteFileMetadata(
        filename="dev_valve.zip",
        url="https://zenodo.org/records/7882613/files/dev_valve.zip?download=1",
        checksum="b2051a2022eadb53cd97581120811cae",
        destination_dir="7882613",
    ),
    "add_train_bandsaw": download_utils.RemoteFileMetadata(
        filename="eval_data_bandsaw_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_bandsaw_train.zip?download=1",
        checksum="9274dfe63de028743823f1123f8b4b47",
        destination_dir="7830345",
    ),
    "add_train_grinder": download_utils.RemoteFileMetadata(
        filename="eval_data_grinder_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_grinder_train.zip?download=1",
        checksum="17569c1f9df23621a0dbabc430684a35",
        destination_dir="7830345",
    ),
    "add_train_shaker": download_utils.RemoteFileMetadata(
        filename="eval_data_shaker_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_shaker_train.zip?download=1",
        checksum="35f821b5645b731fb5a1750e33b95fc3",
        destination_dir="7830345",
    ),
    "add_train_ToyDrone": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyDrone_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_ToyDrone_train.zip?download=1",
        checksum="7fea367d1384a1521ae24f72203238de",
        destination_dir="7830345",
    ),
    "add_train_ToyNscale": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyNscale_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_ToyNscale_train.zip?download=1",
        checksum="9332822f3e47afd984c01f2ecb5ca3af",
        destination_dir="7830345",
    ),
    "add_train_ToyTank": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyTank_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_ToyTank_train.zip?download=1",
        checksum="b1fd3ab7de7561290df2d477de1c9d33",
        destination_dir="7830345",
    ),
    "add_train_Vacuum": download_utils.RemoteFileMetadata(
        filename="eval_data_Vacuum_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_Vacuum_train.zip?download=1",
        checksum="1c8de33d9a8c7850a1f7aaddb97d87be",
        destination_dir="7830345",
    ),
    "eval_bandsaw": download_utils.RemoteFileMetadata(
        filename="eval_data_bandsaw_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_bandsaw_test.zip?download=1",
        checksum="2a8e8f39f6584ab366a8f4da52d4d7a6",
        destination_dir="7860847",
    ),
    "eval_grinder": download_utils.RemoteFileMetadata(
        filename="eval_data_grinder_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_grinder_test.zip?download=1",
        checksum="631b3e1608b6077772829a6e68c82c77",
        destination_dir="7860847",
    ),
    "eval_shaker": download_utils.RemoteFileMetadata(
        filename="eval_data_shaker_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_shaker_test.zip?download=1",
        checksum="ba98c98caa96051ec80e24e44b8fca56",
        destination_dir="7860847",
    ),
    "eval_ToyDrone": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyDrone_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_ToyDrone_test.zip?download=1",
        checksum="fdae7b8d1f4cadb2bea88bc93e2367db",
        destination_dir="7860847",
    ),
    "eval_ToyNscale": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyNscale_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_ToyNscale_test.zip?download=1",
        checksum="62f5f5043d8fb3a305b1c2e1025872de",
        destination_dir="7860847",
    ),
    "eval_ToyTank": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyTank_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_ToyTank_test.zip?download=1",
        checksum="f5639bf58c47169c622751f19c6fc321",
        destination_dir="7860847",
    ),
    "eval_Vacuum": download_utils.RemoteFileMetadata(
        filename="eval_data_Vacuum_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_Vacuum_test.zip?download=1",
        checksum="a32524fd8c45b574a560685b38acc4e1",
        destination_dir="7860847",
    ),
}


LICENSE_INFO = "Creative Commons Attribution Non Commercial 4.0 International"


class Clip(core.Clip):
    """DCASE23_Task2 Clip class
    Args:
        clip_id (str): ID of the clip

    Attributes:
        audio (np.ndarray, float): Array representation of the audio clip
        audio_path (str): Path to the audio file
        file_name (str): Name of the clip file, useful for cross-referencing
        d1p (str): First domain shift parameter specifying the attribute causing the domain shift
        d1v (str): First domain shift value or type associated with the domain shift parameter
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
        """The clip's file name.

        Used for cross-referencing with attribute CSV files for additional metadata.

        Returns:
            * str - name of the clip file
        """
        return self._clip_metadata.get("file_name")

    @property
    def d1p(self):
        """The clip's first domain shift parameter (d1p).

        Returns:
            * str - first domain shift parameter of the clip
        """
        return self._clip_metadata.get("d1p")

    @property
    def d1v(self):
        """The clip's first domain shift value (d1v).

        Returns:
            * str - first domain shift value of the clip
        """
        return self._clip_metadata.get("d1v")

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format
        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=44100) -> Tuple[np.ndarray, float]:
    """Load a DCASE23_Task2 audio file.

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
    The DCASE23_Task2 dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="dcase23_task2",
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
        machines_dev = [
            "fan",
            "gearbox",
            "bearing",
            "slider",
            "ToyCar",
            "ToyTrain",
            "valve",
        ]
        machines_add_train = [
            "Vacuum",
            "ToyTank",
            "ToyNscale",
            "ToyDrone",
            "bandsaw",
            "grinder",
            "shaker",
        ]

        metadata_index = {}

        # Loop through each machine type for dev_data
        for machine in machines_dev:
            # Paths for metadata files
            metadata_dev_path = os.path.join(
                self.data_home, "7882613", machine, "attributes_00.csv"
            )
            # Check for file existence
            if not os.path.exists(metadata_dev_path):
                raise FileNotFoundError(
                    f"Development metadata for {machine} not found. Did you run .download()?"
                )

            # Parsing development metadata for each machine
            with open(metadata_dev_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)  # skipping header
                for row in reader:
                    key = row[0].split("/")[-1].replace(".wav", "")
                    metadata_index[key] = {
                        "file_name": row[0],
                        "d1p": row[1],
                        "d1v": row[2],
                    }

        # Loop through each machine type for add_train_data
        for machine in machines_add_train:
            # Paths for metadata files
            metadata_add_train_path = os.path.join(
                self.data_home, "7830345", machine, "attributes_00.csv"
            )

            # Parsing additional training metadata for each machine
            with open(metadata_add_train_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)  # skipping header
                for row in reader:
                    key = row[0].split("/")[-1].replace(".wav", "")
                    metadata_index[key] = {
                        "file_name": row[0],
                        "d1p": row[1],
                        "d1v": row[2],
                    }

        return metadata_index
