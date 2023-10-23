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
REMOTES = {
    "dev_bearing": download_utils.RemoteFileMetadata(
        filename="dev_bearing.zip",
        url="https://zenodo.org/records/7882613/files/dev_bearing.zip?download=1",
        checksum="8a813bc8d8f156b5395bfccdfac7673c",
    ),
    "dev_fan": download_utils.RemoteFileMetadata(
        filename="dev_fan.zip",
        url="https://zenodo.org/records/7882613/files/dev_fan.zip?download=1",
        checksum="9348591e96fb0ad499a1e33b082562fc",
    ),
    "dev_gearbox": download_utils.RemoteFileMetadata(
        filename="dev_gearbox.zip",
        url="https://zenodo.org/records/7882613/files/dev_gearbox.zip?download=1",
        checksum="b6e55f6a31faa0fc8569ec0afdd53ccf",
    ),
    "dev_slider": download_utils.RemoteFileMetadata(
        filename="dev_slider.zip",
        url="https://zenodo.org/records/7882613/files/dev_slider.zip?download=1",
        checksum="b3f8dee36b4718c36d659a4fd1c4afe0",
    ),
    "dev_ToyCar": download_utils.RemoteFileMetadata(
        filename="dev_ToyCar.zip",
        url="https://zenodo.org/records/7882613/files/dev_ToyCar.zip?download=1",
        checksum="4e3bf15f4101ed4ed4f1fecde2e2b2a3",
    ),
    "dev_ToyTrain": download_utils.RemoteFileMetadata(
        filename="dev_ToyTrain.zip",
        url="https://zenodo.org/records/7882613/files/dev_ToyTrain.zip?download=1",
        checksum="6b02a6c65eebb3b8b1ae59a6b25bb897",
    ),
    "dev_valve": download_utils.RemoteFileMetadata(
        filename="dev_valve.zip",
        url="https://zenodo.org/records/7882613/files/dev_valve.zip?download=1",
        checksum="b2051a2022eadb53cd97581120811cae",
    ),
    "add_train_bandsaw": download_utils.RemoteFileMetadata(
        filename="eval_data_bandsaw_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_bandsaw_train.zip?download=1",
        checksum="9274dfe63de028743823f1123f8b4b47",
    ),
    "add_train_grinder": download_utils.RemoteFileMetadata(
        filename="eval_data_grinder_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_grinder_train.zip?download=1",
        checksum="17569c1f9df23621a0dbabc430684a35",
    ),
    "add_train_shaker": download_utils.RemoteFileMetadata(
        filename="eval_data_shaker_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_shaker_train.zip?download=1",
        checksum="35f821b5645b731fb5a1750e33b95fc3",
    ),
    "add_train_ToyDrone": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyDrone_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_ToyDrone_train.zip?download=1",
        checksum="7fea367d1384a1521ae24f72203238de",
    ),
    "add_train_ToyNscale": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyNscale_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_ToyNscale_train.zip?download=1",
        checksum="9332822f3e47afd984c01f2ecb5ca3af",
    ),
    "add_train_ToyTank": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyTank_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_ToyTank_train.zip?download=1",
        checksum="b1fd3ab7de7561290df2d477de1c9d33",
    ),
    "add_train_Vacuum": download_utils.RemoteFileMetadata(
        filename="eval_data_Vacuum_train.zip",
        url="https://zenodo.org/records/7830345/files/eval_data_Vacuum_train.zip?download=1",
        checksum="1c8de33d9a8c7850a1f7aaddb97d87be",
    ),
    "eval_bandsaw": download_utils.RemoteFileMetadata(
        filename="eval_data_bandsaw_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_bandsaw_test.zip?download=1",
        checksum="2a8e8f39f6584ab366a8f4da52d4d7a6",
    ),
    "eval_grinder": download_utils.RemoteFileMetadata(
        filename="eval_data_grinder_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_grinder_test.zip?download=1",
        checksum="631b3e1608b6077772829a6e68c82c77",
    ),
    "eval_shaker": download_utils.RemoteFileMetadata(
        filename="eval_data_shaker_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_shaker_test.zip?download=1",
        checksum="ba98c98caa96051ec80e24e44b8fca56",
    ),
    "eval_ToyDrone": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyDrone_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_ToyDrone_test.zip?download=1",
        checksum="fdae7b8d1f4cadb2bea88bc93e2367db",
    ),
    "eval_ToyNscale": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyNscale_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_ToyNscale_test.zip?download=1",
        checksum="62f5f5043d8fb3a305b1c2e1025872de",
    ),
    "eval_ToyTank": download_utils.RemoteFileMetadata(
        filename="eval_data_ToyTank_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_ToyTank_test.zip?download=1",
        checksum="f5639bf58c47169c622751f19c6fc321",
    ),
    "eval_Vacuum": download_utils.RemoteFileMetadata(
        filename="eval_data_Vacuum_test.zip",
        url="https://zenodo.org/records/7860847/files/eval_data_Vacuum_test.zip?download=1",
        checksum="a32524fd8c45b574a560685b38acc4e1",
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
        d1v (int | float | str): First domain shift value or type associated with the domain shift parameter
        d2p (str): Second domain shift parameter specifying the attribute causing the domain shift
        d2v (int | float | str): Second domain shift value or type associated with the domain shift parameter
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
            * int | float | str - first domain shift value of the clip
        """
        return self._clip_metadata.get("d1v")

    @property
    def d2p(self):
        """The clip's second domain shift parameter (d2p).

        Returns:
            * str - second domain shift parameter of the clip
        """
        return self._clip_metadata.get("d2p")

    @property
    def d2v(self):
        """The clip's second domain shift value (d2v).

        Returns:
            * int | float | str - second domain shift value of the clip
        """
        return self._clip_metadata.get("d2v")

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

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="dcase23_task2",
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
        machines_dev = ["fan", "gearbox", "bearing", "slider", "ToyCar", "ToyTrain", "valve"]
        machines_add_train = ["Vacuum", "ToyTank", "ToyNscale", "ToyDrone", "bandsaw", "grinder", "shaker"]

        metadata_index = {}

        # Loop through each machine type for dev_data
        for machine in machines_dev:
            # Paths for metadata files
            metadata_dev_path = os.path.join(self.data_home, "dev_" + machine, "attributes_00.csv")

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
                    key = os.path.join("dev_data", machine, row[0]).replace(".wav", "")
                    metadata_index[key] = {
                        "split": "dev",
                        "d1p": row[1],
                        "d1v": row[2],
                        "d2p": row[3],
                        "d2v": row[4],
                    }

        # Loop through each machine type for add_train_data
        for machine in machines_add_train:
            # Paths for metadata files
            metadata_add_train_path = os.path.join(self.data_home, "eval_data_" + machine + "_train", "attributes_00.csv")

            # Check for file existence
            if not os.path.exists(metadata_add_train_path):
                raise FileNotFoundError(
                    f"Additional training metadata for {machine} not found. Did you run .download()?"
                )

            # Parsing additional training metadata for each machine
            with open(metadata_add_train_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)  # skipping header
                for row in reader:
                    key = os.path.join("add_train_data", machine, row[0]).replace(".wav", "")
                    metadata_index[key] = {
                        "split": "add_train",
                        "d1p": row[1],
                        "d1v": row[2],
                    }

        return metadata_index
