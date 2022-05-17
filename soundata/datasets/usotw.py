"""
Urban Soundscapes of the World

.. admonition:: Dataset Info
    :class: dropdown
    
    *Urban Soundscapes of the World*
    
    *Created by:*
        
    *Description:*
        
    *This dataset is also accessible via:*
        - Zenodo (labelled subset only): https://zenodo.org/record/5645825
        - DR-NTU (all): https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/Y8UQ6F
        
    *Please Acknowledge SINGA:PURA in Academic Research:*
        If you use this dataset please cite its original publication:
        
        K. Ooi, K. N. Watcharasupat, S. Peksi, F. A. Karnapi, Z.-T. Ong, D. Chua, H.-W. Leow, L.-L. Kwok, X.-L. Ng, Z.-A. Loh, W.-S. Gan, "A Strongly-Labelled Polyphonic Dataset of Urban Sounds with Spatiotemporal Context," in Proceedings of the 13th Asia Pacific Signal and Information Processing Association Annual Summit and Conference, 2021.
        
    *License:*
        Creative Commons Attribution-ShareAlike 4.0 International.

"""

from importlib.metadata import metadata
import json
import logging
import os
from typing import Dict, List, Optional, TextIO, Union
from unicodedata import name

import librosa
import numpy as np
import pandas as pd
import urllib
from soundata import annotations, core, download_utils, io, jams_utils
from soundata.validate import md5

# -- Add any relevant citations here
BIBTEX = "???"

# -- Include the dataset's license information
LICENSE_INFO = "??"

MAX_INDEX = 133  # as of 18 May 2022
EXCLUDED_TRACKS = [
    21,
    77,
    86,
    93,
    100,
    102,
]  # these tracks are not listed on the website
BASE_URL = "https://urban-soundscapes.s3.eu-central-1.wasabisys.com/soundscapes/"

VIDEO_PATH = "video/spherical"
AMBISONICS_PATH = "audio/ambisonics"
BINAURAL_PATH = "audio/binaural"

VIDEO_FILENAME = "R{:04d}_segment_ambisonics_headphones_highres.360.mono.mov"
AMBISONICS_FILENAME = "R{:04d}_segment_ambisonics.wav"
BINAURAL_FILENAME = "R{:04d}_segment_binaural.wav"

LAEQ_METADATA_URL = "http://urban-soundscapes.org/wp-content/uploads/2021/06/SotW_LAeq_binaural_average_LR.xlsx"

NAME = "usotw"


@io.coerce_to_bytes_io
def load_audio(fhandle):
    """
    Load a Example audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal at 48 kHz
    """
    data, _ = librosa.load(fhandle, sr=48000, mono=False)
    return data


class Clip(core.Clip):
    """Urban Soundscapes of the World Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): the audio data
        audio_path (str): path to the audio file
        clip_id (str): clip id
        spl (tuple, float): 1-minute LAeq values of the binaural recordings (left, right)
    """
    
    #TODO: add location, coordinates, and date of the recording. 
    # The information is available on the website but there is no csv consolidating this.
    # Scraping is possible

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.jams_path = self.get_path("jams")
        self.txt_path = self.get_path("txt")

    @property
    def audio(self) -> np.ndarray:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
        """
        return load_audio(self.audio_path)

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, metadata=self._clip_metadata
        )


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    Urban Soundscapes of the World
    """

    def __init__(
        self,
        audio_format: str,
        include_video: bool = False,
        include_spatiotemporal: bool = False,
        data_home: Optional[str] = None,
    ):

        remotes = self._make_remotes(
            audio_format=audio_format,
            include_video=include_video,
            index_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "datasets/indexes",
                "{}_index.json".format(NAME),
            ),
        )

        super().__init__(
            data_home,
            name=NAME,
            clip_class=Clip,
            bibtex=BIBTEX,
            remotes=remotes,
            license_info=LICENSE_INFO,
        )

        self.audio_format = audio_format
        self.include_video = include_video
        
        if include_spatiotemporal:
            self.scrape_metadata = True
            
    @core.cached_property
    def _metadata(self):
        spl_path = os.path.join(self.data_home, LAEQ_METADATA_URL.split("/")[-1])
        
        if self.scrape_metadata:
            raise NotImplementedError

    @staticmethod
    def _make_remotes(
        index_path: str,
        audio_format: str,
        include_video: bool = False,
    ):

        audio_format = audio_format.lower()
        assert audio_format in ["ambisonics", "binaural", "all"]

        track_ids = np.delete(np.arange(1, MAX_INDEX + 1), EXCLUDED_TRACKS)
        
        with open(index_path) as fhandle:
            checksums = json.load(fhandle)

        remotes = {}

        if audio_format in ["binaural", "all"]:
            binaural_remotes = {
                f"audio/binaural/R{track_id:04d}": download_utils.RemoteFileMetadata(
                    filename=os.path.join(
                        BINAURAL_PATH, BINAURAL_FILENAME.format(track_id)
                    ),
                    url=os.path.join(
                        BASE_URL, BINAURAL_PATH, BINAURAL_FILENAME.format(track_id)
                    ),
                    checksum=checksums[f"R{track_id:04d}"]["audio/binaural"][1],
                )
                for track_id in track_ids
            }

            remotes.update(binaural_remotes)

        if audio_format in ["ambisonics", "all"]:
            ambisonics_remotes = {
                f"audio/ambisonics/R{track_id:04d}": download_utils.RemoteFileMetadata(
                    filename=os.path.join(
                        AMBISONICS_PATH, AMBISONICS_FILENAME.format(track_id)
                    ),
                    url=os.path.join(
                        BASE_URL, AMBISONICS_PATH, AMBISONICS_FILENAME.format(track_id)
                    ),
                    checksum=checksums[f"R{track_id:04d}"]["audio/ambisonics"][1],
                )
                for track_id in track_ids
            }

            remotes.update(ambisonics_remotes)

        if include_video:
            video_remotes = {
                f"video/R{track_id:04d}": download_utils.RemoteFileMetadata(
                    filename=os.path.join(VIDEO_PATH, VIDEO_FILENAME.format(track_id)),
                    url=os.path.join(
                        BASE_URL, VIDEO_PATH, VIDEO_FILENAME.format(track_id)
                    ),
                    checksum=checksums[f"R{track_id:04d}"]["video"][1],
                )
                for track_id in track_ids
            }

            remotes.update(video_remotes)

        remotes["spl"] = download_utils.RemoteFileMetadata(
            filename=LAEQ_METADATA_URL,
            url=LAEQ_METADATA_URL,
            checksum="d001b32ad3ed7c7954f59784690e0875",
        )

        return remotes

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):

        # create the subdirectories
        try:
            if self.audio_format in ["binaural", "all"]:
                os.makedirs(
                    os.path.join(self.data_home, "audio/binaural"),
                    exist_ok=force_overwrite,
                )
            if self.audio_format in ["ambisonics", "all"]:
                os.makedirs(
                    os.path.join(self.data_home, "audio/ambisonics"),
                    exist_ok=force_overwrite,
                )
            if self.include_video:
                os.makedirs(
                    os.path.join(self.data_home, "video"), exist_ok=force_overwrite
                )
        except:
            logging.info(
                "Download directories already exist."
                + "Rerun with force_overwrite=True to delete the files and force the download."
            )

        # download the files as usual
        return super().download(partial_download, force_overwrite, cleanup)


if __name__ == "__main__":
    # TODO: remove
    """
    def download_from_remote(remote, save_dir, force_overwrite):

        if remote.destination_dir is None:
            download_dir = save_dir
        else:
            download_dir = os.path.join(save_dir, remote.destination_dir)

        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        download_path = os.path.join(download_dir, remote.filename)

        if not os.path.exists(download_path) or force_overwrite:
            # if we got here, we want to overwrite any existing file
            if os.path.exists(download_path):
                os.remove(download_path)

            # If file doesn't exist or we want to overwrite, download it
            with download_utils.DownloadProgressBar(
                unit="B", unit_scale=True, unit_divisor=1024, miniters=1
            ) as t:
                try:
                    urllib.request.urlretrieve(
                        remote.url,
                        filename=download_path,
                        reporthook=t.update_to,
                        data=None,
                    )
                except Exception as exc:
                    raise exc
        else:
            logging.info(
                "{} already exists and will not be downloaded. ".format(download_path)
                + "Rerun with force_overwrite=True to delete this file and force the download."
            )

        # checksum = md5(download_path)
        # if remote.checksum != checksum:

        #     raise IOError(
        #         "{} has an MD5 checksum ({}) "
        #         "differing from expected ({}), "
        #         "file may be corrupted.".format(download_path, checksum, remote.checksum)
        #     )
        return download_path

    def downloader(
        save_dir,
        remotes=None,
        partial_download=None,
        info_message=None,
        force_overwrite=True,
        cleanup=False,
    ):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if cleanup:
            logging.warning(
                "Zip and tar files will be deleted after they are uncompressed. "
                + "If you download this dataset again, it will overwrite existing files, even if force_overwrite=False"
            )

        if remotes is not None:
            if partial_download is not None:
                # check the keys in partial_download are in the download dict
                if not isinstance(partial_download, list) or any(
                    [k not in remotes for k in partial_download]
                ):
                    raise ValueError(
                        "partial_download must be a list which is a subset of {}, but got {}".format(
                            list(remotes.keys()), partial_download
                        )
                    )
                objs_to_download = partial_download
            else:
                objs_to_download = list(remotes.keys())

            logging.info("Downloading {} to {}".format(objs_to_download, save_dir))

            for k in objs_to_download:

                if isinstance(remotes[k], list):
                    if all([remote.filename[-4:-2] == ".z" for remote in remotes[k]]):
                        download_utils.download_multipart_zip(
                            remotes[k], save_dir, force_overwrite, cleanup
                        )
                    else:
                        raise NotImplementedError("Only multipart zip supported.")

                else:

                    logging.info("[{}] downloading {}".format(k, remotes[k].filename))
                    extension = os.path.splitext(remotes[k].filename)[-1]
                    if ".zip" in extension:
                        download_utils.download_zip_file(
                            remotes[k], save_dir, force_overwrite, cleanup
                        )
                    elif (
                        ".gz" in extension or ".tar" in extension or ".bz2" in extension
                    ):
                        download_utils.download_tar_file(
                            remotes[k], save_dir, force_overwrite, cleanup
                        )
                    else:
                        download_from_remote(remotes[k], save_dir, force_overwrite)

                    if remotes[k].unpack_directories:
                        for src_dir in remotes[k].unpack_directories:

                            # path to destination directory
                            destination_dir = (
                                os.path.join(save_dir, remotes[k].destination_dir)
                                if remotes[k].destination_dir
                                else save_dir
                            )
                            # path to directory to unpack
                            source_dir = os.path.join(destination_dir, src_dir)

                            if not os.path.exists(source_dir):
                                logging.info(
                                    "Data not downloaded, because it probably already exists on your computer. "
                                    + "Run .validate() to check, or rerun with force_overwrite=True to delete any "
                                    + "existing files and download from scratch"
                                )
                                return

                            download_utils.move_directory_contents(
                                source_dir, destination_dir
                            )

        if info_message is not None:
            logging.info(info_message.format(save_dir))

    remotes = Dataset._make_remotes("all", True)
    downloader("/home/karn/sound_datasets/usotw-tests2", remotes)
    """