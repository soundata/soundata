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

import json
import logging
import os
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import urllib

import requests
from soundata import core, download_utils, io, jams_utils
import skvideo.io

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

LAEQ_METADATA_URL = "https://urban-soundscapes.org/wp-content/uploads/2021/06/SotW_LAeq_binaural_average_LR.xlsx"

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

@io.coerce_to_bytes_io
def load_video(fhandle):
    """
    Load a Example video file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to a video file

    Returns:
        * np.ndarray - the video data
    """
    data = skvideo.io.vread(fhandle)
    return data

class Clip(core.Clip):
    """Urban Soundscapes of the World Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): the audio data
        audio_path (str): path to the audio file
        clip_id (str): clip id
        spl (np.ndarray, float): A two-element numpy array containing the 1-minute LAeq values of the binaural recordings (left, right).
        city (Optional[str]): city name. Returns None if `include_spatiotemporal` is set to False in the dataset class.
        location (Optional[str]): location name. Returns None if `include_spatiotemporal` is set to False in the dataset class.
        coordinates (Optional[str]): coordinates of the recording location. Returns None if `include_spatiotemporal` is set to False in the dataset class.
        date (Optional[str]): date of the recording. Returns None if `include_spatiotemporal` is set to False in the dataset class.
        dotw (Optional[str]): day of the week when the clip was recorded, starting from 0 for Sunday. Returns None if `include_spatiotemporal` is set to False in the dataset class.
    """

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

    @property
    def spl(self) -> np.ndarray:
        """The LAeq values of the binaural recordings.

        Returns:
            * np.ndarray - A two-element numpy array containing the 1-minute LAeq values of the binaural recordings (left, right).
        """
        return self._clip_metadata["spl"]

    def get_scraped_metadata(self, name):
        value = self._clip_metadata.get(name, None)
        if value is None:
            logging.warning(
                f"The {name} field was not found for clip {self.clip_id}. "
                "Please make sure `include_spatiotemporal` is set to True when the dataset was initialized."
            )

        return value

    @property
    def city(self) -> Optional[str]:
        return self.get_scraped_metadata("city")

    @property
    def location(self) -> Optional[str]:
        return self.get_scraped_metadata("location")

    @property
    def date(self) -> Optional[np.datetime64]:
        return self.get_scraped_metadata("date")

    @property
    def dotw(self) -> Optional[int]:
        return self.get_scraped_metadata("dotw")

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        return self.get_scraped_metadata("coordinates")

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
        spatiotemporal_from_archive: bool = True,
        data_home: Optional[str] = None,
    ):
        remotes = self._make_remotes(
            audio_format=audio_format,
            include_video=include_video,
            index_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../datasets/indexes",
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

        self.include_spatiotemporal = include_spatiotemporal
        self.spatiotemporal_from_archive = (
            spatiotemporal_from_archive and include_spatiotemporal
        )

    def scrape_web(self):
        from bs4 import BeautifulSoup

        if self.spatiotemporal_from_archive:
            url = (
                "https://web.archive.org/web/20220519094616/" + BASE_URL + "index.html"
            )
        else:
            url = BASE_URL + "index.html"

        page = requests.get(url).text

        soup = BeautifulSoup(page, "html.parser").body

        items = soup.find("div", class_="container", recursive=False).find_all(
            "div", class_="row"
        )

        if len(items) != 127:
            logging.warning(
                "Expected 127 items from the webpage, got {}.".format(len(items))
            )

        def parse_item(item):

            title = item.find("h5").text
            id = title[:5]
            loc = title[7:]

            city, location = loc.split(", ", 1)

            data = item.find("p")

            coordinates = np.array(
                data.find("a", href=True)["href"].split("query=")[-1].split(",")
            ).astype(float)

            date = (
                data.text.split("\n")[1]
                .replace(" (YouTube preview)", "")
                .replace("Recorded: ", "")
            )

            dotw, date = date.split(", ", 1)
            date = pd.to_datetime(date)

            dotw = (
                date.dayofweek + 1
            ) % 7  # starts with Sunday=0 for compatibility with SINGA:PURA
            date = date.to_datetime64()

            return (
                id,
                {
                    "city": city,
                    "location": location,
                    "coordinates": coordinates,
                    "date": date,
                    "dotw": dotw,
                },
            )

        metadata = {id: meta for id, meta in map(parse_item, items)}

        return metadata

    @core.cached_property
    def _metadata(self):

        spl_path = os.path.join(self.data_home, LAEQ_METADATA_URL.split("/")[-1])
        spl_df = pd.read_excel(spl_path)
        spl_df["spl"] = spl_df.apply(
            lambda r: np.array([r["LAeq_L"], r["LAeq_R"]]), axis=1
        )
        metadata = spl_df.set_index("Recording")[["spl"]].to_dict(orient="index")

        if self.include_spatiotemporal:
            spatiotemporal = self.scrape_web()

            metadata = {
                id: {**metadata[id], **spatiotemporal[id]} for id in spatiotemporal
            }

        return metadata

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
            checksums = json.load(fhandle)["clips"]

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
                    checksum=None,  # checksums[f"R{track_id:04d}"]["video"][1],
                )
                for track_id in track_ids
            }

            remotes.update(video_remotes)

        remotes["spl"] = download_utils.RemoteFileMetadata(
            filename=LAEQ_METADATA_URL.split("/")[-1],
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
                    os.path.join(self.data_home, "video/spherical"),
                    exist_ok=force_overwrite,
                )
        except:
            logging.warn(
                "Download directories already exist."
                + "Rerun with force_overwrite=True to delete the files and force the download."
            )

        # download the files as usual
        return super().download(partial_download, force_overwrite, cleanup)


if __name__ == "__main__":
    ds = Dataset(
        audio_format="all",
        include_video=True,
        data_home="/home/karn/sound_datasets/usotw-tests2",
    )

    ds._metadata

    def download_from_remote(remote, save_dir, force_overwrite):
        """Download a remote dataset into path
        Fetch a dataset pointed by remote's url, save into path using remote's
        filename and ensure its integrity based on the MD5 Checksum of the
        downloaded file.

        Adapted from scikit-learn's sklearn.datasets.base._fetch_remote.

        Args:
            remote (RemoteFileMetadata): Named tuple containing remote dataset
                meta information: url, filename and checksum
            save_dir (str): Directory to save the file to. Usually `data_home`
            force_overwrite  (bool):
                If True, overwrite existing file with the downloaded file.
                If False, does not overwrite, but checks that checksum is consistent.

        Returns:
            str: Full path of the created file.

        """
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
                    error_msg = """
                                soundata failed to download the dataset from {}!
                                Please try again in a few minutes.
                                If this error persists, please raise an issue at
                                https://github.com/soundata/soundata,
                                and tag it with 'broken-link'.
                                """.format(
                        remote.url
                    )
                    logging.error(error_msg)
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

    # rm = {v: r for v, r in ds.remotes.items() if v.startswith("video")}

    # for r in rm:
    #     download_from_remote(rm[r], ds.data_home, force_overwrite=True)
