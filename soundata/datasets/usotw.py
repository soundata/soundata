"""
Urban Soundscapes of the World

.. admonition:: Dataset Info
    :class: dropdown
    
    *Urban Soundscapes of the World*
    
    *Created by:* 
        Prof. Bert De Coensel (WAVES Research Group, Department of Information Technology, Ghent University) and others (please see https://urban-soundscapes.org/cities/). 
        
    *Description:*
        A main goal of the Urban Soundscapes of the World project is to create a reference database of examples of urban acoustic environments, consisting of high-quality immersive audiovisual recordings (360-degree video and spatial audio), in adherence to ISO 12913-2. Ultimately, this database may set the scope for immersive recording and reproducing urban acoustic environments with soundscape in mind.
        
    *This dataset is also accessible via:*
        https://urban-soundscapes.org/
        
    *Please Acknowledge Urban Soundscapes of the World in Academic Research:*
        If you use this dataset please cite its original publication below, as well as relevant works listed in https://urban-soundscapes.org/publications/.
        
        B. De Coensel, K. Sun and D. Botteldooren. Urban Soundscapes of the World: selection and reproduction of urban acoustic environments with soundscape in mind. In Proceedings of the 46th International Congress and Exposition on Noise Control Engineering (Inter·noise), Hong Kong (2017).
        
    *License:*
        Not stated by the creators.

"""

import json
import logging
import os
from typing import Optional
from abc import ABC

import librosa
import numpy as np
import pandas as pd

import requests
from soundata import core, download_utils, io, jams_utils, validate
import skvideo.io

# -- Add any relevant citations here
BIBTEX = """
@inproceedings{de2017urban,
  title={Urban Soundscapes of the World: selection and reproduction of urban acoustic environments with soundscape in mind},
  author={De Coensel, Bert and Sun, Kang and Botteldooren, Dick},
  booktitle={Proceedings of the 46th International Congress and Exposition on Noise Control Engineering},
  pages={5407--5413},
  year={2017},
  organization={Institute of Noise Control Engineering}
}
"""

# -- Include the dataset's license information
LICENSE_INFO = "Not Stated by Creators."

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

VIDEO_NOT_FOUND_ERROR = "Video file not found. Make sure that the video has been downloaded, by setting `include_video` to True in the dataset class."
AMBISONICS_FILE_NOT_FOUND_ERROR = (
    "Ambisonics audio file not found. Make sure that the audio has been downloaded."
)
BINAURAL_FILE_NOT_FOUND_ERROR = (
    "Binaural audio file not found. Make sure that the audio has been downloaded."
)
# AUDIO_FILE_NOT_FOUND_ERROR = "Neither ambisonics nor binaural audio file was found. Make sure that the audio has been downloaded."

DEFAULT_FORMAT_NOT_SPECIFIED_WARNING = "The audio format is set to `all` but the default format is not specified. The `audio` property of the clip will return the ambisonics format."
DEFAULT_FORMAT_CONFLICT_WARNING = "Only one audio format is selected but the default format is not the same as audio format. Setting `default_format` to `audio_format`."


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


class BaseClip(core.Clip, ABC):
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

        self.audio_path_dict = {
            "ambisonics": self.get_path("audio/ambisonics"),
            "binaural": self.get_path("audio/binaural"),
        }
        self.audio_path = None
        self.video_path = self.get_path("video")

    @property
    def ambisonics_audio(self):
        """The clip's ambisonics audio.

        Returns:
            * np.ndarray - audio signal
        """
        if os.path.exists(self.audio_path_dict["ambisonics"]):
            return load_audio(self.audio_path_dict["ambisonics"])
        else:
            raise FileNotFoundError(AMBISONICS_FILE_NOT_FOUND_ERROR)

    @property
    def binaural_audio(self):
        """The clip's binaural audio.

        Returns:
            * np.ndarray - audio signal
        """
        if os.path.exists(self.audio_path_dict["binaural"]):
            return load_audio(self.audio_path_dict["binaural"])
        else:
            raise FileNotFoundError(BINAURAL_FILE_NOT_FOUND_ERROR)

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
                f"The {name} field was not found for clip {self.clip_id}. Returning None."
                "Please make sure `include_spatiotemporal` is set to True when the dataset was initialized."
            )

        return value

    @property
    def city(self) -> Optional[str]:
        """Name of the city where the clip was recorded. Returns None if `include_spatiotemporal` is set to False in the dataset class.

        Returns:
            * str or None - city name
        """
        return self.get_scraped_metadata("city")

    @property
    def location(self) -> Optional[str]:
        """Name of the location where the clip was recorded. Returns None if `include_spatiotemporal` is set to False in the dataset class.

        Returns:
            * str or None - location name
        """
        return self.get_scraped_metadata("location")

    @property
    def date(self) -> Optional[np.datetime64]:
        """Date when the clip was recorded. Returns None if `include_spatiotemporal` is set to False in the dataset class.

        Returns:
            * np.datetime64 or None - date of recording
        """
        return self.get_scraped_metadata("date")

    @property
    def dotw(self) -> Optional[int]:
        """Day of the week when the clip was recorded, starting from 0 for Sunday. Returns None if `include_spatiotemporal` is set to False in the dataset class.

        Returns:
            * int or None - day of the week
        """
        return self.get_scraped_metadata("dotw")

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        """Coordinates of the location where the clip was recorded. Returns None if `include_spatiotemporal` is set to False in the dataset class.

        Returns:
            * np.ndarray or None - location coordinates
        """
        return self.get_scraped_metadata("coordinates")

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, metadata=self._clip_metadata
        )

    @property
    def audio(self) -> np.ndarray:
        """The clip's audio. Loads whichever format of audio file is available. If both ambisonics and binaural files are available, the binaural file is returned.

        Returns:
            * np.ndarray - audio signal
        """
        return load_audio(self.audio_path)


class BinauralClip(BaseClip):

    __doc__ = BaseClip.__doc__

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.audio_path_dict["binaural"]


class AmbisonicsClip(BaseClip):
    __doc__ = BaseClip.__doc__

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.audio_path_dict["ambisonics"]


@core.docstring_inherit(BaseClip)
class BinauralClipWithVideo(BinauralClip):
    __doc__ = BaseClip.__doc__

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.audio_path_dict["binaural"]

    @property
    def video(self) -> np.ndarray:
        if os.path.exists(self.video_path):
            return load_video(self.video_path)
        else:
            raise FileNotFoundError(VIDEO_NOT_FOUND_ERROR)


class AmbisonicsClipWithVideo(AmbisonicsClip):
    __doc__ = BaseClip.__doc__

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.audio_path_dict["ambisonics"]

    @property
    def video(self) -> np.ndarray:
        if os.path.exists(self.video_path):
            return load_video(self.video_path)
        else:
            raise FileNotFoundError(VIDEO_NOT_FOUND_ERROR)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    Urban Soundscapes of the World
    """

    def __init__(
        self,
        audio_format: str = "all",
        default_format: str = None,
        include_video: bool = False,
        include_spatiotemporal: bool = False,
        spatiotemporal_from_archive: bool = True,
        data_home: Optional[str] = None,
    ):
        audio_format = audio_format.lower()
        assert audio_format in ["ambisonics", "binaural", "all"]

        default_format = (
            default_format if default_format is None else default_format.lower()
        )
        assert default_format in ["ambisonics", "binaural", None]

        remotes = self._make_remotes(
            audio_format=audio_format,
            include_video=include_video,
            index_path=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "../datasets/indexes",
                "{}_index.json".format(NAME),
            ),
        )

        if audio_format == "all":
            if default_format is None:
                logging.warning(DEFAULT_FORMAT_NOT_SPECIFIED_WARNING)
                default_format = "ambisonics"
        else:
            if default_format is not None and default_format != audio_format:
                logging.warning(DEFAULT_FORMAT_CONFLICT_WARNING)
            default_format = audio_format

        if include_video:
            clip_class = {
                "ambisonics": AmbisonicsClipWithVideo,
                "binaural": BinauralClipWithVideo,
            }[default_format]
        else:
            clip_class = {
                "ambisonics": AmbisonicsClip,
                "binaural": BinauralClip,
            }[default_format]

        super().__init__(
            data_home,
            name=NAME,
            clip_class=clip_class,
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

        if len(items) != 127:  # pragma: no cover
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
                id: {**metadata[id], **spatiotemporal[id]}
                for id in (spatiotemporal.keys() & metadata.keys())
            }

        return metadata

    @staticmethod
    def _make_remotes(
        index_path: str,
        audio_format: str,
        include_video: bool = False,
    ):

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

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.copy_docs(load_video)
    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    def validate(self, verbose=True):
        """Validate if the stored dataset is a valid version

        Args:
            verbose (bool): If False, don't print output

        Returns:
            * list - files in the index but are missing locally
            * list - files which have an invalid checksum

        """

        index = self._index

        if not self.include_video:
            for track in index["clips"]:
                index["clips"][track].pop("video")

        if self.audio_format == "binaural":
            for track in index["clips"]:
                index["clips"][track].pop("audio/ambisonics")

        if self.audio_format == "ambisonics":
            for track in index["clips"]:
                index["clips"][track].pop("audio/binaural")

        missing_files, invalid_checksums = validate.validator(
            index, self.data_home, verbose=verbose
        )
        return missing_files, invalid_checksums
