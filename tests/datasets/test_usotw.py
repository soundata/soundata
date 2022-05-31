import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
from soundata import annotations
from soundata.datasets import usotw
from tests.test_full_dataset import dataset
from tests.test_utils import run_clip_tests
import pytest

TEST_DATA_HOME = "tests/resources/sound_datasets/usotw"


class TestInit:
    @pytest.mark.parametrize(
        ["audio_format", "has_error"],
        [
            ("binaural", False),
            ("ambisonics", False),
            ("all", False),
            ("Binaural", False),
            ("Ambisonics", False),
            ("All", False),
            ("nonsense", True),
            (None, True),
        ],
    )
    def test_audio_format(self, audio_format, has_error):
        if has_error:
            with pytest.raises(
                AssertionError if audio_format is not None else AttributeError
            ):
                usotw.Dataset(audio_format=audio_format)
        else:
            usotw.Dataset(audio_format=audio_format)

    @pytest.mark.parametrize(
        ["default_format", "has_error"],
        [
            ("binaural", False),
            ("ambisonics", False),
            ("all", True),
            ("Binaural", False),
            ("Ambisonics", False),
            ("All", True),
            ("nonsense", True),
            (None, False),
        ],
    )
    def test_default_format(self, default_format, has_error):
        if has_error:
            with pytest.raises(AssertionError):
                usotw.Dataset(default_format=default_format)
        else:
            usotw.Dataset(default_format=default_format)

    def test_no_default_audio_format_warning(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            usotw.Dataset(audio_format="all", default_format=None)

        assert caplog.record_tuples == [
            ("root", logging.WARNING, usotw.DEFAULT_FORMAT_NOT_SPECIFIED_WARNING)
        ]

    @pytest.mark.parametrize(
        "audio_format, default_format",
        [("binaural", "ambisonics"), ("ambisonics", "binaural")],
    )
    def test_conflicting_audio_format_warning(
        self, audio_format: str, default_format: str, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING):
            usotw.Dataset(audio_format=audio_format, default_format=default_format)

        assert caplog.record_tuples == [
            ("root", logging.WARNING, usotw.DEFAULT_FORMAT_CONFLICT_WARNING)
        ]

    @pytest.mark.parametrize(
        ["include_video", "audio_format", "default_format", "expected_clip_class"],
        [
            (True, "binaural", "binaural", usotw.BinauralClipWithVideo),
            (True, "ambisonics", "ambisonics", usotw.AmbisonicsClipWithVideo),
            (True, "all", "ambisonics", usotw.AmbisonicsClipWithVideo),
            (True, "all", None, usotw.AmbisonicsClipWithVideo),
            (True, "all", "binaural", usotw.BinauralClipWithVideo),
            (False, "binaural", "binaural", usotw.BinauralClip),
            (False, "ambisonics", "ambisonics", usotw.AmbisonicsClip),
            (False, "all", "ambisonics", usotw.AmbisonicsClip),
            (False, "all", None, usotw.AmbisonicsClip),
            (False, "all", "binaural", usotw.BinauralClip),
        ],
    )
    def test_clip_class(
        self,
        include_video: bool,
        audio_format: str,
        default_format: str,
        expected_clip_class: usotw.BaseClip,
    ):

        dataset = usotw.Dataset(
            audio_format=audio_format,
            default_format=default_format,
            include_video=include_video,
        )

        assert dataset._clip_class == expected_clip_class

    @pytest.mark.parametrize(
        ["include_spatiotemporal", "spatiotemporal_from_archive", "expected_sfa"],
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ],
    )
    def test_spatiotemporal_from_archive(
        self, include_spatiotemporal, spatiotemporal_from_archive, expected_sfa
    ):

        dataset = usotw.Dataset(
            include_spatiotemporal=include_spatiotemporal,
            spatiotemporal_from_archive=spatiotemporal_from_archive,
        )

        assert dataset.spatiotemporal_from_archive == expected_sfa

    @pytest.mark.parametrize(
        ["audio_format", "include_video"],
        [
            ("binaural", True),
            ("ambisonics", True),
            ("all", True),
            ("binaural", False),
            ("ambisonics", False),
            ("all", False),
        ],
    )
    def test_remotes(self, audio_format, include_video):

        dataset = usotw.Dataset(audio_format=audio_format, include_video=include_video)

        remotes = dataset.remotes

        ambisonics_ok = 127 if (audio_format == "binaural") else 0
        binaural_ok = 127 if (audio_format == "ambisonics") else 0
        video_ok = 127 if (not include_video) else 0

        for k in remotes:
            if "ambisonics" in k:
                ambisonics_ok += 1

            if "binaural" in k:
                binaural_ok += 1

            if "video" in k:
                video_ok += 1

            if ambisonics_ok == 127 and binaural_ok == 127 and video_ok == 127:
                break

        assert ambisonics_ok == 127 and binaural_ok == 127 and video_ok == 127


class TestClipProperties:
    default_clipid = "R0001"

    @pytest.mark.parametrize(
        ["audio_format", "default_format", "final_audio_format"],
        [
            ["binaural", "binaural", "binaural"],
            ["ambisonics", "ambisonics", "ambisonics"],
            ["all", "ambisonics", "ambisonics"],
            ["all", "binaural", "binaural"],
        ],
    )
    def test_metadata_no_scrape(self, audio_format, default_format, final_audio_format):

        dataset = usotw.Dataset(
            audio_format=audio_format,
            default_format=default_format,
            data_home=TEST_DATA_HOME,
            include_spatiotemporal=False,
        )
        clip = dataset.clip(self.default_clipid)

        audio_path_dict = {
            af: os.path.join(TEST_DATA_HOME, "audio", af, f"R0001_segment_{af}.wav")
            for af in ["binaural", "ambisonics"]
        }

        expected_attributes = {
            "clip_id": self.default_clipid,
            "audio_path_dict": audio_path_dict,
            "audio_path": audio_path_dict[final_audio_format],
            "video_path": os.path.join(
                TEST_DATA_HOME,
                "video/spherical/R0001_segment_ambisonics_headphones_highres.360.mono.mov",
            ),
            "spl": np.array([66.7, 64.4]),
        }

        expected_property_types = {
            "clip_id": str,
            "audio_path_dict": Dict[str, str],
            "audio_path": str,
            "video_path": str,
            "spl": np.ndarray,
            "ambisonics_audio": np.ndarray,
            "binaural_audio": np.ndarray,
            "audio": np.ndarray,
            "city": type(None),
            "location": type(None),
            "coordinates": type(None),
            "date": type(None),
            "dotw": type(None),
        }

        run_clip_tests(clip, expected_attributes, expected_property_types)

    @pytest.mark.parametrize("spatiotemporal_from_archive", [True, False])
    def test_metadata_with_scrape(
        self,
        spatiotemporal_from_archive,
        audio_format="binaural",
        default_format="binaural",
        final_audio_format="binaural",
    ):
        # there is no need to check multiple format in audio_path
        # since the no_scrape function already did it
        dataset = usotw.Dataset(
            audio_format=audio_format,
            default_format=default_format,
            data_home=TEST_DATA_HOME,
            include_spatiotemporal=True,
            spatiotemporal_from_archive=spatiotemporal_from_archive,
        )
        clip = dataset.clip(self.default_clipid)

        audio_path_dict = {
            af: os.path.join(TEST_DATA_HOME, "audio", af, f"R0001_segment_{af}.wav")
            for af in ["binaural", "ambisonics"]
        }

        expected_attributes = {
            "clip_id": self.default_clipid,
            "audio_path_dict": audio_path_dict,
            "audio_path": audio_path_dict[final_audio_format],
            "video_path": os.path.join(
                TEST_DATA_HOME,
                "video/spherical/R0001_segment_ambisonics_headphones_highres.360.mono.mov",
            ),
            "spl": np.array([66.7, 64.4]),
            "city": "Montreal",
            "location": "Palais des congrès",
            "coordinates": np.array([45.503457, -73.561461]),
            "date": pd.to_datetime("22 June 2017").to_datetime64(),  # June 22, 2017
            "dotw": 4,  # Thursday
        }

        expected_property_types = {
            "clip_id": str,
            "audio_path_dict": Dict[str, str],
            "audio_path": str,
            "video_path": str,
            "spl": np.ndarray,
            "ambisonics_audio": np.ndarray,
            "binaural_audio": np.ndarray,
            "audio": np.ndarray,
            "city": str,
            "location": str,
            "coordinates": np.ndarray,
            "date": np.datetime64,
            "dotw": int,
        }

        run_clip_tests(clip, expected_attributes, expected_property_types)

    @pytest.mark.parametrize("include_spatiotemporal", [True, False])
    def test_jams(self, include_spatiotemporal):
        dataset = usotw.Dataset(
            data_home=TEST_DATA_HOME, include_spatiotemporal=include_spatiotemporal
        )
        clip = dataset.clip(self.default_clipid)
        jams = clip.to_jams()
        assert jams.validate()


class TestLoad:
    default_clipid = "R0001"
    dataset = usotw.Dataset(data_home=TEST_DATA_HOME, include_spatiotemporal=False, include_video=True)
    clip = dataset.clip(default_clipid)

    def test_load_ambisonics(self):
        audio = self.clip.ambisonics_audio
        trimmed_length = 2
        assert type(audio) == np.ndarray
        assert len(audio.shape) == 2  # check audio is loaded correctly
        assert audio.shape == (4, 48000 * trimmed_length)

    def test_load_binaural(self):
        audio = self.clip.binaural_audio
        trimmed_length = 2
        assert type(audio) == np.ndarray
        assert len(audio.shape) == 2  # check audio is loaded correctly
        assert audio.shape == (2, 48000 * trimmed_length)

    def test_load_video(self):
        video = self.clip.video
        trimmed_frames = 3
        assert type(video) == np.ndarray
        assert len(video.shape) == 4
        assert video.shape == (trimmed_frames, 2048, 4096, 3)

    @pytest.mark.parametrize("default_format", ["ambisonics", "binaural"])
    def test_load_automatic_audio(self, default_format):

        dataset = usotw.Dataset(default_format=default_format, data_home=TEST_DATA_HOME)
        clip = dataset.clip(self.default_clipid)

        if default_format == "ambisonics":
            target = clip.ambisonics_audio
        else:
            target = clip.binaural_audio

        np.testing.assert_array_equal(clip.audio, target)
