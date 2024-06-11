import pytest
import os
import sys
import numpy as np

import soundata
from soundata import core
from tests.test_utils import DEFAULT_DATA_HOME
from unittest.mock import Mock, patch


def test_clip():
    index = {
        "clips": {
            "a": {
                "audio": (None, None),
                "annotation": (os.path.normpath("asdf/asdd"), "asdfasdfasdfasdf"),
            }
        }
    }
    clip_id = "a"
    dataset_name = "test"
    data_home = os.path.normpath("tests/resources/sound_datasets")
    clip = core.Clip(clip_id, data_home, dataset_name, index, lambda: None)

    assert clip.clip_id == clip_id
    assert clip._dataset_name == dataset_name
    assert clip._data_home == data_home
    assert clip._clip_paths == {
        "audio": (None, None),
        "annotation": (os.path.normpath("asdf/asdd"), "asdfasdfasdfasdf"),
    }
    assert clip._metadata() is None
    with pytest.raises(AttributeError):
        clip._clip_metadata

    with pytest.raises(NotImplementedError):
        clip.to_jams()

    path_good = clip.get_path("annotation")
    assert path_good == os.path.normpath("tests/resources/sound_datasets/asdf/asdd")
    path_none = clip.get_path("audio")
    assert path_none is None

    # clip with metadata
    metadata_clip_index = lambda: {"a": {"x": 1, "y": 2, "z": 3}}
    metadata_global = lambda: {"asdf": [1, 2, 3], "asdd": [4, 5, 6]}
    metadata_none = lambda: None

    clip_metadata_cidx = core.Clip(
        clip_id, data_home, dataset_name, index, metadata_clip_index
    )
    assert clip_metadata_cidx._clip_metadata == {"x": 1, "y": 2, "z": 3}

    clip_metadata_global = core.Clip(
        clip_id, data_home, dataset_name, index, metadata_global
    )
    assert clip_metadata_global._clip_metadata == {"asdf": [1, 2, 3], "asdd": [4, 5, 6]}

    clip_metadata_none = core.Clip(
        clip_id, data_home, dataset_name, index, metadata_none
    )
    with pytest.raises(AttributeError):
        clip_metadata_none._clip_metadata


def test_clip_repr():
    class TestClip_bad(core.Clip):
        def __init__(self):
            self.a = "asdf"
            self.b = 1.2345678
            self.c = {1: "a", "b": 2}
            self._d = "hidden"
            self.e = None
            self.long = "a" + "b" * 50 + "c" * 50

        @property
        def f(self):
            """ThisObjectType: I have a docstring"""
            return None

    class TestClip(core.Clip):
        def __init__(self):
            self.a = "asdf"
            self.b = 1.2345678
            self.c = {1: "a", "b": 2}
            self._d = "hidden"
            self.e = None
            self.long = "a" + "b" * 50 + "c" * 50

        @property
        def f(self):
            """The proper docstring.

            Returns:
                * str - yay this is correct

            """
            return None

        def h(self):
            return "I'm a function!"

    expected1 = """Clip(\n  a="asdf",\n  b=1.2345678,\n  """
    expected2 = """c={1: 'a', 'b': 2},\n  e=None,\n  """
    expected3 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
    expected4 = (
        """f: The proper docstring.\n                * str - yay this is correct,\n)"""
    )

    test_clip = TestClip()
    actual = test_clip.__repr__()
    print("aaaaa")
    print(expected1 + expected2 + expected3 + expected4)
    assert actual == expected1 + expected2 + expected3 + expected4

    test_clip_bad = TestClip_bad()
    with pytest.raises(NotImplementedError):
        test_clip_bad.__repr__()

    with pytest.raises(NotImplementedError):
        test_clip.to_jams()


# def test_clipgroup_repr():
#     class TestClip(core.Clip):
#         def __init__(self):
#             self.a = "asdf"
#
#     class TestClipGroup_bad(core.ClipGroup):
#         def __init__(self, clipgroup_id, data_home):
#             self.a = "asdf"
#
#         @property
#         def g(self):
#             """I have an improper docstring"""
#             return None
#
#     class TestClipGroup(core.ClipGroup):
#         def __init__(self, clipgroup_id, data_home):
#             self.a = "asdf"
#             self.b = 1.2345678
#             self.c = {1: "a", "b": 2}
#             self._d = "hidden"
#             self.e = None
#             self.long = "a" + "b" * 50 + "c" * 50
#             self.clipgroup_id = clipgroup_id
#             self._data_home = data_home
#             self._dataset_name = "foo"
#             self._index = None
#             self._metadata = None
#             self._clip_class = TestClip
#             self.clip_ids = ["a", "b", "c"]
#
#         @property
#         def f(self):
#             """The proper docstring.
#
#             Returns:
#                 * str - yay this is correct
#
#             """
#             return None
#
#         def h(self):
#             return "I'm a function!"
#
#     expected1 = """Clip(\n  a="asdf",\n  b=1.2345678,\n  """
#     expected2 = """c={1: \'a\', \'b\': 2},\n  clip_ids=[\'a\', \'b\', \'c\'],\n  """
#     expected3 = """clipgroup_id="test",\n  e=None,\n  """
#     expected4 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
#     expected5 = """clip_audio_property: ,\n  clips: ,\n  f: ThisObjectType,\n  """
#     expected6 = """g: I have an improper docstring,\n)"""
#
#     test_clipgroup = TestClipGroup("test", "foo")
#     actual = test_clipgroup.__repr__()
#     assert (
#         actual == expected1 + expected2 + expected3 + expected4 + expected5 + expected6
#     )
#
#     with pytest.raises(NotImplementedError):
#         test_clipgroup.to_jams()


def test_dataset():
    dataset = soundata.initialize("urbansound8k")
    assert isinstance(dataset, core.Dataset)

    print(dataset)  # test that repr doesn't fail


def test_list_versions():
    assert (
        soundata.list_dataset_versions("urbansound8k")
        == "Available versions for urbansound8k: ['1.0']. Default version: 1.0"
    )
    with pytest.raises(ValueError):
        soundata.list_dataset_versions("asdf")


def test_dataset_versions():
    class VersionTest(core.Dataset):
        def __init__(self, data_home=None, version="default"):
            super().__init__(
                data_home,
                version,
                indexes={
                    "default": "1",
                    "test": "0",
                    "0": core.Index("blah_0.json"),
                    "1": core.Index(
                        "blah_1.json", url="https://google.com", checksum="asdf"
                    ),
                    "real": core.Index("dcase_bioacoustic_index_3.0_sample.json"),
                },
            )

    class VersionTest2(core.Dataset):
        def __init__(self, data_home=None, version="default"):
            super().__init__(
                data_home,
                version,
                indexes={
                    "default": "2",
                    "2": core.Index("blah_2.json", url="https://google.com"),
                },
            )

    dataset = VersionTest("asdf")
    assert dataset.version == "1"
    assert os.path.join(
        *dataset.index_path.split(os.path.sep)[-4:]
    ) == os.path.normpath("soundata/datasets/indexes/blah_1.json")

    dataset_default = VersionTest("asdf")
    assert dataset_default.version == "1"
    assert os.path.join(
        *dataset.index_path.split(os.path.sep)[-4:]
    ) == os.path.normpath("soundata/datasets/indexes/blah_1.json")

    dataset_1 = VersionTest("asdf", version="1")
    assert dataset_1.version == "1"
    assert os.path.join(
        *dataset_1.index_path.split(os.path.sep)[-4:]
    ) == os.path.normpath("soundata/datasets/indexes/blah_1.json")
    with pytest.raises(FileNotFoundError):
        dataset_1._index

    local_index_path = os.path.dirname(os.path.realpath(__file__))[:-5]
    dataset_test = VersionTest("asdf", version="test")
    assert dataset_test.version == "0"
    assert dataset_test.index_path == os.path.join(
        local_index_path, "tests", "indexes", "blah_0.json"
    )

    with pytest.raises(IOError):
        dataset_test._index

    dataset_0 = VersionTest("asdf", version="0")
    assert dataset_0.version == "0"
    assert dataset_0.index_path == os.path.join(
        local_index_path, "tests", "indexes", "blah_0.json"
    )

    dataset_real = VersionTest("asdf", version="real")
    assert dataset_real.version == "real"
    assert dataset_real.index_path == os.path.join(
        local_index_path, "tests", "indexes", "dcase_bioacoustic_index_3.0_sample.json"
    )
    idx_test = dataset_real._index
    assert isinstance(idx_test, dict)

    with pytest.raises(ValueError):
        VersionTest("asdf", version="not_a_version")

    with pytest.raises(ValueError):
        VersionTest2("asdf", version="2")


def test_explore_dataset():
    dataset = soundata.initialize("urbansound8k")

    with patch(
        "soundata.display_plot_utils.perform_dataset_exploration"
    ) as mock_function:
        clip_id = "test_clip_id"
        dataset.explore_dataset(clip_id)
        mock_function.assert_called_once_with(dataset, clip_id)

        mock_function.reset_mock()

        dataset.explore_dataset()
        mock_function.assert_called_with(dataset, None)


def test_dataset_errors():
    with pytest.raises(ValueError):
        soundata.initialize("not_a_dataset")

    d = soundata.initialize("esc50", version="sample")
    d._clip_class = None
    with pytest.raises(AttributeError):
        d.clip("asdf")

    with pytest.raises(AttributeError):
        d.clipgroup("asdf")

    with pytest.raises(AttributeError):
        d.load_clips()

    with pytest.raises(AttributeError):
        d.load_clipgroups()

    with pytest.raises(AttributeError):
        d.choice_clip()

    with pytest.raises(AttributeError):
        d.choice_clipgroup()

    # uncomment this to test in dataset with clip_group
    # d = soundata.initialize("dataset_with_clip_group")
    # with pytest.raises(ValueError):
    #     d._clipgroup("a")


def test_clipgroup():
    index_clips = {
        "clips": {
            "a": {
                "audio": (None, None),
                "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
            },
            "b": {
                "audio": (None, None),
                "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
            },
        }
    }
    index_clipgroups = {
        "clipgroups": {
            "ab": {
                "clips": ["a", "b"],
                "audio_master": (os.path.normpath("foo/bar"), "asdfasdfasdfasdf"),
                "score": (None, None),
            }
        }
    }
    index = {}
    index.update(index_clips)
    index.update(index_clipgroups)
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = os.path.normpath("tests/resources/sound_datasets")
    clipgroup = core.ClipGroup(
        clipgroup_id, data_home, dataset_name, index, core.Clip, lambda: None
    )

    path_good = clipgroup.get_path("audio_master")
    assert path_good == os.path.normpath("tests/resources/sound_datasets/foo/bar")
    path_none = clipgroup.get_path("score")
    assert path_none is None

    assert clipgroup.clipgroup_id == clipgroup_id
    assert clipgroup._dataset_name == dataset_name
    assert clipgroup._data_home == data_home
    assert list(clipgroup.clips.keys()) == ["a", "b"]

    assert clipgroup._metadata() is None
    with pytest.raises(AttributeError):
        clipgroup._clipgroup_metadata

    with pytest.raises(NotImplementedError):
        clipgroup.to_jams()

    with pytest.raises(KeyError):
        clipgroup.get_target(["c"])

    with pytest.raises(NotImplementedError):
        clipgroup.get_random_target()

    with pytest.raises(NotImplementedError):
        clipgroup.get_mix()

    with pytest.raises(NotImplementedError):
        clipgroup.clip_audio_property

    # clips with metadata
    metadata_clipgroup_index = lambda: {"ab": {"x": 1, "y": 2, "z": 3}}
    metadata_global = lambda: {"asdf": [1, 2, 3], "asdd": [4, 5, 6]}
    metadata_none = lambda: None

    clipgroup_metadata_cidx = core.ClipGroup(
        clipgroup_id,
        data_home,
        dataset_name,
        index,
        core.Clip,
        metadata_clipgroup_index,
    )
    assert clipgroup_metadata_cidx._clipgroup_metadata == {"x": 1, "y": 2, "z": 3}

    clipgroup_metadata_global = core.ClipGroup(
        clipgroup_id, data_home, dataset_name, index, core.Clip, metadata_global
    )
    assert clipgroup_metadata_global._clipgroup_metadata == {
        "asdf": [1, 2, 3],
        "asdd": [4, 5, 6],
    }

    clipgroup_metadata_none = core.ClipGroup(
        clipgroup_id, data_home, dataset_name, index, core.Clip, metadata_none
    )
    with pytest.raises(AttributeError):
        clipgroup_metadata_none._clipgroup_metadata

    class TestClip(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiClip1(core.ClipGroup):
        def __init__(
            self, clipgroup_id, data_home, dataset_name, index, clip_class, metadata
        ):
            super().__init__(
                clipgroup_id, data_home, dataset_name, index, clip_class, metadata
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            """The clip's audio property.

            Returns:
                * str - some property

            """
            return "f"

    # import pdb;pdb.set_trace()
    clipgroup = TestMultiClip1(
        clipgroup_id, data_home, dataset_name, index, TestClip, lambda: None
    )
    clipgroup.to_jams()
    clipgroup.get_target(["a"])
    clipgroup.get_random_target()


def test_multiclip_mixing():
    class TestClip(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiClip1(core.ClipGroup):
        def __init__(
            self, clipgroup_id, data_home, dataset_name, index, clip_class, metadata
        ):
            super().__init__(
                clipgroup_id, data_home, dataset_name, index, clip_class, metadata
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            """The clip's audio property.

            Returns:
                * str - some property

            """
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestMultiClip1(
        clipgroup_id, data_home, dataset_name, index, TestClip, lambda: None
    )

    target1 = clipgroup.get_target(["a", "c"])
    assert target1.shape == (2, 100)
    assert np.max(np.abs(target1)) <= 1

    target2 = clipgroup.get_target(["b", "c"], weights=[0.5, 0.2])
    assert target2.shape == (2, 100)
    assert np.max(np.abs(target2)) <= 1

    target3 = clipgroup.get_target(["b", "c"], weights=[0.5, 5])
    assert target3.shape == (2, 100)
    assert np.max(np.abs(target3)) <= 1

    target4 = clipgroup.get_target(["a", "c"], average=False)
    assert target4.shape == (2, 100)
    assert np.max(np.abs(target4)) <= 2

    target5 = clipgroup.get_target(["a", "c"], average=False, weights=[0.1, 0.5])
    assert target5.shape == (2, 100)
    assert np.max(np.abs(target5)) <= 0.6

    random_target1, t1, w1 = clipgroup.get_random_target(n_clips=2)
    assert random_target1.shape == (2, 100)
    assert np.max(np.abs(random_target1)) <= 1
    assert len(t1) == 2
    assert len(w1) == 2
    assert np.all(w1 >= 0.3)
    assert np.all(w1 <= 1.0)

    random_target2, t2, w2 = clipgroup.get_random_target(n_clips=5)
    assert random_target2.shape == (2, 100)
    assert np.max(np.abs(random_target2)) <= 1
    assert len(t2) == 3
    assert len(w2) == 3
    assert np.all(w2 >= 0.3)
    assert np.all(w2 <= 1.0)

    random_target3, t3, w3 = clipgroup.get_random_target()
    assert random_target3.shape == (2, 100)
    assert np.max(np.abs(random_target3)) <= 1
    assert len(t3) == 3
    assert len(w3) == 3
    assert np.all(w3 >= 0.3)
    assert np.all(w3 <= 1.0)

    random_target4, t4, w4 = clipgroup.get_random_target(
        n_clips=2, min_weight=0.1, max_weight=0.4
    )
    assert random_target4.shape == (2, 100)
    assert np.max(np.abs(random_target4)) <= 1
    assert len(t4) == 2
    assert len(w4) == 2
    assert np.all(w4 >= 0.1)
    assert np.all(w4 <= 0.4)

    mix = clipgroup.get_mix()
    assert mix.shape == (2, 100)


def test_multiclip_unequal_len():
    class TestClip(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, np.random.randint(50, 100))), 1000

    class TestMultiClip1(core.ClipGroup):
        def __init__(
            self, clipgroup_id, data_home, dataset_name, index, clip_class, metadata
        ):
            super().__init__(
                clipgroup_id, data_home, dataset_name, index, clip_class, metadata
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            """The clip's audio property.

            Returns:
                * str - some property

            """
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestMultiClip1(
        clipgroup_id, data_home, dataset_name, index, TestClip, lambda: None
    )

    with pytest.raises(ValueError):
        clipgroup.get_target(["a", "b", "c"])

    with pytest.raises(KeyError):
        clipgroup.get_target(["d", "e"])

    target1 = clipgroup.get_target(["a", "b", "c"], enforce_length=False)
    assert target1.shape[0] == 2
    assert np.max(np.abs(target1)) <= 1

    target2 = clipgroup.get_target(["a", "b", "c"], average=False, enforce_length=False)
    assert target2.shape[0] == 2
    assert np.max(np.abs(target2)) <= 3


# def test_multiclip_unequal_sr():
#     class TestClip(core.Clip):
#         def __init__(
#             self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
#         ):
#             self.key = key
#
#         @property
#         def f(self):
#             return np.random.uniform(-1, 1, (2, 100)), np.random.randint(10, 1000)
#
#     class TestMultiClip1(core.ClipGroup):
#         def __init__(
#             self, clipgroup_id, data_home, dataset_name, index, clip_class, metadata
#         ):
#             super().__init__(
#                 clipgroup_id, data_home, dataset_name, index, clip_class, metadata
#             )
#
#         def to_jams(self):
#             return None
#
#         @property
#         def clip_audio_property(self):
#             """The clip's audio property.
#
#             Returns:
#                 * str - some property
#
#             """
#             return "f"
#
#     index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
#     clipgroup_id = "ab"
#     dataset_name = "test"
#     data_home = "tests/resources/sound_datasets"
#     clipgroup = TestMultiClip1(
#         clipgroup_id,
#         data_home,
#         dataset_name,
#         index,
#         TestClip,
#         lambda: clip_metadata_none,
#     )
#
#     with pytest.raises(ValueError):
#         clipgroup.get_target(["a", "b", "c"])


def test_multiclip_mono():
    ### no first channel - audio shapes (100,)
    class TestClip(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (100)), 1000

    class TestClipGroup1(core.ClipGroup):
        def __init__(
            self, clipgroup_id, data_home, dataset_name, index, clip_class, metadata
        ):
            super().__init__(
                clipgroup_id, data_home, dataset_name, index, clip_class, metadata
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            """The clip's audio property.

            Returns:
                * str - some property

            """
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestClipGroup1(
        clipgroup_id, data_home, dataset_name, index, TestClip, lambda: None
    )

    target1 = clipgroup.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = clipgroup.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2

    ### one channel mono shape (1, 100)
    class TestClip1(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (1, 100)), 1000

    class TestClipGroup1(core.ClipGroup):
        def __init__(
            self, clipgroup_id, data_home, dataset_name, index, clip_class, metadata
        ):
            super().__init__(
                clipgroup_id, data_home, dataset_name, index, clip_class, metadata
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            """The clip's audio property.

            Returns:
                * str - some property

            """
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestClipGroup1(
        clipgroup_id, data_home, dataset_name, index, TestClip, lambda: None
    )

    target1 = clipgroup.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = clipgroup.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2
