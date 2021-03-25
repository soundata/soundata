import pytest
import numpy as np

import soundata
from soundata import core


def test_clip():
    index = {
        "clips": {
            "a": {
                "audio": (None, None),
                "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
            }
        }
    }
    clip_id = "a"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clip = core.Clip(clip_id, data_home, dataset_name, index, lambda: None)

    assert clip.clip_id == clip_id
    assert clip._dataset_name == dataset_name
    assert clip._data_home == data_home
    assert clip._clip_paths == {
        "audio": (None, None),
        "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
    }
    assert clip._metadata() is None
    with pytest.raises(AttributeError):
        clip._clip_metadata

    with pytest.raises(NotImplementedError):
        clip.to_jams()

    path_good = clip.get_path("annotation")
    assert path_good == "tests/resources/sound_datasets/asdf/asdd"
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
    assert clip_metadata_global._clip_metadata == {
        "asdf": [1, 2, 3],
        "asdd": [4, 5, 6],
    }

    clip_metadata_none = core.Clip(
        clip_id, data_home, dataset_name, index, metadata_none
    )
    with pytest.raises(AttributeError):
        clip_metadata_none._clip_metadata


def test_clip_repr():
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
            """ThisObjectType: I have a docstring"""
            return None

        @property
        def g(self):
            """I have an improper docstring"""
            return None

        def h(self):
            return "I'm a function!"

    expected1 = """Clip(\n  a="asdf",\n  b=1.2345678,\n  """
    expected2 = """c={1: 'a', 'b': 2},\n  e=None,\n  """
    expected3 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
    expected4 = """f: ThisObjectType,\n  g: I have an improper docstring,\n)"""

    test_clip = TestClip()
    actual = test_clip.__repr__()
    assert actual == expected1 + expected2 + expected3 + expected4

    with pytest.raises(NotImplementedError):
        test_clip.to_jams()


def test_clipgroup_repr():
    class TestClip(core.Clip):
        def __init__(self):
            self.a = "asdf"

    class TestClipGroup(core.ClipGroup):
        def __init__(self, clipgroup_id, data_home):
            self.a = "asdf"
            self.b = 1.2345678
            self.c = {1: "a", "b": 2}
            self._d = "hidden"
            self.e = None
            self.long = "a" + "b" * 50 + "c" * 50
            self.clipgroup_id = clipgroup_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._clip_class = TestClip
            self.clip_ids = ["a", "b", "c"]

        @property
        def f(self):
            """ThisObjectType: I have a docstring"""
            return None

        @property
        def g(self):
            """I have an improper docstring"""
            return None

        def h(self):
            return "I'm a function!"

    expected1 = """Clip(\n  a="asdf",\n  b=1.2345678,\n  """
    expected2 = """c={1: \'a\', \'b\': 2},\n  clip_ids=[\'a\', \'b\', \'c\'],\n  """
    expected3 = """clipgroup_id="test",\n  e=None,\n  """
    expected4 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
    expected5 = """clip_audio_property: ,\n  clips: ,\n  f: ThisObjectType,\n  """
    expected6 = """g: I have an improper docstring,\n)"""

    test_clipgroup = TestClipGroup("test", "foo")
    actual = test_clipgroup.__repr__()
    assert (
        actual == expected1 + expected2 + expected3 + expected4 + expected5 + expected6
    )

    with pytest.raises(NotImplementedError):
        test_clipgroup.to_jams()


def test_dataset():
    dataset = soundata.initialize("esc50")
    assert isinstance(dataset, core.Dataset)

    dataset = soundata.initialize("urbansound8k")
    assert isinstance(dataset, core.Dataset)

    dataset = soundata.initialize("urbansed")
    assert isinstance(dataset, core.Dataset)

    print(dataset)  # test that repr doesn't fail


def test_dataset_errors():
    with pytest.raises(ValueError):
        soundata.initialize("not_a_dataset")

    d = soundata.initialize("esc50")
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

    # uncomment this to test \in dataset with remote index
    # d = soundata.initialize("dataset_with_remote_index")
    # with pytest.raises(FileNotFoundError):
    #     d._index

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
                "audio_master": ("foo/bar", "asdfasdfasdfasdf"),
                "score": (None, None),
            }
        }
    }
    index = {}
    index.update(index_clips)
    index.update(index_clipgroups)
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = core.ClipGroup(
        clipgroup_id, data_home, dataset_name, index, core.Clip, lambda: None
    )

    path_good = clipgroup.get_path("audio_master")
    assert path_good == "tests/resources/sound_datasets/foo/bar"
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

    class TestTrack(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack1(core.ClipGroup):
        def __init__(
            self,
            clipgroup_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                clipgroup_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            return "f"

    # import pdb;pdb.set_trace()
    clipgroup = TestMultiTrack1(
        clipgroup_id, data_home, dataset_name, index, TestTrack, lambda: None
    )
    clipgroup.to_jams()
    clipgroup.get_target(["a"])
    clipgroup.get_random_target()


def test_multitrack_mixing():
    class TestTrack(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack1(core.ClipGroup):
        def __init__(
            self,
            clipgroup_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                clipgroup_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestMultiTrack1(
        clipgroup_id, data_home, dataset_name, index, TestTrack, lambda: None
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


def test_multitrack_unequal_len():
    class TestTrack(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, np.random.randint(50, 100))), 1000

    class TestMultiTrack1(core.ClipGroup):
        def __init__(
            self,
            clipgroup_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                clipgroup_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestMultiTrack1(
        clipgroup_id, data_home, dataset_name, index, TestTrack, lambda: None
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


def test_multitrack_unequal_sr():
    class TestTrack(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), np.random.randint(10, 1000)

    class TestMultiTrack1(core.ClipGroup):
        def __init__(
            self,
            clipgroup_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                clipgroup_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestMultiTrack1(
        clipgroup_id,
        data_home,
        dataset_name,
        index,
        TestTrack,
        lambda: clip_metadata_none,
    )

    with pytest.raises(ValueError):
        clipgroup.get_target(["a", "b", "c"])


def test_multitrack_mono():
    ### no first channel - audio shapes (100,)
    class TestTrack(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (100)), 1000

    class TestClipGroup1(core.ClipGroup):
        def __init__(
            self,
            clipgroup_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                clipgroup_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestClipGroup1(
        clipgroup_id, data_home, dataset_name, index, TestTrack, lambda: None
    )

    target1 = clipgroup.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = clipgroup.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2

    ### one channel mono shape (1, 100)
    class TestTrack1(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (1, 100)), 1000

    class TestClipGroup1(core.ClipGroup):
        def __init__(
            self,
            clipgroup_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                clipgroup_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def clip_audio_property(self):
            return "f"

    index = {"clipgroups": {"ab": {"clips": ["a", "b", "c"]}}}
    clipgroup_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/sound_datasets"
    clipgroup = TestClipGroup1(
        clipgroup_id, data_home, dataset_name, index, TestTrack, lambda: None
    )

    target1 = clipgroup.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = clipgroup.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2
