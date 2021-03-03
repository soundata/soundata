import pytest
import numpy as np

import soundata
from soundata import core


def test_clip():
    index = {
        "tracks": {
            "a": {
                "audio": (None, None),
                "annotation": ("asdf/asdd", "asdfasdfasdfasdf"),
            }
        }
    }
    clip_id = "a"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
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
    assert path_good == "tests/resources/mir_datasets/asdf/asdd"
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


def test_multitrack_repr():
    class TestTrack(core.Clip):
        def __init__(self):
            self.a = "asdf"

    class TestMultiTrack(core.MultiTrack):
        def __init__(self, mtrack_id, data_home):
            self.a = "asdf"
            self.b = 1.2345678
            self.c = {1: "a", "b": 2}
            self._d = "hidden"
            self.e = None
            self.long = "a" + "b" * 50 + "c" * 50
            self.mtrack_id = mtrack_id
            self._data_home = data_home
            self._dataset_name = "foo"
            self._index = None
            self._metadata = None
            self._track_class = TestTrack
            self.track_ids = ["a", "b", "c"]

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
    expected2 = """c={1: \'a\', \'b\': 2},\n  e=None,\n  """
    expected3 = """long="...{}",\n  """.format("b" * 50 + "c" * 50)
    expected4 = """mtrack_id="test",\n  track_ids=[\'a\', \'b\', \'c\'],\n  """
    expected5 = """f: ThisObjectType,\n  g: I have an improper docstring,\n  """
    expected6 = """track_audio_property: ,\n  tracks: ,\n)"""

    test_mtrack = TestMultiTrack("test", "foo")
    actual = test_mtrack.__repr__()
    assert (
        actual == expected1 + expected2 + expected3 + expected4 + expected5 + expected6
    )

    with pytest.raises(NotImplementedError):
        test_mtrack.to_jams()


def test_dataset():
    dataset = soundata.initialize("guitarset")
    assert isinstance(dataset, core.Dataset)

    dataset = soundata.initialize("rwc_jazz")
    assert isinstance(dataset, core.Dataset)

    dataset = soundata.initialize("ikala")
    assert isinstance(dataset, core.Dataset)

    dataset = soundata.initialize("phenicx_anechoic")
    assert isinstance(dataset, core.Dataset)

    print(dataset)  # test that repr doesn't fail


def test_dataset_errors():
    with pytest.raises(ValueError):
        soundata.initialize("not_a_dataset")

    d = soundata.initialize("orchset")
    d._track_class = None
    with pytest.raises(AttributeError):
        d.track("asdf")

    with pytest.raises(AttributeError):
        d.multitrack("asdf")

    with pytest.raises(AttributeError):
        d.load_tracks()

    with pytest.raises(AttributeError):
        d.load_multitracks()

    with pytest.raises(AttributeError):
        d.choice_track()

    with pytest.raises(AttributeError):
        d.choice_multitrack()

    d = soundata.initialize("acousticbrainz_genre")
    with pytest.raises(FileNotFoundError):
        d._index

    d = soundata.initialize("phenicx_anechoic")
    with pytest.raises(ValueError):
        d._multitrack("a")


def test_multitrack():
    index_tracks = {
        "tracks": {
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
    index_mtracks = {
        "multitracks": {
            "ab": {
                "tracks": ["a", "b"],
                "audio_master": ("foo/bar", "asdfasdfasdfasdf"),
                "score": (None, None),
            }
        }
    }
    index = {}
    index.update(index_tracks)
    index.update(index_mtracks)
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = core.MultiTrack(
        mtrack_id, data_home, dataset_name, index, core.Clip, lambda: None
    )

    path_good = mtrack.get_path("audio_master")
    assert path_good == "tests/resources/mir_datasets/foo/bar"
    path_none = mtrack.get_path("score")
    assert path_none is None

    assert mtrack.mtrack_id == mtrack_id
    assert mtrack._dataset_name == dataset_name
    assert mtrack._data_home == data_home
    assert list(mtrack.tracks.keys()) == ["a", "b"]

    assert mtrack._metadata() is None
    with pytest.raises(AttributeError):
        mtrack._multitrack_metadata

    with pytest.raises(NotImplementedError):
        mtrack.to_jams()

    with pytest.raises(KeyError):
        mtrack.get_target(["c"])

    with pytest.raises(NotImplementedError):
        mtrack.get_random_target()

    with pytest.raises(NotImplementedError):
        mtrack.get_mix()

    with pytest.raises(NotImplementedError):
        mtrack.track_audio_property

    # tracks with metadata
    metadata_mtrack_index = lambda: {"ab": {"x": 1, "y": 2, "z": 3}}
    metadata_global = lambda: {"asdf": [1, 2, 3], "asdd": [4, 5, 6]}
    metadata_none = lambda: None

    mtrack_metadata_tidx = core.MultiTrack(
        mtrack_id, data_home, dataset_name, index, core.Clip, metadata_mtrack_index
    )
    assert mtrack_metadata_tidx._multitrack_metadata == {"x": 1, "y": 2, "z": 3}

    mtrack_metadata_global = core.MultiTrack(
        mtrack_id, data_home, dataset_name, index, core.Clip, metadata_global
    )
    assert mtrack_metadata_global._multitrack_metadata == {
        "asdf": [1, 2, 3],
        "asdd": [4, 5, 6],
    }

    mtrack_metadata_none = core.MultiTrack(
        mtrack_id, data_home, dataset_name, index, core.Clip, metadata_none
    )
    with pytest.raises(AttributeError):
        mtrack_metadata_none._multitrack_metadata

    class TestTrack(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(
            self,
            mtrack_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                mtrack_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    # import pdb;pdb.set_trace()
    mtrack = TestMultiTrack1(
        mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None
    )
    mtrack.to_jams()
    mtrack.get_target(["a"])
    mtrack.get_random_target()


def test_multitrack_mixing():
    class TestTrack(core.Clip):
        def __init__(
            self, key, data_home="foo", dataset_name="foo", index=None, metadata=None
        ):
            self.key = key

        @property
        def f(self):
            return np.random.uniform(-1, 1, (2, 100)), 1000

    class TestMultiTrack1(core.MultiTrack):
        def __init__(
            self,
            mtrack_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                mtrack_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(
        mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None
    )

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (2, 100)
    assert np.max(np.abs(target1)) <= 1

    target2 = mtrack.get_target(["b", "c"], weights=[0.5, 0.2])
    assert target2.shape == (2, 100)
    assert np.max(np.abs(target2)) <= 1

    target3 = mtrack.get_target(["b", "c"], weights=[0.5, 5])
    assert target3.shape == (2, 100)
    assert np.max(np.abs(target3)) <= 1

    target4 = mtrack.get_target(["a", "c"], average=False)
    assert target4.shape == (2, 100)
    assert np.max(np.abs(target4)) <= 2

    target5 = mtrack.get_target(["a", "c"], average=False, weights=[0.1, 0.5])
    assert target5.shape == (2, 100)
    assert np.max(np.abs(target5)) <= 0.6

    random_target1, t1, w1 = mtrack.get_random_target(n_tracks=2)
    assert random_target1.shape == (2, 100)
    assert np.max(np.abs(random_target1)) <= 1
    assert len(t1) == 2
    assert len(w1) == 2
    assert np.all(w1 >= 0.3)
    assert np.all(w1 <= 1.0)

    random_target2, t2, w2 = mtrack.get_random_target(n_tracks=5)
    assert random_target2.shape == (2, 100)
    assert np.max(np.abs(random_target2)) <= 1
    assert len(t2) == 3
    assert len(w2) == 3
    assert np.all(w2 >= 0.3)
    assert np.all(w2 <= 1.0)

    random_target3, t3, w3 = mtrack.get_random_target()
    assert random_target3.shape == (2, 100)
    assert np.max(np.abs(random_target3)) <= 1
    assert len(t3) == 3
    assert len(w3) == 3
    assert np.all(w3 >= 0.3)
    assert np.all(w3 <= 1.0)

    random_target4, t4, w4 = mtrack.get_random_target(
        n_tracks=2, min_weight=0.1, max_weight=0.4
    )
    assert random_target4.shape == (2, 100)
    assert np.max(np.abs(random_target4)) <= 1
    assert len(t4) == 2
    assert len(w4) == 2
    assert np.all(w4 >= 0.1)
    assert np.all(w4 <= 0.4)

    mix = mtrack.get_mix()
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

    class TestMultiTrack1(core.MultiTrack):
        def __init__(
            self,
            mtrack_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                mtrack_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(
        mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None
    )

    with pytest.raises(ValueError):
        mtrack.get_target(["a", "b", "c"])

    with pytest.raises(KeyError):
        mtrack.get_target(["d", "e"])

    target1 = mtrack.get_target(["a", "b", "c"], enforce_length=False)
    assert target1.shape[0] == 2
    assert np.max(np.abs(target1)) <= 1

    target2 = mtrack.get_target(["a", "b", "c"], average=False, enforce_length=False)
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

    class TestMultiTrack1(core.MultiTrack):
        def __init__(
            self,
            mtrack_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                mtrack_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(
        mtrack_id,
        data_home,
        dataset_name,
        index,
        TestTrack,
        lambda: track_metadata_none,
    )

    with pytest.raises(ValueError):
        mtrack.get_target(["a", "b", "c"])


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

    class TestMultiTrack1(core.MultiTrack):
        def __init__(
            self,
            mtrack_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                mtrack_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(
        mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None
    )

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = mtrack.get_target(["a", "c"], average=False)
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

    class TestMultiTrack1(core.MultiTrack):
        def __init__(
            self,
            mtrack_id,
            data_home,
            dataset_name,
            index,
            track_class,
            metadata,
        ):
            super().__init__(
                mtrack_id,
                data_home,
                dataset_name,
                index,
                track_class,
                metadata,
            )

        def to_jams(self):
            return None

        @property
        def track_audio_property(self):
            return "f"

    index = {"multitracks": {"ab": {"tracks": ["a", "b", "c"]}}}
    mtrack_id = "ab"
    dataset_name = "test"
    data_home = "tests/resources/mir_datasets"
    mtrack = TestMultiTrack1(
        mtrack_id, data_home, dataset_name, index, TestTrack, lambda: None
    )

    target1 = mtrack.get_target(["a", "c"])
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 1

    target1 = mtrack.get_target(["a", "c"], average=False)
    assert target1.shape == (1, 100)
    assert np.max(np.abs(target1)) <= 2
