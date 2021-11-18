import numpy as np
import pytest
import jams

from soundata import jams_utils, annotations


def get_jam_data(jam, namespace, annot_numb):
    time = []
    duration = []
    value = []
    confidence = []
    for obs in jam.search(namespace=namespace)[annot_numb]["data"]:
        time.append(obs.time)
        duration.append(round(obs.duration, 3))
        value.append(obs.value)
        confidence.append(obs.confidence)
    return time, duration, value, confidence


def test_tags():
    tag_data1 = annotations.Tags(
        ["blues", "I am a description"], "open", np.array([1.0, 1.0])
    )
    tag_data3 = ("jazz", "wrong format")
    tag_data4 = [(123, "asdf")]
    jam1 = jams_utils.jams_converter(tags=tag_data1, metadata={"duration": 10.0})
    assert jam1.validate()

    with pytest.raises(TypeError):
        jams_utils.jams_converter(tags=tag_data3)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(tags=tag_data4)


def test_multiannotator_tags():
    tag_data1 = annotations.Tags(
        ["blues", "I am a description"], "open", np.array([1.0, 1.0])
    )

    tag_data2 = annotations.Tags(
        ["reds", "We are a description"], "open", np.array([1.0, 1.0])
    )

    tag_data3 = annotations.Tags(
        ["greens", "They are description"], "open", np.array([1.0, 1.0])
    )

    multiannotator_data = annotations.MultiAnnotator(
        ["01", "02", "03"], [tag_data1, tag_data2, tag_data3]
    )

    jam = jams_utils.jams_converter(
        tags=multiannotator_data, metadata={"duration": 10.0}
    )
    assert jam.validate()


def test_events():
    event_data1 = annotations.Events(
        np.array([[0.2, 0.3], [0.3, 0.4]]),
        "seconds",
        ["event A", "event B"],
        "open",
        np.array([1.0, 1.0]),
    )

    event_data2 = annotations.Events(
        np.array([[0.2, 0.3], [0.3, 0.4]]),
        "seconds",
        ["", "a great label"],
        "open",
        np.array([0.0, 1.0]),
    )

    event_data3 = annotations.Events(
        np.array([[0.2, 0.3], [0.3, 20.0]]),  # invalid because > duration
        "seconds",
        ["", "a great label"],
        "open",
        np.array([0.0, 1.0]),
    )

    event_data4 = ("jazz", "wrong format")
    event_data5 = ["wrong format too"]
    event_data6 = [("wrong", "description")]

    jam1 = jams_utils.jams_converter(events=event_data1, metadata={"duration": 10.0})
    assert jam1.validate()

    jam2 = jams_utils.jams_converter(events=event_data2, metadata={"duration": 10.0})
    assert jam2.validate()

    jam3 = jams_utils.jams_converter(events=event_data3, metadata={"duration": 10.0})
    assert jam3.validate()

    with pytest.raises(TypeError):
        jams_utils.jams_converter(events=event_data4)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(events=event_data5)
    with pytest.raises(TypeError):
        jams_utils.jams_converter(events=event_data6)


def test_multiannotator_events():
    event_data1 = annotations.Events(
        np.array([[0.2, 0.3], [0.3, 0.4]]),
        "seconds",
        ["event A", "event B"],
        "open",
        np.array([1.0, 1.0]),
    )

    event_data2 = annotations.Events(
        np.array([[0.2, 0.3], [0.3, 0.4]]),
        "seconds",
        ["", "a great label"],
        "open",
        np.array([0.0, 1.0]),
    )

    event_data3 = annotations.Events(
        np.array([[0.2, 0.3], [0.3, 20.0]]),  # invalid because > duration
        "seconds",
        ["", "a great label"],
        "open",
        np.array([0.0, 1.0]),
    )

    multiannotator_data = annotations.MultiAnnotator(
        ["01", "02", "03"], [event_data1, event_data2, event_data3]
    )

    jam = jams_utils.jams_converter(
        events=multiannotator_data, metadata={"duration": 10.0}
    )
    assert jam.validate()


def test_metadata():
    metadata_1 = {
        "duration": 1.5,
        "artist": "Meatloaf",
        "title": "Le ciel est blue",
        "favourite_color": "rainbow",
    }

    jam_1 = jams_utils.jams_converter(metadata=metadata_1)

    assert jam_1["file_metadata"]["title"] == "Le ciel est blue"
    assert jam_1["file_metadata"]["artist"] == "Meatloaf"
    assert jam_1["file_metadata"]["duration"] == 1.5
    assert jam_1["sandbox"]["favourite_color"] == "rainbow"

    # test meatadata value None
    metadata_2 = {
        "duration": 1.5,
        "artist": "breakmaster cylinder",
        "title": None,
        "extra": None,
    }
    jam2 = jams_utils.jams_converter(metadata=metadata_2)
    assert jam2.validate()
    assert jam2["file_metadata"]["duration"] == 1.5
    assert jam2["file_metadata"]["artist"] == "breakmaster cylinder"
    assert jam2["file_metadata"]["title"] == ""
    assert "extra" not in jam2["sandbox"]


def test_duration():
    # duration from audio file
    jam = jams_utils.jams_converter(audio_path="tests/resources/test.wav")
    assert jam.file_metadata.duration == 1.0
    assert jam.validate()

    # test invalid file path
    with pytest.raises(OSError):
        jams_utils.jams_converter(audio_path="i/dont/exist")

    jam1 = jams_utils.jams_converter(metadata={"duration": 4})
    assert jam1.file_metadata.duration == 4.0
    assert jam1.validate()

    # test incomplete metadata
    jam2 = jams_utils.jams_converter(metadata={"artist": "b"})
    with pytest.raises(jams_utils.jams.SchemaError):
        jam2.validate()

    # test metadata duration and audio file equal
    jam3 = jams_utils.jams_converter(
        audio_path="tests/resources/test.wav", metadata={"duration": 1}
    )
    assert jam3.file_metadata.duration == 1
    assert jam3.validate()

    # test metadata and duration not equal
    jam4 = jams_utils.jams_converter(
        audio_path="tests/resources/test.wav", metadata={"duration": 1000}
    )
    assert jam4.file_metadata.duration == 1000
    assert jam4.validate()
