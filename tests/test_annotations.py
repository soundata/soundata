import sys
import pytest
import numpy as np

import soundata
from soundata import annotations


def test_repr():
    class TestAnnotation(annotations.Annotation):
        def __init__(self):
            self.a = ["a", "b", "c"]
            self.b = np.array([[1, 2], [1, 4]])
            self._c = "hidden"

    test_track = TestAnnotation()
    assert test_track.__repr__() == """TestAnnotation(a, b)"""

    event_data = annotations.Events(
        np.array([[1.0, 2.0], [3.0, 4.0]]), ["Siren", "Dog"]
    )
    assert event_data.__repr__() == "Events(confidence, intervals, labels)"


def test_tags():
    # test good data
    labels = ["Siren", "Laughter", "Engine"]
    confidence = np.array([1, 0.5, 0.2])
    tags = annotations.Tags(labels, confidence)
    assert tags.labels == labels
    assert np.allclose(tags.confidence, confidence)

    # test bad data
    bad_labels = ["Siren", "Laughter", 5]
    pytest.raises(TypeError, annotations.Tags, bad_labels, confidence)

    bad_confidence = np.array([1, 0.5, -0.2])
    pytest.raises(ValueError, annotations.Tags, labels, bad_confidence)


def test_events():
    # test good data
    intervals = np.array([[1.0, 2.0], [1.5, 3.0], [2.0, 3.0]])
    labels = ["Siren", "Laughter", "Engine"]
    confidence = np.array([1, 0.5, 0.2])
    events = annotations.Events(intervals, labels, confidence)
    assert np.allclose(events.intervals, intervals)
    assert events.labels == labels
    assert np.allclose(events.confidence, confidence)

    # test bad data
    bad_intervals = np.array([[1.0, 0.0], [1.5, 3.0], [2.0, 3.0]])
    pytest.raises(ValueError, annotations.Events, bad_intervals, labels, confidence)

    bad_labels = ["Siren", "Laughter", 5]
    pytest.raises(TypeError, annotations.Events, intervals, bad_labels, confidence)

    bad_confidence = np.array([1, 0.5, -0.2])
    pytest.raises(ValueError, annotations.Events, intervals, labels, bad_confidence)


def test_multiannotator():
    # test good data
    annotators = ["annotator_1", "annotator_2"]
    labels_1 = ["Siren", "Engine"]
    labels_2 = ["Siren", "Dog"]
    confidence_1 = np.array([1.0, 1.0])
    confidence_2 = np.array([1.0, 1.0])
    multi_annot = [
        annotations.Tags(labels_1, confidence_1),
        annotations.Tags(labels_2, confidence_2),
    ]
    tags = annotations.MultiAnnotator(annotators, multi_annot)

    assert tags.labels[0].labels == labels_1
    assert tags.annotators[1] == "annotator_2"
    assert np.allclose(tags.labels[1].confidence, confidence_2)

    # test bad data
    bad_labels = ["Siren", "Laughter", 5]
    pytest.raises(TypeError, annotations.MultiAnnotator, annotators, bad_labels)
    pytest.raises(TypeError, annotations.MultiAnnotator, [0, 1], multi_annot)
    pytest.raises(
        TypeError,
        annotations.MultiAnnotator,
        annotators,
        [["bad", "format"], ["indeed"]],
    )


def test_validate_array_like():
    with pytest.raises(ValueError):
        annotations.validate_array_like(None, list, str)

    annotations.validate_array_like(None, list, str, none_allowed=True)

    with pytest.raises(TypeError):
        annotations.validate_array_like([1, 2], np.ndarray, str)

    with pytest.raises(TypeError):
        annotations.validate_array_like([1, 2], list, str)

    with pytest.raises(TypeError):
        annotations.validate_array_like(np.array([1, 2]), np.ndarray, str)

    with pytest.raises(ValueError):
        annotations.validate_array_like([], list, int)


def test_validate_lengths_equal():
    annotations.validate_lengths_equal([np.array([0, 1])])
    annotations.validate_lengths_equal([np.array([]), None])

    with pytest.raises(ValueError):
        annotations.validate_lengths_equal([np.array([0, 1]), np.array([0])])


def test_validate_confidence():
    annotations.validate_confidence(None)

    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([[0, 1], [0, 2]]))
    with pytest.raises(ValueError):
        annotations.validate_confidence(np.array([0, 2]))


def test_validate_times():
    annotations.validate_times(None)

    with pytest.raises(ValueError):
        annotations.validate_times(np.array([[0, 1], [0, 2]]))

    with pytest.raises(ValueError):
        annotations.validate_times(np.array([2, 0]))


def test_validate_intervals():
    annotations.validate_intervals(None)

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([0, 2]))

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([0, -2]))

    with pytest.raises(ValueError):
        annotations.validate_intervals(np.array([[0, 1], [1, 0.5]]))
