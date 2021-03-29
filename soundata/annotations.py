"""soundata annotation data types
"""
import numpy as np


class Annotation(object):
    """Annotation base class"""

    def __repr__(self):
        attributes = [v for v in dir(self) if not v.startswith("_")]
        repr_str = f"{self.__class__.__name__}({', '.join(attributes)})"
        return repr_str


class Tags(Annotation):
    """Tags class

    Attributes:
        labels (list): list of string tags
    """

    def __init__(self, labels, confidence=None) -> None:
        validate_array_like(labels, list, str)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_confidence(confidence)
        validate_lengths_equal([labels, confidence])
        self.labels = labels
        self.confidence = confidence


class Events(Annotation):
    """Events class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
        labels (list): list of event labels (as strings)
        confidence (np.ndarray or None): array of float confidence values
            in the range [0, 1]
    """

    def __init__(self, intervals, labels, confidence=None) -> None:
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, labels, confidence])
        validate_intervals(intervals)
        validate_confidence(confidence)

        self.intervals = intervals
        self.labels = labels
        self.confidence = confidence


class MultiAnnotator(Annotation):
    """Multiple annotator class.
    This class should be used for datasets with multiple annotators (e.g. multiple annotators per clip).

    Attributes:
        annotators (list): list with annotator ids
        labels (list): list of annotations (e.g. [annotations.Tags, annotations.Tags]
    """

    def __init__(self, annotators, labels) -> None:
        validate_array_like(annotators, list, str)
        validate_array_like(labels, list, Annotation, check_child=True)
        validate_lengths_equal([annotators, labels])

        self.annotators = annotators
        self.labels = labels


def validate_array_like(
    array_like, expected_type, expected_dtype, check_child=False, none_allowed=False
):
    """Validate that array-like object is well formed
    If array_like is None, validation passes automatically.
    Args:
        array_like (array-like): object to validate
        expected_type (type): expected type, either list or np.ndarray
        expected_dtype (type): expected dtype
        check_child (bool): if True, checks if all elements of array are children of expected_dtype
        none_allowed (bool): if True, allows array to be None
    Raises:
        TypeError: if type/dtype does not match expected_type/expected_dtype
        ValueError: if array
    """
    if array_like is None:
        if none_allowed:
            return
        else:
            raise ValueError("array_like cannot be None")

    assert expected_type in [
        list,
        np.ndarray,
    ], "expected type must be a list or np.ndarray"

    if not isinstance(array_like, expected_type):
        raise TypeError(
            f"Object should be a {expected_type}, but is a {type(array_like)}"
        )

    if expected_type == list and not all(
        isinstance(n, expected_dtype) for n in array_like
    ):
        if check_child:
            if not all(issubclass(type(n), expected_dtype) for n in array_like):
                raise TypeError(
                    f"List elements should all be instances of {expected_dtype} class"
                )

        raise TypeError(f"List elements should all have type {expected_dtype}")

    if expected_type == np.ndarray and array_like.dtype != expected_dtype:
        raise TypeError(
            f"Array should have dtype {expected_dtype} but has {array_like.dtype}"
        )

    if np.asarray(array_like).size == 0:
        raise ValueError("Object should not be empty, use None instead")


def validate_lengths_equal(array_list):
    """Validate that arrays in list are equal in length

    Some arrays may be None, and the validation for these are skipped.

    Args:
        array_list (list): list of array-like objects

    Raises:
        ValueError: if arrays are not equal in length

    """
    if len(array_list) == 1:
        return

    for att1, att2 in zip(array_list[:1], array_list[1:]):
        if att1 is None or att2 is None:
            continue

        if not len(att1) == len(att2):
            raise ValueError("Arrays have unequal length")


def validate_confidence(confidence):
    """Validate if confidence is well-formed.

    If confidence is None, validation passes automatically

    Args:
        confidence (np.ndarray): an array of confidence values

    Raises:
        ValueError: if confidence are not between 0 and 1

    """
    if confidence is None:
        return

    confidence_shape = np.shape(confidence)
    if len(confidence_shape) != 1:
        raise ValueError(
            f"Confidence should be 1d, but array has shape {confidence_shape}"
        )

    if (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("confidence should be between 0 and 1")


def validate_times(times):
    """Validate if times are well-formed.

    If times is None, validation passes automatically

    Args:
        times (np.ndarray): an array of time stamps

    Raises:
        ValueError: if times have negative values or are non-increasing

    """
    if times is None:
        return

    time_shape = np.shape(times)
    if len(time_shape) != 1:
        raise ValueError(f"Times should be 1d, but array has shape {time_shape}")

    if (times < 0).any():
        raise ValueError("times should be positive numbers")

    if (times[1:] - times[:-1] <= 0).any():
        raise ValueError("times should be strictly increasing")


def validate_intervals(intervals):
    """Validate if intervals are well-formed.

    If intervals is None, validation passes automatically

    Args:
        intervals (np.ndarray): (n x 2) array

    Raises:
        ValueError: if intervals have an invalid shape, have negative values
        or if end times are smaller than start times.

    """
    if intervals is None:
        return

    # validate that intervals have the correct shape
    interval_shape = np.shape(intervals)
    if len(interval_shape) != 2 or interval_shape[1] != 2:
        raise ValueError(
            f"Intervals should be arrays with two columns, but array has {interval_shape}"
        )

    # validate that time stamps are all positive numbers
    if (intervals < 0).any():
        raise ValueError(f"Interval values should be nonnegative numbers")

    # validate that end times are bigger than start times
    elif (intervals[:, 1] - intervals[:, 0] < 0).any():
        raise ValueError(f"Interval start times must be smaller than end times")
