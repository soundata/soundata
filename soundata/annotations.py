"""soundata annotation data types
"""

import numpy as np

#: Time units
TIME_UNITS = {"seconds": "seconds", "milliseconds": "milliseconds"}

#: Label units
LABEL_UNITS = {"open": "no strict schema or units"}

#: Azimuth units
AZIMUTH_UNITS = {
    "radians": "values in the interval [-2*pi, 2*pi]",
    "degrees": "values in the interval [-360, 360]",
}

#: Distance units
DISTANCE_UNITS = {
    "meters": "meters",
    "centimeters": "centimeters",
    "millimeters": "millimeters",
}


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
        confidence (np.ndarray or None): array of confidence values, float in [0, 1]
        labels_unit (str): labels unit, one of LABELS_UNITS
    """

    def __init__(self, labels, labels_unit, confidence=None) -> None:
        validate_array_like(labels, list, str)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_confidence(confidence)
        validate_lengths_equal([labels, confidence])
        validate_unit(labels_unit, LABEL_UNITS)

        self.labels = labels
        self.confidence = confidence
        self.labels_unit = labels_unit


class Events(Annotation):
    """Events class

    Attributes:
        intervals (np.ndarray): (n x 2) array of intervals
            (as floats) in seconds in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
        labels (list): list of event labels (as strings)
        confidence (np.ndarray or None): array of confidence values, float in [0, 1]
        labels_unit (str): labels unit, one of LABELS_UNITS
        intervals_unit (str): intervals unit, one of TIME_UNITS
        azimuth (np.ndarray or None): list of size n with np.ndarrays with dtype float,
            indicating the azimuth of the sound event. Values between -360 and 360 for degrees
            and between -2*pi, 2*pi for radians or None.
        azimuth_unit (str): azimuth unit, one of AZIMUTH_UNITS
        elevation (np.ndarray or None): list of size n with np.ndarrays with dtype float,
            indicating the elevation of the sound event. Values between -90 and 90 or None.
        elevation_unit (str): elevation unit, one of AZIMUTH_UNITS
        distance (np.ndarray or None):list of size n with np.ndarrays with dtype float,
            indicating the distance of the sound event. Values must be positive or None.
        distance_unit (str): distance unit, one of DISTANCE_UNITS
        cartesian_coord (np.ndarray or None):
        cartesian_coord_unit (str): cartesian_coord unit, one of DISTANCE_UNITS


    """

    def __init__(
        self,
        intervals,
        intervals_unit,
        labels,
        labels_unit,
        confidence=None,
        azimuth=None,
        azimuth_unit=None,
        elevation=None,
        elevation_unit=None,
        distance=None,
        distance_unit=None,
        cartesian_coord=None,
        cartesian_coord_unit=None,
    ) -> None:
        validate_array_like(intervals, np.ndarray, float)
        validate_array_like(labels, list, str)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        validate_lengths_equal([intervals, labels, confidence])
        validate_intervals(intervals)
        validate_confidence(confidence)
        validate_unit(labels_unit, LABEL_UNITS)
        validate_unit(intervals_unit, TIME_UNITS)
        validate_array_like(azimuth, np.ndarray, float, none_allowed=True)
        validate_array_like(elevation, np.ndarray, float, none_allowed=True)
        validate_array_like(distance, np.ndarray, float, none_allowed=True)
        validate_array_like(cartesian_coord, np.ndarray, float, none_allowed=True)
        validate_lengths_equal(
            [intervals, azimuth, elevation, distance, cartesian_coord]
        )
        validate_azimuth(azimuth, azimuth_unit, allow_none=True)
        validate_distance(distance, allow_none=True)
        validate_elevation(elevation, elevation_unit, allow_none=True)
        validate_cartesian_coord(cartesian_coord, allow_none=True)
        validate_unit(azimuth_unit, AZIMUTH_UNITS, allow_none=True)
        validate_unit(distance_unit, DISTANCE_UNITS, allow_none=True)
        validate_unit(elevation_unit, AZIMUTH_UNITS, allow_none=True)
        validate_unit(cartesian_coord_unit, DISTANCE_UNITS, allow_none=True)

        self.intervals = intervals
        self.intervals_unit = intervals_unit
        self.labels = labels
        self.labels_unit = labels_unit
        self.confidence = confidence
        self.azimuth = azimuth
        self.azimuth_unit = azimuth_unit
        self.elevation = elevation
        self.elevation_unit = elevation_unit
        self.distance = distance
        self.distance_unit = distance_unit
        self.cartesian_coord = cartesian_coord
        self.cartesian_coord_unit = cartesian_coord_unit


#: position units
ELEVATIONS_UNITS = {
    "degrees": "degrees",
}
AZIMUTHS_UNITS = {
    "degrees": "degrees",
}
DISTANCES_UNITS = {
    "meters": "meters",
}

#: Time units
TIME_UNITS = {
    "seconds": "seconds",
    "milliseconds": "milliseconds",
}

#: Label units
LABEL_UNITS = {"open": "no strict schema or units"}


class SpatialEvents:
    """SpatialEvents class

    Attributes:
        intervals (list): list of size n np.ndarrays of shape (m, 2), with intervals
            (as floats) in TIME_UNITS in the form [start_time, end_time]
            with positive time stamps and end_time >= start_time.
            n is the number of sound events.
            m is the number of sounding instances for each sound event.
        intervals_unit (str): intervals unit, one of TIME_UNITS
        time_step (int, float, or None): the time-step between events
            over time in intervals_unit
        elevations (list): list of size n with np.ndarrays with dtype int,
            indicating the elevation of the sound event per time_step if moving
            or a single value if static. Values between -90 and 90
        elevations_unit (str): elevations unit, one of ELEVATIONS_UNITS
        azimuths (list): list of size n with np.ndarrays with dtype int,
            indicating the azimuth of the sound event per time_step if moving
            or a single value if static. Values between -180 and 180
        azimuths_unit (str): azimuths unit, one of AZIMUTHS_UNITS
        distances (list): list of size n with np.ndarrays with dtype int,
            indicating the distance of the sound event per time_step if moving
            or a single value if static. Values must be positive or None
        distances_unit (str): distances unit, one of DISTANCES_UNITS
        labels (list): list of event labels (as strings)
        labels_unit (str): labels unit, one of LABELS_UNITS
        clip_number_indices (list): list of clip number indices (as strings)
        confidence (np.ndarray or None): array of confidence values, float in [0, 1]
    """

    def __init__(
        self,
        intervals,
        intervals_unit,
        elevations,
        elevations_unit,
        azimuths,
        azimuths_unit,
        distances,
        distances_unit,
        labels,
        labels_unit,
        clip_number_index=None,
        time_step=None,
        confidence=None,
    ):
        validate_array_like(intervals, list, list, none_allowed=True)
        validate_array_like(labels, list, str, none_allowed=True)
        validate_array_like(confidence, np.ndarray, float, none_allowed=True)
        if intervals is not None:
            [
                [
                    validate_intervals(intervals[np.newaxis, :])
                    for intervals in event_intervals
                ]
                for event_intervals in intervals
            ]
        validate_confidence(confidence)
        validate_unit(labels_unit, LABEL_UNITS)
        validate_unit(intervals_unit, TIME_UNITS)

        self.intervals = intervals
        self.intervals_unit = intervals_unit
        self.labels = labels
        self.labels_unit = labels_unit
        self.confidence = confidence

        validate_array_like(clip_number_index, list, str, none_allowed=True)
        validate_array_like(elevations, list, list, none_allowed=True)
        validate_array_like(azimuths, list, list, none_allowed=True)
        validate_array_like(distances, list, list, none_allowed=True)
        validate_lengths_equal(
            [
                intervals,
                elevations,
                azimuths,
                distances,
                labels,
                clip_number_index,
                confidence,
            ]
        )
        # validate location information for each event are numpy arrays
        if elevations is not None:
            [
                [
                    validate_array_like(subitem, np.ndarray, int, none_allowed=True)
                    for subitem in sitem
                ]
                for sitem in elevations
            ]

        if azimuths is not None:
            [
                [
                    validate_array_like(subitem, np.ndarray, int, none_allowed=True)
                    for subitem in sitem
                ]
                for sitem in azimuths
            ]
        if distances is not None:
            [
                [
                    validate_array_like(
                        subitem, np.ndarray, np.array([None]).dtype, none_allowed=True
                    )
                    for subitem in item
                ]
                for item in distances
            ]
        # validate length of location information is consistent
        # for each event
        if elevations is not None and azimuths is not None and distances is not None:
            [
                [validate_lengths_equal([e, a, d]) for e, a, d in zip(els, azs, dis)]
                for els, azs, dis in zip(elevations, azimuths, distances)
            ]
        if elevations is not None and azimuths is not None and distances is not None:
            [
                [
                    validate_locations(
                        np.concatenate(
                            [e[:, np.newaxis], a[:, np.newaxis], d[:, np.newaxis]],
                            axis=1,
                        )
                    )
                    for e, a, d in zip(els, azs, dis)
                ]
                for els, azs, dis in zip(elevations, azimuths, distances)
            ]
        if (
            elevations is not None
            and azimuths is not None
            and distances is not None
            and intervals is not None
        ):
            [
                [
                    validate_time_steps(
                        time_step,
                        np.concatenate(
                            [e[:, np.newaxis], a[:, np.newaxis], d[:, np.newaxis]],
                            axis=1,
                        ),
                        i,
                    )
                    for e, a, d, i in zip(els, azs, dis, ivl)
                ]
                for els, azs, dis, ivl in zip(
                    elevations, azimuths, distances, intervals
                )
            ]

        validate_unit(elevations_unit, ELEVATIONS_UNITS)
        validate_unit(azimuths_unit, AZIMUTHS_UNITS)
        validate_unit(distances_unit, DISTANCES_UNITS)

        self.time_step = time_step
        self.elevations = elevations
        self.azimuths = azimuths
        self.distances = distances
        self.clip_number_index = clip_number_index
        self.elevations_unit = elevations_unit
        self.azimuths_unit = azimuths_unit
        self.distances_unit = distances_unit


def validate_time_steps(time_step, locations, interval):
    """Validate if timesteps are well-formed.

    If locations is None, validation passes automatically

    Args:
        time_step (float): spacing between location steps
        locations (np.ndarray): (n x 3) array
        interval (np.ndarray): (n x 2) expected start and end time
            for the locations
    Raises:
        ValueError: if the number of locations does not match
            the number of time_steps that fit in the interval
    """
    if interval[0] > interval[1]:
        raise ValueError("The interval has a start_time greater than the end_time")
    elif len(locations) == 1 and interval[1] - interval[0] > 0:
        pass  # the event is static
    # if the object is static, validation passes
    elif not np.isclose(len(locations) - 1, (interval[1] - interval[0]) / time_step):
        raise ValueError(
            "The number of locations does not fit in the interval, given the time_step"
        )


def validate_locations(locations):
    """Validate if locations are well-formed.
    If locations is None, validation passes automatically
    Args:
        locations (np.ndarray): (n x 3) array
    Raises:
        ValueError: if locations have an invalid shape or
                have cartesian coordinate values outside the expected ranges.
    """

    # validate that locations have the correct shape
    locations_shape = np.shape(locations)
    if len(locations_shape) != 2 or locations_shape[1] != 3:
        raise ValueError(
            f"Locations should be arrays with three columns, but array has shape {locations_shape}"
        )

    # validate that values are within expected ranges
    if (np.abs(locations[:, 0]) > 90).any():
        raise ValueError(f"Elevation values should have magnitude less than 90")
    if (np.abs(locations[:, 1]) > 180).any():
        raise ValueError(f"Azimuth values should have magnitude less than 180")
    elif (locations[:, 2] != None).any():
        raise ValueError(f"Distance values should be nonnegative numbers")


class MultiAnnotator(Annotation):
    """Multiple annotator class.
    This class should be used for datasets with multiple annotators (e.g. multiple annotators per clip).

    Attributes:
        annotators (list): list with annotator ids
        annotations (list): list of annotations (e.g. [annotations.Tags, annotations.Tags]
    """

    def __init__(self, annotators, annotations) -> None:
        validate_array_like(annotators, list, str)
        validate_array_like(annotations, list, Annotation, check_child=True)
        validate_lengths_equal([annotators, annotations])

        self.annotators = annotators
        self.annotations = annotations


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

    if np.asarray(array_like, dtype=object).size == 0:
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

    else:
        for att1, att2 in zip(array_list[:1], array_list[1:]):
            if att1 is None or att2 is None:
                continue

            if not len(att1) == len(att2):
                raise ValueError("Arrays have unequal length")

        # recurse
        validate_lengths_equal(array_list[1:])


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

    if np.any(np.isnan(confidence)):
        raise ValueError("confidence values cannot be nan")

    if any([c < 0 for c in confidence]) or any([c > 1 for c in confidence]):
        raise ValueError(
            "confidence with unit 'likelihood' should be between 0 and 1. "
            + "Found values outside [0, 1]."
        )


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


def validate_unit(unit, unit_values, allow_none=False):
    """Validate that the given unit is one of the allowed unit values.

    Args:
        unit (str): the unit name
        unit_values (dict): dictionary of possible unit values
        allow_none (bool): if true, allows unit=None to pass validation
    Raises:
        ValueError: If the given unit is not one of the allowed unit values
    """
    if allow_none and not unit:
        return

    if unit not in unit_values:
        raise ValueError("unit={} is not one of {}".format(unit, unit_values))


def validate_azimuth(azimuth, azimuth_unit=None, allow_none=False):
    if allow_none and not azimuth_unit:
        return

    if azimuth_unit == "radians":
        if any([a < -2 * np.pi for a in azimuth]) or any(
            [a > 2 * np.pi for a in azimuth]
        ):
            raise ValueError(
                "azimuth with unit 'radians' should be between -2*np.pi and 2*np.pi. "
                + "Found values outside that interval."
            )

    if azimuth_unit == "degrees":
        if any([a < -360 for a in azimuth]) or any([a > 360 for a in azimuth]):
            raise ValueError(
                "azimuth with unit 'degrees' should be between -360 and 360. "
                + "Found values outside that interval."
            )


def validate_elevation(elevation, elevation_unit=None, allow_none=False):
    if allow_none and not elevation_unit:
        return

    if elevation_unit == "radians":
        if any([a < -np.pi / 2 for a in elevation]) or any(
            [a > np.pi / 2 for a in elevation]
        ):
            raise ValueError(
                "elevation with unit 'radians' should be between -np.pi / 2 and np.pi / 2."
                + "Found values outside that interval."
            )

    if elevation_unit == "degrees":
        if any([a < -90 for a in elevation]) or any([a > 90 for a in elevation]):
            raise ValueError(
                "elevation with unit 'degrees' should be between -90 and 90."
                + "Found values outside that interval."
            )


def validate_distance(distance, allow_none=False):
    if allow_none and distance is None:
        return

    if np.any([d < 0 for d in distance]):
        raise ValueError(
            "distance should be bigger or equal than zero. " + "Found negative values."
        )


def validate_cartesian_coord(cartesian_coord, allow_none=False):
    # print(np.shape(cartesian_coord))
    if allow_none and cartesian_coord is None:
        return

    if not np.shape(cartesian_coord)[1] == 3:
        raise ValueError(
            f"cartesian coordinates should have three columns corresponding to x, y and z coordinates. "
            f"Found {len(cartesian_coord)} columns instead."
        )
