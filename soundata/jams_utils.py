"""Utilities for converting soundata Annotation classes to jams format.
"""
import logging
import os

import jams
import librosa

from soundata import annotations


def jams_converter(
    audio_path=None,
    spectrogram_path=None,
    metadata=None,
    tags=None,
    events=None,
):
    """Convert annotations from a clip to JAMS format.

    Args:
        audio_path (str or None):
            A path to the corresponding audio file, or None. If provided,
            the audio file will be read to compute the duration. If None,
            'duration' must be a field in the metadata dictionary, or the
            resulting jam object will not validate.
        spectrogram_path (str or None):
            A path to the corresponding spectrum file, or None.
        tags (annotations.Tags or None):
            An instance of annotations.Tags describing the audio tags.
        events (annotations.Events or None):
            An instance of annotations.Events describing the sound events.
    Returns:
        jams.JAMS: A JAMS object containing the annotations.

    """

    jam = jams.JAMS()

    # duration
    duration = None
    if audio_path is not None:
        if os.path.exists(audio_path):
            duration = librosa.get_duration(filename=audio_path)
        else:
            raise OSError(
                "jams conversion failed because the audio file "
                + "for this clip cannot be found, and it is required "
                + "to compute duration."
            )
    if spectrogram_path is not None:
        if audio_path is None:
            duration = metadata["duration"]

    # metadata
    if metadata is not None:
        for key in metadata:
            if (
                key == "duration"
                and duration is not None
                and metadata[key] != duration
                and audio_path is not None
            ):
                logging.warning(
                    "Duration provided in metadata does not"
                    + "match the duration computed from the audio file."
                    + "Using the duration provided by the metadata."
                )

            if metadata[key] is None:
                continue

            if hasattr(jam.file_metadata, key):
                setattr(jam.file_metadata, key, metadata[key])
            else:
                setattr(jam.sandbox, key, metadata[key])

    if jam.file_metadata.duration is None:
        jam.file_metadata.duration = duration

    # soundata tags
    if tags is not None:
        if not isinstance(tags, annotations.Tags):
            raise TypeError("tags should be of type annotations.Tags")
        jam.annotations.append(tags_to_jams(tags, duration=jam.file_metadata.duration))

    # soundata events
    if events is not None:
        if not isinstance(events, annotations.Events):
            raise TypeError("events should be of type annotations.Events")
        jam.annotations.append(events_to_jams(events))

    return jam


def tags_to_jams(tags, duration=0, namespace="tag_open", description=None):
    """Convert tags annotations into jams format.

    Args:
        tags (annotations.Tags): tags annotation object
        namespace (str): the jams-compatible tag namespace
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    ann = jams.Annotation(namespace=namespace)
    ann.annotation_metadata = jams.AnnotationMetadata(data_source="soundata")
    for t, c in zip(tags.labels, tags.confidence):
        ann.append(time=0.0, duration=duration, value=t, confidence=c)
    if description is not None:
        ann.sandbox = jams.Sandbox(name=description)
    return ann


def events_to_jams(events, description=None):
    """Convert events annotations into jams format.

    Args:
        events (annotations.Events): events data object
        description (str): annotation description

    Returns:
        jams.Annotation: jams annotation object.

    """
    jannot_events = jams.Annotation(namespace="segment_open")
    jannot_events.annotation_metadata = jams.AnnotationMetadata(data_source="soundata")

    for inter, label, conf in zip(events.intervals, events.labels, events.confidence):
        jannot_events.append(
            time=inter[0],
            duration=inter[1] - inter[0],
            value=label,
            confidence=conf,
        )
    if description is not None:
        jannot_events.sandbox = jams.Sandbox(name=description)
    return jannot_events
