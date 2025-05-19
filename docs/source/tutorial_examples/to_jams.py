import jams_utils    

def to_jams(clip):
    """
    Get the clip's data in jams format

    Returns:
        jams.JAMS: the clip's data in jams format

    """
    return jams_utils.jams_converter(
        audio_path=clip.audio_path, tags=None, metadata=clip._clip_metadata
    )


# Example usage
clip = ...  # load your clip object here
jams = to_jams(clip)
jams.save("example.jams")