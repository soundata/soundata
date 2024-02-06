import soundata
import tensorflow as tf


def data_generator(dataset_name):
    dataset = soundata.initialize(dataset_name)
    dataset.download()  # Download dataset if needed
    for clip_id in dataset.clip(clip_id):
        clip = dataset.clip(clip_id)
        (
            audio_signal,
            _,
        ) = clip.audio  # Assume sample rate consistency or handle as needed
        yield audio_signal.astype("float32"), clip.tags.labels[
            0
        ] if clip.tags.labels else "Unknown"


# Create a Tensorflow dataset
tf_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator("urbansound8k"), output_types=(tf.float32, tf_string)
)

# Example: Iterate through the dataset
for audio, label in tf_dataset.take(1):
    print("Audio Shape:", audio.shape)
    print("Label:", label)
