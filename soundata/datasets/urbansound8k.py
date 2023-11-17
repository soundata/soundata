"""UrbanSound8K Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    *Created By:*
        Justin Salamon*^, Christopher Jacoby* and Juan Pablo Bello*
        * Music and Audio Research Lab (MARL), New York University, USA
        ^ Center for Urban Science and Progress (CUSP), New York University, USA
        https://urbansounddataset.weebly.com/
        https://steinhardt.nyu.edu/marl
        http://cusp.nyu.edu/

    Version 1.0

    *Description:*
        This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes: air_conditioner, car_horn, 
        children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, and street_music. The classes are 
        drawn from the urban sound taxonomy described in the following article, which also includes a detailed description of 
        the dataset and how it was compiled:

        .. code-block:: latex
            J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
            22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

        All excerpts are taken from field recordings uploaded to www.freesound.org. The files are pre-sorted into ten folds
        (folders named fold1-fold10) to help in the reproduction of and comparison with the automatic classification results
        reported in the article above.

        In addition to the sound excerpts, a CSV file containing metadata about each excerpt is also provided.


    *Audio Files Included:*
        8732 audio files of urban sounds (see description above) in WAV format. The sampling rate, bit depth, and number of 
        channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file).


    *Meta-data Files Included:*
    ------------------------

        UrbanSound8k.csv

        This file contains meta-data information about every audio file in the dataset. This includes:

        * slice_file_name: 
        The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav, where:
        [fsID] = the Freesound ID of the recording from which this excerpt (slice) is taken
        [classID] = a numeric identifier of the sound class (see description of classID below for further details)
        [occurrenceID] = a numeric identifier to distinguish different occurrences of the sound within the original recording
        [sliceID] = a numeric identifier to distinguish different slices taken from the same occurrence

        * fsID:
        The Freesound ID of the recording from which this excerpt (slice) is taken

        * start
        The start time of the slice in the original Freesound recording

        * end:
        The end time of slice in the original Freesound recording

        * salience:
        A (subjective) salience rating of the sound. 1 = foreground, 2 = background.

        * fold:
        The fold number (1-10) to which this file has been allocated.

        * classID:
        A numeric identifier of the sound class:
        0 = air_conditioner
        1 = car_horn
        2 = children_playing
        3 = dog_bark
        4 = drilling
        5 = engine_idling
        6 = gun_shot
        7 = jackhammer
        8 = siren
        9 = street_music

        * class:
        The class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, 
        siren, street_music.


    *Please Acknowledge EigenScape in Academic Research:*

        When UrbanSound8K is used for academic research, we would highly appreciate it if scientific publications of works 
        partly based on the UrbanSound8K dataset cite the following publication:

        .. code-block:: latex
            J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
            22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

        The creation of this dataset was supported by a seed grant by NYU's Center for Urban Science and Progress (CUSP).


    *Conditions of Use*

        Dataset compiled by Justin Salamon, Christopher Jacoby and Juan Pablo Bello. All files are excerpts of recordings
        uploaded to www.freesound.org. Please see FREESOUNDCREDITS.txt for an attribution list.
        
        The UrbanSound8K dataset is offered free of charge for non-commercial use only under the terms of the Creative Commons
        Attribution Noncommercial License (by-nc), version 3.0: http://creativecommons.org/licenses/by-nc/3.0/
        
        The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including 
        without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or 
        completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, NYU is not 
        liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of
        the UrbanSound8K dataset or any part of it.

    *Feedback*

        Please help us improve UrbanSound8K by sending your feedback to: justin.salamon@nyu.edu
        In case of a problem report please include as many details as possible.

"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io

from functools import lru_cache
from pydub import AudioSegment
from pydub.playback import play
import threading
import time
from ipywidgets import FloatSlider, Button, VBox, HBox, Checkbox, Label, Output
import matplotlib.pyplot as plt
import ipywidgets as widgets
import simpleaudio as sa
import seaborn as sns
import numpy as np
from tqdm import tqdm

BIBTEX = """
@inproceedings{Salamon:UrbanSound:ACMMM:14,
	Address = {Orlando, FL, USA},
	Author = {Salamon, J. and Jacoby, C. and Bello, J. P.},
	Booktitle = {22nd {ACM} International Conference on Multimedia (ACM-MM'14)},
	Month = {Nov.},
	Pages = {1041--1044},
	Title = {A Dataset and Taxonomy for Urban Sound Research},
	Year = {2014}}
"""
REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="UrbanSound8K.tar.gz",
        url="https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1",
        checksum="9aa69802bbf37fb986f71ec1483a196e",
        unpack_directories=["UrbanSound8K"],
    )
}

LICENSE_INFO = "Creative Commons Attribution Non Commercial 4.0 International"


class Clip(core.Clip):
    """urbansound8k Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        class_id (int): integer representation of the class label (0-9). See Dataset Info in the documentation for mapping
        class_label (str): string class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
        clip_id (str): clip id
        fold (int): fold number (1-10) to which this clip is allocated. Use these folds for cross validation
        freesound_end_time (float): end time in seconds of the clip in the original freesound recording
        freesound_id (str): ID of the freesound.org recording from which this clip was taken
        freesound_start_time (float): start time in seconds of the clip in the original freesound recording
        salience (int): annotator estimate of class sailence in the clip: 1 = foreground, 2 = background
        slice_file_name (str): The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav
            Please see the Dataset Info in the soundata documentation for further details
        tags (soundata.annotations.Tags): tag (label) of the clip + confidence. In UrbanSound8K every clip has one tag
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def slice_file_name(self):
        """The clip's slice filename.

        Returns:
            * str - The name of the audio file. The name takes the following format: [fsID]-[classID]-[occurrenceID]-[sliceID].wav

        """
        return self._clip_metadata.get("slice_file_name")

    @property
    def freesound_id(self):
        """The clip's Freesound ID.

        Returns:
            * str - ID of the freesound.org recording from which this clip was taken

        """
        return self._clip_metadata.get("freesound_id")

    @property
    def freesound_start_time(self):
        """The clip's start time in Freesound.

        Returns:
            * float - start time in seconds of the clip in the original freesound recording

        """
        return self._clip_metadata.get("freesound_start_time")

    @property
    def freesound_end_time(self):
        """The clip's end time in Freesound.

        Returns:
            * float - end time in seconds of the clip in the original freesound recording

        """
        return self._clip_metadata.get("freesound_end_time")

    @property
    def salience(self):
        """The clip's salience.

        Returns:
            * int - annotator estimate of class sailence in the clip: 1 = foreground, 2 = background

        """
        return self._clip_metadata.get("salience")

    @property
    def fold(self):
        """The clip's fold.

        Returns:
            * int - fold number (1-10) to which this clip is allocated. Use these folds for cross validation

        """
        return self._clip_metadata.get("fold")

    @property
    def class_id(self):
        """The clip's class id.

        Returns:
            * int - integer representation of the class label (0-9). See Dataset Info in the documentation for mapping

        """
        return self._clip_metadata.get("class_id")

    @property
    def class_label(self):
        """The clip's class label.

        Returns:
            * str - string class name: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music

        """
        return self._clip_metadata.get("class_label")

    @property
    def tags(self):
        """The clip's tags.

        Returns:
            * annotations.Tags - tag (label) of the clip + confidence. In UrbanSound8K every clip has one tag

        """
        return annotations.Tags(
            [self._clip_metadata.get("class_label")], "open", np.array([1.0])
        )

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path, tags=self.tags, metadata=self._clip_metadata
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=44100) -> Tuple[np.ndarray, float]:
    """Load a UrbanSound8K audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, 44100 Hz by default.
            If different from file's sample rate it will be resampled on load.
            Use None to load the file using its original sample rate (sample rate
            varies from file to file).

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The urbansound8k dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="urbansound8k",
            clip_class=Clip,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata", "UrbanSound8K.csv")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        with open(metadata_path, "r") as fhandle:
            reader = csv.reader(fhandle, delimiter=",")
            raw_data = []
            for line in reader:
                if line[0] != "slice_file_name":
                    raw_data.append(line)

        metadata_index = {}
        for line in raw_data:
            clip_id = line[0].replace(".wav", "")

            metadata_index[clip_id] = {
                "slice_file_name": line[0],
                "freesound_id": line[1],
                "freesound_start_time": float(line[2]),
                "freesound_end_time": float(line[3]),
                "salience": int(line[4]),
                "fold": int(line[5]),
                "class_id": int(line[6]),
                "class_label": line[7],
            }

        return metadata_index


    @lru_cache(maxsize=None)  # Setting maxsize to None for an unbounded cache
    def compute_clip_statistics(self):
        durations = [
            len(self.clip(c_id).audio[0]) / self.clip(c_id).audio[1]
            for c_id in tqdm(list(self._index["clips"].keys())[:100], desc="Calculating durations")
        ]

        # Calculate statistics
        total_duration = sum(durations)
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        std_deviation = np.std(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)

        return {
            "durations": durations,
            "total_duration": total_duration,
            "mean_duration": mean_duration,
            "median_duration": median_duration,
            "std_deviation": std_deviation,
            "min_duration": min_duration,
            "max_duration": max_duration,
        }

    def plot_clip_durations(self):
        stats = self.compute_clip_statistics()
        durations = stats['durations']
        total_duration = stats['total_duration']
        mean_duration = stats['mean_duration']
        median_duration = stats['median_duration']
        std_deviation = stats['std_deviation']
        min_duration = stats['min_duration']
        max_duration = stats['max_duration']

        # Determine unit conversion (seconds or minutes)
        convert_to_minutes = mean_duration > 60 or median_duration > 60
        conversion_factor = 60 if convert_to_minutes else 1
        unit = "minutes" if convert_to_minutes else "seconds"

        durations = [d / conversion_factor for d in durations]
        mean_duration /= conversion_factor
        median_duration /= conversion_factor
        total_duration /= conversion_factor

        # Convert total duration to hours if it exceeds 120 minutes
        if convert_to_minutes:
            if total_duration > 120:
                total_duration /= 60
                total_duration_unit = "hours"
            else:
                total_duration_unit = "minutes"
        else:
            if total_duration > 120:
                total_duration /= 60
                total_duration_unit = "minutes"
            else:
                total_duration_unit = "seconds"

        # Define the base colors
        base_colors = ['#404040', '#126782', '#C9C9C9']

        # Create the main figure and the two axes
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, frame_on=False)
        ax2.axis('off')

        # Histogram with base color for bars
        n, bins, patches = ax1.hist(durations, bins=30, color=base_colors[0], edgecolor='black')

        mean_bin = np.digitize(mean_duration, bins) - 1  # Correct bin for the mean
        median_bin = np.digitize(median_duration, bins) - 1  # Correct bin for the median

        # Set the mean and median bins colors if they are within the range
        if 0 <= mean_bin < len(patches):
            patches[mean_bin].set_fc(base_colors[1])
        if 0 <= median_bin < len(patches):
            patches[median_bin].set_fc(base_colors[2])

        # Lines and text for mean and median
        ax1.axvline(mean_duration, color=base_colors[1], linestyle='dashed', linewidth=1)
        ax1.text(mean_duration + 0.2, max(n) * 0.9, 'Mean', color=base_colors[1])
        ax1.axvline(median_duration, color=base_colors[2], linestyle='dashed', linewidth=1)
        ax1.text(median_duration + 0.2, max(n) * 0.8, 'Median', color=base_colors[2])

        ax1.set_title('Distribution of Clip Durations', fontsize=8)
        ax1.set_xlabel(f'Duration ({unit})', fontsize=8)
        ax1.set_ylabel('Number of Clips', fontsize=8)
        ax1.grid(axis='y', alpha=0.75)

        # Text box for statistics
        analysis_results = (
            f"$\\bf{{Total\\ duration:}}$ {total_duration:.2f} {total_duration_unit}\n"
            f"$\\bf{{Mean\\ duration:}}$ {mean_duration:.2f} {unit}\n"
            f"$\\bf{{Median\\ duration:}}$ {median_duration:.2f} {unit}\n"
            f"$\\bf{{Standard\\ Deviation:}}$ {std_deviation:.2f} {unit}\n"
            f"$\\bf{{Min\\ Duration:}}$ {min_duration:.2f} {unit}\n"
            f"$\\bf{{Max\\ Duration:}}$ {max_duration:.2f} {unit}\n"
            f"$\\bf{{Total\\ Clips:}}$ {len(self._index['clips'])}"
        )
        ax2.text(0.1, 0.4, analysis_results, transform=ax2.transAxes, fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_hierarchical_distribution(self):

        def plot_distribution(data, title, x_label, y_label, subplot_position):
            my_palette = sns.color_palette("light:b", as_cmap=False)
            my_palette = ['#404040', '#126782', '#C9C9C9']
            sns.countplot(y=data, order=pd.value_counts(data).index, palette=my_palette, ax=axes[subplot_position])
            axes[subplot_position].set_title(title, fontsize=8)
            axes[subplot_position].set_xlabel(x_label, fontsize=6)
            axes[subplot_position].set_ylabel(y_label, fontsize=6)
            axes[subplot_position].tick_params(axis='both', which='major', labelsize=6)

            ax = axes[subplot_position]
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            for p in ax.patches:
                ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height()/2),
                            ha='left', va='center', xytext=(3, 0), textcoords='offset points', fontsize=6)

        # Determine the number of plots
        plot_count = 1
        if 'subdatasets' in self._metadata:
            plot_count += 1

        layer = 0
        while f'subdataset_layer_{layer}' in self._metadata:
            plot_count += 1
            layer += 1

        plt.figure(figsize=(6 * plot_count, 4))
        axes = [plt.subplot(1, plot_count, i+1) for i in range(plot_count)]

        # Plot Event Distribution
        events = [label for clip_id in self._index["clips"] for label in self.clip(clip_id).events.labels]
        plot_distribution(events, 'Event Distribution in the Dataset', 'Count', 'Event', 0)

        # Plot Subclasses Distribution and then Hierarchical layers
        subplot_position = 1  # We've already plotted events at position 0
        if 'subdatasets' in self._metadata:
            subclasses = [self._metadata[clip_id]['subdataset'] for clip_id in self._index["clips"]]
            plot_distribution(subclasses, 'Subclass Distribution in the Dataset', 'Count', 'Subclass', subplot_position)
            subplot_position += 1
        else:
            print("Subclass information not available.")

        layer = 0
        while f'subdataset_layer_{layer}' in self._metadata:
            layer_data = [self._metadata[clip_id][f'subdataset_layer_{layer}'] for clip_id in self._index["clips"]]
            plot_distribution(layer_data, f'Subdataset Layer {layer} Distribution in the Dataset', 'Count', f'Subdataset Layer {layer}', subplot_position)
            layer += 1
            subplot_position += 1

        plt.tight_layout()
        plt.show()
        print("\n")

    def explore_dataset(self, clip_id=None):
        """Explore the dataset for a given clip_id or a random clip if clip_id is None."""
        
        # Interactive checkboxes for user input
        event_dist_check = Checkbox(value=True, description='Class Distribution')
        dataset_analysis_check = Checkbox(value=False, description='Statistics (Computational)')
        audio_plot_check = Checkbox(value=True, description='Audio Visualization')
        
        # Button to execute plotting based on selected checkboxes
        plot_button = Button(description="Explore Dataset")
        output = Output()

        # Loader HTML widget
        loader = widgets.HTML(
            value='<img src="https://example.com/loader.gif" />', # Replace with the path to your loader GIF
            placeholder='Some HTML',
            description='Status:',
        )
        
        # Button callback function
        def on_button_clicked(b):
            output.clear_output(wait=True)  # Clear the previous outputs
            with output:
                display(loader)  # Display the loader
                # Update the page with a loading message
                loader.value = "<p style='font-size:15px;'>Rendering plots...please wait!</p>"
                
            # This allows the loader to be displayed before starting heavy computations
            time.sleep(0.1)

            # Now perform the computations and update the output accordingly
            with output:
                if event_dist_check.value:
                    print("Analyzing event distribution... Please wait.")
                    self.plot_hierarchical_distribution()

                if dataset_analysis_check.value:
                    print("Conducting dataset analysis... Please wait.")
                    self.plot_clip_durations()

                if audio_plot_check.value:
                    print("Generating audio plot... Please wait.")
                    self.visualize_audio(clip_id)
                
                # Remove the loader after the content is loaded
                loader.value = "<p style='font-size:15px;'>Completed the processes!</p>"
                
        plot_button.on_click(on_button_clicked)
        
        # Provide user instructions
        intro_text = "Welcome to the Dataset Explorer!\nSelect the options below to explore your dataset:"
        
        # Display checkboxes, button, and output widget for user interaction
        display(VBox([widgets.HTML(value=intro_text), HBox([event_dist_check, dataset_analysis_check, audio_plot_check]), plot_button, output]))
        
    def visualize_audio(self, clip_id):

        if clip_id is None:  # Use the local variable
            clip_id = np.random.choice(list(self._index["clips"].keys()))  # Modify the local variable
        clip = self.clip(clip_id)  # Use the local variable
        
        stop_event = threading.Event()
        current_time_lock = threading.Lock()

        audio, sr = clip.audio
        duration = len(audio) / sr

        if audio.max() > 1 or audio.min() < -1:
            audio = audio / np.max(np.abs(audio))

        # Convert to int16 for playback
        audio_playback = np.int16(audio * 32767)

        audio_segment = AudioSegment(
            audio_playback.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )

        if duration > 60:
            print("Audio is longer than 1 minute. Displaying only the first 1 minute.")
            audio = audio[:int(60 * sr)]
            duration = 60

        # Compute the Mel spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Update the figure and axes to show both plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4))
        # Plotting the waveform
        ax1.plot(np.linspace(0, duration, len(audio)), audio)
        ax1.set_title(f"Audio waveform for clip: {clip_id}", fontsize=8)
        ax1.set_xlabel('Time (s)', fontsize=8)
        ax1.set_ylabel('Amplitude', fontsize=8)
        ax1.set_xlim(0, duration)
        line1, = ax1.plot([0, 0], [min(audio), max(audio)], color='#C9C9C9')

        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontsize(8)

        # Plotting the Mel spectrogram
        im = librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_title('Mel spectrogram', fontsize=8)
        ax2.set_xlim(0, duration)
        line2, = ax2.plot([0, 0], ax2.get_ylim(), color='#126782', linestyle='--')

        # Reduce font size for time and mel axis labels
        ax2.set_xlabel('Time (s)', fontsize=8)
        ax2.set_ylabel('Hz', fontsize=8)

        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontsize(8)

        fig.subplots_adjust(right=0.8)  # Adjust the right side of the layout

        # Add the colorbar
        bbox = ax2.get_position()
# Add the colorbar axes with the same y position and height as the Mel spectrogram
        cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])        
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('dB', rotation=270, labelpad=15, fontsize=8)
        cbar.ax.tick_params(labelsize=8)  # Set the size of the tick labels

        # Ensure the tight_layout does not overlap the axes with colorbar
        plt.tight_layout(rect=[0, 0, 0.8, 1])

        playing = [False]
        current_time = [0.0]
        play_thread = [None]

        def play_segment(start_time):
            try:
                segment_start = start_time * 1000  # convert to milliseconds
                segment_end = segment_start + 60 * 1000
                segment = audio_segment[segment_start:segment_end]

                play_obj = sa.play_buffer(segment.raw_data, 1, 2, sr)

                while play_obj.is_playing():
                    if stop_event.is_set():
                        play_obj.stop()
                        break
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in play_segment: {e}")

        def update_line():
            try:
                step = 0.1
                while playing[0]:
                    with current_time_lock:
                        current_time[0] += step
                        if current_time[0] > duration:
                            playing[0] = False
                            current_time[0] = 0.0
                    line1.set_xdata([current_time[0], current_time[0]])
                    line2.set_xdata([current_time[0], current_time[0]])
                    fig.canvas.draw_idle()
                    time.sleep(step)
            except Exception as e:
                print(f"Error in update_line: {e}")

        def on_play_pause_clicked(b):
            nonlocal play_thread
            if playing[0]:
                stop_event.set()
                if play_thread[0].is_alive():
                    play_thread[0].join()
                playing[0] = False
                play_pause_button.description = "► Play"
            else:
                stop_event.clear()
                playing[0] = True
                play_pause_button.description = "❚❚ Pause"
                play_thread[0] = threading.Thread(target=play_segment, args=(current_time[0],))
                play_thread[0].start()
                update_line_thread = threading.Thread(target=update_line)
                update_line_thread.start()

        def on_reset_clicked(b):
            nonlocal play_thread
            if playing[0]:
                stop_event.set()
                if play_thread[0].is_alive():
                    play_thread[0].join()
                playing[0] = False
            with current_time_lock:
                current_time[0] = 0.0
            line1.set_xdata([0, 0])
            line2.set_xdata([0, 0])
            slider.value = 0.0
            play_pause_button.description = "► Play"
            fig.canvas.draw_idle()

        def on_slider_changed(change):
            nonlocal play_thread
            if playing[0]:
                stop_event.set()
                if play_thread[0].is_alive():
                    play_thread[0].join()
            with current_time_lock:
                current_time[0] = change.new
            line1.set_xdata([current_time[0], current_time[0]])
            line2.set_xdata([current_time[0], current_time[0]])
            fig.canvas.draw_idle()

        slider = FloatSlider(value=0.0, min=0.0, max=duration, step=0.1, description='Time (s)')
        slider.observe(on_slider_changed, names='value')

        play_pause_button = Button(description="► Play")
        play_pause_button.on_click(on_play_pause_clicked)

        reset_button = Button(description="Reset")
        reset_button.on_click(on_reset_clicked)

                # Set the description for the slider that indicates its purpose.
        slider.description = 'Seek:'
        slider.tooltip = 'Drag the slider to a specific point in the audio to play from that time.'

        # You can also add a label above the slider for clarity, if the UI framework you are using supports it.
        slider_label = Label('Drag the slider to navigate through the audio:')

        # Now, display the slider with its label.
        display(VBox([HBox([play_pause_button, reset_button]), slider_label, slider]))