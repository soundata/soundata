"""Core soundata classes
"""
import json
import os
import random
import types
from typing import Any

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
import pandas as pd
import librosa
import numpy as np

from soundata import download_utils
from soundata import validate

MAX_STR_LEN = 100
DOCS_URL = "https://soundata.readthedocs.io/en/stable/source/soundata.html"
DISCLAIMER = """
******************************************************************************************
DISCLAIMER: soundata is a software package with its own license which is independent from
this dataset's license. We don not take responsibility for possible inaccuracies in the
license information provided in soundata. It is the user's responsibility to be informed
and respect the dataset's license.
******************************************************************************************
"""

##### decorators ######


class cached_property(object):
    """Cached propery decorator

    A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76

    """

    def __init__(self, func):
        self.__doc__ = getattr(func, "__doc__")
        self.func = func

    def __get__(self, obj: Any, cls: type) -> Any:
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def docstring_inherit(parent):
    """Decorator function to inherit docstrings from the parent class.

    Adds documented Attributes from the parent to the child docs.

    """

    def inherit(obj):
        spaces = "    "
        if not str(obj.__doc__).__contains__("Attributes:"):
            obj.__doc__ += "\n" + spaces + "Attributes:\n"
        obj.__doc__ = str(obj.__doc__).rstrip() + "\n"
        for attribute in parent.__doc__.split("Attributes:\n")[-1].lstrip().split("\n"):
            obj.__doc__ += spaces * 2 + str(attribute).lstrip().rstrip() + "\n"

        return obj

    return inherit


def copy_docs(original):
    """
    Decorator function to copy docs from one function to another
    """

    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper


##### Core Classes #####


class Dataset(object):
    """soundata Dataset class

    Attributes:
        data_home (str): path where soundata will look for the dataset
        name (str): the identifier of the dataset
        bibtex (str or None): dataset citation/s in bibtex format
        remotes (dict or None): data to be downloaded
        readme (str): information about the dataset
        clip (function): a function mapping a clip_id to a soundata.core.Clip
        clipgroup (function): a function mapping a clipgroup_id to a soundata.core.Clipgroup

    """

    def __init__(
        self,
        data_home=None,
        name=None,
        clip_class=None,
        clipgroup_class=None,
        bibtex=None,
        remotes=None,
        download_info=None,
        license_info=None,
        custom_index_path=None,
    ):
        """Dataset init method

        Args:
            data_home (str or None): path where soundata will look for the dataset
            name (str or None): the identifier of the dataset
            clip_class (soundata.core.Clip or None): a Clip class
            clipgroup_class (soundata.core.Clipgroup or None): a Clipgroup class
            bibtex (str or None): dataset citation/s in bibtex format
            remotes (dict or None): data to be downloaded
            download_info (str or None): download instructions or caveats
            license_info (str or None): license of the dataset
            custom_index_path (str or None): overwrites the default index path for remote indexes

        """
        self.name = name
        self.data_home = self.default_path if data_home is None else data_home
        if custom_index_path:
            self.index_path = os.path.join(self.data_home, custom_index_path)
            self.remote_index = True
        else:
            self.index_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "datasets/indexes",
                "{}_index.json".format(self.name),
            )
            self.remote_index = False
        self._clip_class = clip_class
        self._clipgroup_class = clipgroup_class
        self.bibtex = bibtex
        self.remotes = remotes
        self._download_info = download_info
        self._license_info = license_info
        self.readme = "{}#module-soundata.datasets.{}".format(DOCS_URL, self.name)

        # this is a hack to be able to have dataset-specific docstrings
        self.clip = lambda clip_id: self._clip(clip_id)
        self.clip.__doc__ = self._clip_class.__doc__  # set the docstring
        self.clipgroup = lambda clipgroup_id: self._clipgroup(clipgroup_id)
        self.clipgroup.__doc__ = self._clipgroup_class.__doc__  # set the docstring

    def __repr__(self):
        repr_string = "The {} dataset\n".format(self.name)
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n\n\n"
        repr_string += "Call the .cite method for bibtex citations.\n"
        repr_string += "-" * MAX_STR_LEN
        repr_string += "\n\n\n"
        if self._clip_class is not None:
            repr_string += self.clip.__doc__
            repr_string += "-" * MAX_STR_LEN
            repr_string += "\n"
        if self._clipgroup_class is not None:
            repr_string += self.clipgroup.__doc__
            repr_string += "-" * MAX_STR_LEN
            repr_string += "\n"

        return repr_string

    @cached_property
    def _index(self):
        if self.remote_index and not os.path.exists(self.index_path):
            raise FileNotFoundError(
                "This dataset's index is not available locally. You may need to first run .download()"
            )
        with open(self.index_path) as fhandle:
            index = json.load(fhandle)
        return index

    @cached_property
    def _metadata(self):
        return None

    @property
    def default_path(self):
        """Get the default path for the dataset

        Returns:
            str: Local path to the dataset

        """
        sound_datasets_dir = os.path.join(os.getenv("HOME", "/tmp"), "sound_datasets")
        return os.path.join(sound_datasets_dir, self.name)

    def _clip(self, clip_id):
        """Load a clip by clip_id.

        Hidden helper function that gets called as a lambda.

        Args:
            clip_id (str): clip id of the clip

        Returns:
           Clip: a Clip object

        """
        if self._clip_class is None:
            raise AttributeError("This dataset does not have clips")
        else:
            return self._clip_class(
                clip_id, self.data_home, self.name, self._index, lambda: self._metadata
            )

    def _clipgroup(self, clipgroup_id):
        """Load a clipgroup by clipgroup_id.

        Hidden helper function that gets called as a lambda.

        Args:
            clipgroup_id (str): clipgroup id of the clipgroup

        Returns:
            ClipGroup: an instance of this dataset's ClipGroup object

        """
        if self._clipgroup_class is None:
            raise AttributeError("This dataset does not have clipgroups")
        else:
            return self._clipgroup_class(
                clipgroup_id,
                self.data_home,
                self.name,
                self._index,
                self._clip_class,
                lambda: self._metadata,
            )

    def load_clips(self):
        """Load all clips in the dataset

        Returns:
            dict:
                {`clip_id`: clip data}

        Raises:
            NotImplementedError: If the dataset does not support Clips

        """
        return {clip_id: self.clip(clip_id) for clip_id in self.clip_ids}

    def load_clipgroups(self):
        """Load all clipgroups in the dataset

        Returns:
            dict:
                {`clipgroup_id`: clipgroup data}

        Raises:
            NotImplementedError: If the dataset does not support Clipgroups

        """
        return {
            clipgroup_id: self.clipgroup(clipgroup_id)
            for clipgroup_id in self.clipgroup_ids
        }

    def choice_clip(self):
        """Choose a random clip

        Returns:
            Clip: a Clip object instantiated by a random clip_id

        """
        return self.clip(random.choice(self.clip_ids))

    def choice_clipgroup(self):
        """Choose a random clipgroup

        Returns:
            Clipgroup: a Clipgroup object instantiated by a random clipgroup_id

        """
        return self.clipgroup(random.choice(self.clipgroup_ids))

    def cite(self):
        """
        Print the reference
        """
        print("========== BibTeX ==========")
        print(self.bibtex)

    def license(self):
        """
        Print the license
        """
        print("========== License ==========")
        print(self._license_info)
        print(DISCLAIMER)

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
        """Download data to `save_dir` and optionally print a message.

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        download_utils.downloader(
            self.data_home,
            remotes=self.remotes,
            partial_download=partial_download,
            info_message=self._download_info,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

    @cached_property
    def clip_ids(self):
        """Return clip ids

        Returns:
            list: A list of clip ids

        """
        if "clips" not in self._index:
            raise AttributeError("This dataset does not have clips")
        return list(self._index["clips"].keys())

    @cached_property
    def clipgroup_ids(self):
        """Return clip ids

        Returns:
            list: A list of clip ids

        """
        if "clipgroups" not in self._index:
            raise AttributeError("This dataset does not have clipgroups")
        return list(self._index["clipgroups"].keys())

    def validate(self, verbose=True):
        """Validate if the stored dataset is a valid version

        Args:
            verbose (bool): If False, don't print output

        Returns:
            * list - files in the index but are missing locally
            * list - files which have an invalid checksum

        """
        missing_files, invalid_checksums = validate.validator(
            self._index, self.data_home, verbose=verbose
        )
        return missing_files, invalid_checksums

    @lru_cache(maxsize=None)  # Setting maxsize to None for an unbounded cache
    def compute_clip_statistics(self):
        durations = [
            len(self.clip(c_id).audio[0]) / self.clip(c_id).audio[1]
            for c_id in tqdm(
                list(self._index["clips"].keys())[:100], desc="Calculating durations"
            )
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
        durations = stats["durations"]
        total_duration = stats["total_duration"]
        mean_duration = stats["mean_duration"]
        median_duration = stats["median_duration"]
        std_deviation = stats["std_deviation"]
        min_duration = stats["min_duration"]
        max_duration = stats["max_duration"]

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
        base_colors = ["#404040", "#126782", "#C9C9C9"]

        # Create the main figure and the two axes
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, frame_on=False)
        ax2.axis("off")

        # Histogram with base color for bars
        n, bins, patches = ax1.hist(
            durations, bins=30, color=base_colors[0], edgecolor="black"
        )

        mean_bin = np.digitize(mean_duration, bins) - 1  # Correct bin for the mean
        median_bin = (
            np.digitize(median_duration, bins) - 1
        )  # Correct bin for the median

        # Set the mean and median bins colors if they are within the range
        if 0 <= mean_bin < len(patches):
            patches[mean_bin].set_fc(base_colors[1])
        if 0 <= median_bin < len(patches):
            patches[median_bin].set_fc(base_colors[2])

        # Lines and text for mean and median
        ax1.axvline(
            mean_duration, color=base_colors[1], linestyle="dashed", linewidth=1
        )
        ax1.text(mean_duration + 0.2, max(n) * 0.9, "Mean", color=base_colors[1])
        ax1.axvline(
            median_duration, color=base_colors[2], linestyle="dashed", linewidth=1
        )
        ax1.text(median_duration + 0.2, max(n) * 0.8, "Median", color=base_colors[2])

        ax1.set_title("Distribution of Clip Durations", fontsize=8)
        ax1.set_xlabel(f"Duration ({unit})", fontsize=8)
        ax1.set_ylabel("Number of Clips", fontsize=8)
        ax1.grid(axis="y", alpha=0.75)

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
            my_palette = ["#404040", "#126782", "#C9C9C9"]
            sns.countplot(
                y=data,
                order=pd.value_counts(data).index,
                palette=my_palette,
                ax=axes[subplot_position],
            )
            axes[subplot_position].set_title(title, fontsize=8)
            axes[subplot_position].set_xlabel(x_label, fontsize=6)
            axes[subplot_position].set_ylabel(y_label, fontsize=6)
            axes[subplot_position].tick_params(axis="both", which="major", labelsize=6)

            ax = axes[subplot_position]
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_width())}",
                    (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha="left",
                    va="center",
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=6,
                )

        # Determine the number of plots
        plot_count = 1
        if "subdatasets" in self._metadata:
            plot_count += 1

        layer = 0
        while f"subdataset_layer_{layer}" in self._metadata:
            plot_count += 1
            layer += 1

        plt.figure(figsize=(6 * plot_count, 4))
        axes = [plt.subplot(1, plot_count, i + 1) for i in range(plot_count)]

        # Plot Event Distribution
        events = []
        for clip_id in self._index["clips"]:
            clip = self.clip(clip_id)
            if hasattr(clip, "tags") and hasattr(clip.tags, "labels"):
                events.extend(clip.tags.labels)
            elif hasattr(clip, "events") and hasattr(clip.events, "labels"):
                events.extend(clip.events.labels)

        # events = [label for clip_id in self._index["clips"] for label in self.clip(clip_id).tags.labels]
        plot_distribution(
            events, "Event Distribution in the Dataset", "Count", "Event", 0
        )

        # Plot Subclasses Distribution and then Hierarchical layers
        subplot_position = 1  # We've already plotted events at position 0
        if "subdatasets" in self._metadata:
            subclasses = [
                self._metadata[clip_id]["subdataset"]
                for clip_id in self._index["clips"]
            ]
            plot_distribution(
                subclasses,
                "Subclass Distribution in the Dataset",
                "Count",
                "Subclass",
                subplot_position,
            )
            subplot_position += 1
        else:
            print("Subclass information not available.")

        layer = 0
        while f"subdataset_layer_{layer}" in self._metadata:
            layer_data = [
                self._metadata[clip_id][f"subdataset_layer_{layer}"]
                for clip_id in self._index["clips"]
            ]
            plot_distribution(
                layer_data,
                f"Subdataset Layer {layer} Distribution in the Dataset",
                "Count",
                f"Subdataset Layer {layer}",
                subplot_position,
            )
            layer += 1
            subplot_position += 1

        plt.tight_layout()
        plt.show()
        print("\n")

    def explore_dataset(self, clip_id=None):
        """Explore the dataset for a given clip_id or a random clip if clip_id is None."""

        # Interactive checkboxes for user input
        event_dist_check = Checkbox(value=True, description="Class Distribution")
        dataset_analysis_check = Checkbox(
            value=False, description="Statistics (Computational)"
        )
        audio_plot_check = Checkbox(value=True, description="Audio Visualization")

        # Button to execute plotting based on selected checkboxes
        plot_button = Button(description="Explore Dataset")
        output = Output()

        # Loader HTML widget
        loader = widgets.HTML(
            value='<img src="https://example.com/loader.gif" />',  # Replace with the path to your loader GIF
            placeholder="Some HTML",
            description="Status:",
        )

        # Button callback function
        def on_button_clicked(b):
            output.clear_output(wait=True)  # Clear the previous outputs
            with output:
                display(loader)  # Display the loader
                # Update the page with a loading message
                loader.value = (
                    "<p style='font-size:15px;'>Rendering plots...please wait!</p>"
                )

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
        display(
            VBox(
                [
                    widgets.HTML(value=intro_text),
                    HBox([event_dist_check, dataset_analysis_check, audio_plot_check]),
                    plot_button,
                    output,
                ]
            )
        )

    def visualize_audio(self, clip_id):
        if clip_id is None:  # Use the local variable
            clip_id = np.random.choice(
                list(self._index["clips"].keys())
            )  # Modify the local variable
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
            audio_playback.tobytes(), frame_rate=sr, sample_width=2, channels=1
        )

        if duration > 60:
            print("Audio is longer than 1 minute. Displaying only the first 1 minute.")
            audio = audio[: int(60 * sr)]
            duration = 60

        # Compute the Mel spectrogram
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        log_S = librosa.power_to_db(S, ref=np.max)

        # Update the figure and axes to show both plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4))
        # Plotting the waveform
        ax1.plot(np.linspace(0, duration, len(audio)), audio)
        ax1.set_title(f"Audio waveform for clip: {clip_id}", fontsize=8)
        ax1.set_xlabel("Time (s)", fontsize=8)
        ax1.set_ylabel("Amplitude", fontsize=8)
        ax1.set_xlim(0, duration)
        (line1,) = ax1.plot([0, 0], [min(audio), max(audio)], color="#C9C9C9")

        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontsize(8)

        # Plotting the Mel spectrogram
        im = librosa.display.specshow(log_S, sr=sr, x_axis="time", y_axis="mel", ax=ax2)
        ax2.set_title("Mel spectrogram", fontsize=8)
        ax2.set_xlim(0, duration)
        (line2,) = ax2.plot([0, 0], ax2.get_ylim(), color="#126782", linestyle="--")

        # Reduce font size for time and mel axis labels
        ax2.set_xlabel("Time (s)", fontsize=8)
        ax2.set_ylabel("Hz", fontsize=8)

        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontsize(8)

        fig.subplots_adjust(right=0.8)  # Adjust the right side of the layout

        # Add the colorbar
        bbox = ax2.get_position()
        # Add the colorbar axes with the same y position and height as the Mel spectrogram
        cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("dB", rotation=270, labelpad=15, fontsize=8)
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
                play_thread[0] = threading.Thread(
                    target=play_segment, args=(current_time[0],)
                )
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

        slider = FloatSlider(
            value=0.0, min=0.0, max=duration, step=0.1, description="Time (s)"
        )
        slider.observe(on_slider_changed, names="value")

        play_pause_button = Button(description="► Play")
        play_pause_button.on_click(on_play_pause_clicked)

        reset_button = Button(description="Reset")
        reset_button.on_click(on_reset_clicked)

        # Set the description for the slider that indicates its purpose.
        slider.description = "Seek:"
        slider.tooltip = (
            "Drag the slider to a specific point in the audio to play from that time."
        )

        # You can also add a label above the slider for clarity, if the UI framework you are using supports it.
        slider_label = Label("Drag the slider to navigate through the audio:")

        # Now, display the slider with its label.
        display(VBox([HBox([play_pause_button, reset_button]), slider_label, slider]))


class Clip(object):
    """Clip base class

    See the docs for each dataset loader's Clip class for details

    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        """Clip init method. Sets boilerplate attributes, including:

        - ``clip_id``
        - ``_dataset_name``
        - ``_data_home``
        - ``_clip_paths``
        - ``_clip_metadata``

        Args:
            clip_id (str): clip id
            data_home (str): path where soundata will look for the dataset
            dataset_name (str): the identifier of the dataset
            index (dict): the dataset's file index
            metadata (function or None): a function returning a dictionary of metadata or None

        """
        if clip_id not in index["clips"]:
            raise ValueError(
                "{} is not a valid clip_id in {}".format(clip_id, dataset_name)
            )

        self.clip_id = clip_id
        self._dataset_name = dataset_name

        self._data_home = data_home
        self._clip_paths = index["clips"][clip_id]
        self._metadata = metadata

    @property
    def _clip_metadata(self):
        metadata = self._metadata()
        if metadata and self.clip_id in metadata:
            return metadata[self.clip_id]
        elif metadata:
            return metadata
        raise AttributeError("This Clip does not have metadata.")

    def __repr__(self):
        properties = [v for v in dir(self.__class__) if not v.startswith("_")]
        attributes = [
            v for v in dir(self) if not v.startswith("_") and v not in properties
        ]

        repr_str = "Clip(\n"

        for attr in attributes:
            val = getattr(self, attr)
            if isinstance(val, str):
                if len(val) > MAX_STR_LEN:
                    val = "...{}".format(val[-MAX_STR_LEN:])
                val = '"{}"'.format(val)
            repr_str += "  {}={},\n".format(attr, val)

        for prop in properties:
            val = getattr(self.__class__, prop)
            if isinstance(val, types.FunctionType):
                continue

            if val.__doc__ is None:
                doc = ""
            else:
                doc = val.__doc__.split("\n")

            desc = [f"{st}\n" for st in doc[1:] if "*" in st]
            if not len(desc):
                raise NotImplementedError(
                    f"This data loader is missing documentation in the {prop} property"
                )
            val_type_str = f"{doc[0]}\n{''.join(desc)[:-1]}"
            repr_str += "  {}: {},\n".format(prop, val_type_str)

        repr_str += ")"
        return repr_str

    def to_jams(self):
        raise NotImplementedError

    def get_path(self, key):
        """Get absolute path to clip audio and annotations. Returns None if
        the path in the index is None

        Args:
            key (string): Index key of the audio or annotation type

        Returns:
            str or None: joined path string or None

        """
        if self._clip_paths[key][0] is None:
            return None
        else:
            return os.path.join(self._data_home, self._clip_paths[key][0])


class ClipGroup(Clip):
    """ClipGroup class.

    A clipgroup class is a collection of clip objects and their associated audio
    that can be mixed together.
    A clipgroup is itself a Clip, and can have its own associated audio (such as
    a mastered mix), its own metadata and its own annotations.

    """

    def __init__(
        self, clipgroup_id, data_home, dataset_name, index, clip_class, metadata
    ):
        """Clipgroup init method. Sets boilerplate attributes, including:

        - ``clipgroup_id``
        - ``_dataset_name``
        - ``_data_home``
        - ``_clipgroup_paths``
        - ``_clipgroup_metadata``

        Args:
            clipgroup_id (str): clipgroup id
            data_home (str): path where soundata will look for the dataset
            dataset_name (str): the identifier of the dataset
            index (dict): the dataset's file index
            metadata (function or None): a function returning a dictionary of metadata or None

        """
        if clipgroup_id not in index["clipgroups"]:
            raise ValueError(
                "{} is not a valid clipgroup_id in {}".format(
                    clipgroup_id, dataset_name
                )
            )

        self.clipgroup_id = clipgroup_id
        self._dataset_name = dataset_name

        self._data_home = data_home
        self._clipgroup_paths = index["clipgroups"][self.clipgroup_id]
        self._metadata = metadata
        self._clip_class = clip_class

        self._index = index
        self.clip_ids = self._index["clipgroups"][self.clipgroup_id]["clips"]

    @property
    def clips(self):
        return {
            t: self._clip_class(
                t, self._data_home, self._dataset_name, self._index, self._metadata
            )
            for t in self.clip_ids
        }

    @property
    def clip_audio_property(self):
        """The clip's audio property.

        Returns:

        """
        raise NotImplementedError("Mixing is not supported for this dataset")

    @property
    def _clipgroup_metadata(self):
        metadata = self._metadata()
        if metadata and self.clipgroup_id in metadata:
            return metadata[self.clipgroup_id]
        elif metadata:
            return metadata
        raise AttributeError("This ClipGroup does not have metadata")

    def get_path(self, key):
        """Get absolute path to clipgroup audio and annotations. Returns None if
        the path in the index is None

        Args:
            key (string): Index key of the audio or annotation type

        Returns:
            str or None: joined path string or None

        """
        if self._clipgroup_paths[key][0] is None:
            return None
        else:
            return os.path.join(self._data_home, self._clipgroup_paths[key][0])

    def get_target(self, clip_keys, weights=None, average=True, enforce_length=True):
        """Get target which is a linear mixture of clips

        Args:
            clip_keys (list): list of clip keys to mix together
            weights (list or None): list of positive scalars to be used in the average
            average (bool): if True, computes a weighted average of the clips
                if False, computes a weighted sum of the clips
            enforce_length (bool): If True, raises ValueError if the clips are
                not the same length. If False, pads audio with zeros to match the length
                of the longest clip

        Returns:
            np.ndarray: target audio with shape (n_channels, n_samples)

        Raises:
            ValueError:
                if sample rates of the clips are not equal
                if enforce_length=True and lengths are not equal

        """
        signals = []
        lengths = []
        sample_rates = []
        for k in clip_keys:
            audio, sample_rate = getattr(self.clips[k], self.clip_audio_property)
            # ensure all signals are shape (n_channels, n_samples)
            if len(audio.shape) == 1:
                audio = audio[np.newaxis, :]
            signals.append(audio)
            lengths.append(audio.shape[1])
            sample_rates.append(sample_rate)

        if len(set(sample_rates)) > 1:
            raise ValueError(
                "Sample rates for clips {} are not equal: {}".format(
                    clip_keys, sample_rates
                )
            )

        max_length = np.max(lengths)
        if any([l != max_length for l in lengths]):
            if enforce_length:
                raise ValueError(
                    "Clip's {} audio are not the same length {}. Use enforce_length=False to pad with zeros.".format(
                        clip_keys, lengths
                    )
                )
            else:
                # pad signals to the max length
                signals = [
                    np.pad(signal, ((0, 0), (0, max_length - signal.shape[1])))
                    for signal in signals
                ]

        if weights is None:
            weights = np.ones((len(clip_keys),))

        target = np.average(signals, axis=0, weights=weights)
        if not average:
            target *= np.sum(weights)

        return target

    def get_random_target(self, n_clips=None, min_weight=0.3, max_weight=1.0):
        """Get a random target by combining a random selection of clips with random weights

        Args:
            n_clips (int or None): number of clips to randomly mix. If None, uses all clips
            min_weight (float): minimum possible weight when mixing
            max_weight (float): maximum possible weight when mixing

        Returns:
            * np.ndarray - mixture audio with shape (n_samples, n_channels)
            * list - list of keys of included clips
            * list - list of weights used to mix clips

        """
        clips = list(self.clips.keys())
        assert len(clips) > 0
        if n_clips is not None and n_clips < len(clips):
            clips = np.random.choice(clips, n_clips, replace=False)

        weights = np.random.uniform(low=min_weight, high=max_weight, size=len(clips))
        target = self.get_target(clips, weights=weights)
        return target, clips, weights

    def get_mix(self):
        """Create a linear mixture given a subset of clips.

        Args:
            clip_keys (list): list of clip keys to mix together

        Returns:
            np.ndarray: mixture audio with shape (n_samples, n_channels)

        """
        clips = list(self.clips.keys())
        assert len(clips) > 0
        return self.get_target(clips)
