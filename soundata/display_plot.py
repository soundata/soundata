# Audio Processing and Playback
from pydub import AudioSegment  # For manipulating audio files
from pydub.playback import play  # For playing audio files
import simpleaudio as sa  # Alternative library for audio playback
import librosa  # For advanced audio analysis

# Multithreading and Time Management
import threading  # For running processes in parallel
import time  # For handling time-related functions

# Data Handling and Visualization
import pandas as pd  # For handling and analyzing data structures
import numpy as np  # For numerical operations
import seaborn as sns  # For statistical data visualization
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations

# User Interface and Widgets
import ipywidgets as widgets  # For creating interactive UI components
from ipywidgets import (
    FloatSlider,
    Button,
    VBox,
    HBox,
    Checkbox,
    Label,
    Output,
)  # Specific UI widgets
from IPython.display import display  # For displaying widgets in IPython environments

# Miscellaneous
from functools import lru_cache  # For caching function call results
from tqdm import tqdm  # For displaying progress bars


@lru_cache(maxsize=None)  # Setting maxsize to None for an unbounded cache
def compute_clip_statistics(self):
    durations = [
        len(self.clip(c_id).audio[0]) / self.clip(c_id).audio[1]
        for c_id in tqdm(
            list(self._index["clips"].keys()), desc="Calculating durations"
        )
        if hasattr(self.clip(c_id), "audio")  # Adding the check here
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
    median_bin = np.digitize(median_duration, bins) - 1  # Correct bin for the median

    # Set the mean and median bins colors if they are within the range
    if 0 <= mean_bin < len(patches):
        patches[mean_bin].set_fc(base_colors[1])
    if 0 <= median_bin < len(patches):
        patches[median_bin].set_fc(base_colors[2])

    # Lines and text for mean and median
    ax1.axvline(mean_duration, color=base_colors[1], linestyle="dashed", linewidth=1)
    ax1.text(mean_duration + 0.2, max(n) * 0.9, "Mean", color=base_colors[1])
    ax1.axvline(median_duration, color=base_colors[2], linestyle="dashed", linewidth=1)
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
    plot_distribution(events, "Event Distribution in the Dataset", "Count", "Event", 0)

    # Plot Subclasses Distribution and then Hierarchical layers
    subplot_position = 1  # We've already plotted events at position 0
    if "subdatasets" in self._metadata:
        subclasses = [
            self._metadata[clip_id]["subdataset"] for clip_id in self._index["clips"]
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


def display_utils(self, clip_id=None):
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
