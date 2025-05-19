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


def on_button_clicked(
    event_dist_check,
    dataset_analysis_check,
    audio_plot_check,
    output,
    loader,
    self,
    clip_id,
):
    """Download data to `save_dir` and optionally print a message.

    Args:
        event_dist_check (Checkbox):
            Checkbox widget for event distribution analysis.
        dataset_analysis_check (Checkbox):
            Checkbox widget for dataset analysis.
        audio_plot_check (Checkbox):
            Checkbox widget for audio plot generation.
        output (Output):
            Output widget to display results.
        loader (HTML):
            HTML widget displaying a loader.
        self:
            Reference to the current instance of the class.
        clip_id (str or None):
            The identifier of the clip to explore. If None, a random clip will be chosen.

    Clears previous outputs, displays a loader, performs selected computations, and updates the output accordingly.
    """
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
            plot_hierarchical_distribution(self)

        if dataset_analysis_check.value:
            print("Conducting dataset analysis... Please wait.")
            plot_clip_durations(self)

        if audio_plot_check.value:
            print("Generating audio plot... Please wait.")
            visualize_audio(self, clip_id)

        # Remove the loader after the content is loaded
        loader.value = "<p style='font-size:15px;'>Completed the processes!</p>"


def perform_dataset_exploration(self, clip_id=None):
    """Explore the dataset for a given clip_id or a random clip if clip_id is None.

    Args:
        self:
            Reference to the current instance of the class.
        clip_id (str or None):
            The identifier of the clip to explore. If None, a random clip will be chosen.

    Displays interactive checkboxes for user input, a button to trigger exploration, and the exploration results.
    """
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

    plot_button.on_click(
        lambda b: on_button_clicked(
            event_dist_check,
            dataset_analysis_check,
            audio_plot_check,
            output,
            loader,
            self,
            clip_id,
        )
    )

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


@lru_cache(maxsize=None)  # Setting maxsize to None for an unbounded cache
def compute_clip_statistics(self):
    """Compute statistics for clip durations in the dataset.

    Args:
        self:
            Reference to the current instance of the class.

    Returns:
        dict: Dictionary containing clip duration statistics.

    Calculates statistics such as total duration, mean duration, median duration, standard deviation,
    minimum duration, maximum duration, and total clip count.
    """
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
    """Plot the distribution of clip durations in the dataset.

    Args:
        self:
            Reference to the current instance of the class.

    Generates a histogram of clip durations, overlays mean and median lines, and displays statistics.
    """
    stats = compute_clip_statistics(self)
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

    if total_duration > 120:
        total_duration /= 60
        total_duration_unit = "hours" if convert_to_minutes else "minutes"
    else:
        total_duration_unit = "minutes" if convert_to_minutes else "seconds"

    # Define the base colors for soundata template
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


def plot_distribution(data, title, x_label, y_label, axes, subplot_position):
    """Plot the distribution of data.

    Args:
        data (list):
            Data values to be plotted.
        title (str):
            Title for the plot.
        x_label (str):
            Label for the x-axis.
        y_label (str):
            Label for the y-axis.
        axes (list of Axes):
            List of subplot axes.
        subplot_position (int):
            Position of the subplot.

    Plots the distribution of data with count labels and adjusts font sizes.
    """
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


def plot_hierarchical_distribution(self):
    """Plot hierarchical distributions of events, subclasses, and subdataset layers.

    Args:
        self:
            Reference to the current instance of the class.

    Generates count plots for event distribution, subclass distribution, and subdataset layers distribution.
    """
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

    plot_distribution(
        events, "Event Distribution in the Dataset", "Count", "Event", axes, 0
    )

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
            axes,
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
            axes,
            subplot_position,
        )
        layer += 1
        subplot_position += 1

    plt.tight_layout()
    plt.show()
    print("\n")


def play_segment(audio_segment, start_time, stop_event, sr):
    """Play an audio segment.

    Args:
        audio_segment (AudioSegment):
            Audio segment to be played.
        start_time (float):
            Start time in seconds.
        stop_event (Event):
            Event to signal the stop of audio playback.
        sr (int):
            Sample rate.

    Plays an audio segment from the specified start time until the stop event is set.
    """
    try:
        segment_start = start_time * 1000  # Convert to milliseconds
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


def update_line(
    playing, current_time, duration, current_time_lock, line1, line2, fig, step=0.1
):
    """Update the position of a vertical line on a plot.

    Args:
        playing (list):
            List indicating if audio is currently playing.
        current_time (list):
            List containing the current time position.
        duration (float):
            Total duration of the audio.
        current_time_lock (Lock):
            Lock to ensure thread-safe access to current_time.
        line1 (Line2D):
            Line to be updated.
        line2 (Line2D):
            Another line to be updated.
        fig (Figure):
            Figure containing the plot.
        step (float, optional):
            Step size for updating the line position. Defaults to 0.1.

    Updates the position of the vertical lines on the plot while audio is playing.
    """
    try:
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


def on_play_pause_clicked(
    playing,
    current_time,
    play_thread,
    stop_event,
    play_pause_button,
    play_segment_function,
    update_line_function,
):
    """Handle the play/pause button click event.

    Args:
        playing (list):
            List indicating if audio is currently playing.
        current_time (list):
            List containing the current time position.
        play_thread (list):
            List containing the audio playback thread.
        stop_event (Event):
            Event to signal the stop of audio playback.
        play_pause_button (Button):
            Button widget for play/pause control.
        play_segment_function (function):
            Function to play an audio segment.
        update_line_function (function):
            Function to update the position of a vertical line on the plot.

    Handles the play/pause button click event to control audio playback.
    """
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
            target=play_segment_function, args=(current_time[0],)
        )
        play_thread[0].start()
        update_line_thread = threading.Thread(target=update_line_function)
        update_line_thread.start()


def on_reset_clicked(
    playing,
    current_time,
    play_thread,
    stop_event,
    line1,
    line2,
    slider,
    play_pause_button,
    fig,
    current_time_lock,
):
    """Handle the reset button click event.

    Args:
        playing (list):
            List indicating if audio is currently playing.
        current_time (list):
            List containing the current time position.
        play_thread (list):
            List containing the audio playback thread.
        stop_event (Event):
            Event to signal the stop of audio playback.
        line1 (Line2D):
            Line to be reset.
        line2 (Line2D):
            Another line to be reset.
        slider (FloatSlider):
            Slider widget for audio navigation.
        play_pause_button (Button):
            Button widget for play/pause control.
        fig (Figure):
            Figure containing the plot.
        current_time_lock (Lock):
            Lock to ensure thread-safe access to current_time.

    Handles the reset button click event to stop audio playback and reset the plot and slider.
    """
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


def on_slider_changed(
    change,
    playing,
    current_time,
    play_thread,
    stop_event,
    line1,
    line2,
    fig,
    current_time_lock,
):
    """Handle slider value change event.

    Args:
        change:
            Change event object.
        playing (list):
            List indicating if audio is currently playing.
        current_time (list):
            List containing the current time position.
        play_thread (list):
            List containing the audio playback thread.
        stop_event (Event):
            Event to signal the stop of audio playback.
        line1 (Line2D):
            Line to be updated.
        line2 (Line2D):
            Another line to be updated.
        fig (Figure):
            Figure containing the plot.
        current_time_lock (Lock):
            Lock to ensure thread-safe access to current_time.

    Handles the slider value change event to update the audio playback position and plot.
    """
    if playing[0]:
        stop_event.set()
        if play_thread[0].is_alive():
            play_thread[0].join()
    with current_time_lock:
        current_time[0] = change.new
    line1.set_xdata([current_time[0], current_time[0]])
    line2.set_xdata([current_time[0], current_time[0]])
    fig.canvas.draw_idle()


def visualize_audio(self, clip_id):
    """Visualize audio data for a specified clip.

    Args:
        self:
            Reference to the current instance of the class.
        clip_id (str or None):
            The identifier of the clip to explore. If None, a random clip will be chosen.

    Displays audio waveform, a Mel spectrogram, and provides playback controls.
    """
    if clip_id is None:  # Use the local variable
        clip_id = np.random.choice(
            list(self._index["clips"].keys())
        )  # Modify the local variable
    clip = self.clip(clip_id)  # Use the local variable

    stop_event = threading.Event()
    current_time_lock = threading.Lock()

    audio, sr = clip.audio
    duration = len(audio) / sr

    audio = audio / np.max(np.abs(audio))

    # Convert to int16 for playback
    audio_playback = np.int16(audio * 32767)

    audio_segment = AudioSegment(
        audio_playback.tobytes(), frame_rate=sr, sample_width=2, channels=1
    )

    # Truncate the audio to a maximum duration (e.g., 1 minute)
    max_duration_secs = 60
    print("Truncating audio to the first 1 minute if less than 1 minute.")
    audio = audio[: int(max_duration_secs * sr)]
    duration = min(duration, max_duration_secs)

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

    # Adjusting the font size for axis labels
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(8)

    # Plotting the Mel spectrogram
    im = librosa.display.specshow(log_S, sr=sr, x_axis="time", y_axis="mel", ax=ax2)
    ax2.set_title("Mel spectrogram", fontsize=8)
    ax2.set_xlim(0, duration)
    (line2,) = ax2.plot([0, 0], ax2.get_ylim(), color="#126782", linestyle="--")

    # Adjusting the font size for axis labels
    ax2.set_xlabel("Time (s)", fontsize=8)
    ax2.set_ylabel("Hz", fontsize=8)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(8)

    fig.subplots_adjust(right=0.8)  # Adjust the right side of the layout

    # Add the colorbar
    bbox = ax2.get_position()
    cbar_ax = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("dB", rotation=270, labelpad=15, fontsize=8)
    cbar.ax.tick_params(labelsize=8)  # Set the size of the tick labels

    # Ensure the tight_layout does not overlap the axes with colorbar
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    playing = [False]
    current_time = [0.0]
    play_thread = [None]

    # Create UI elements
    slider = FloatSlider(
        value=0.0,
        min=0.0,
        max=duration,
        step=0.1,
        description="Seek:",
        tooltip="Drag the slider to a specific point in the audio to play from that time.",
    )
    play_pause_button = Button(description="► Play")
    reset_button = Button(description="Reset")

    # Setting up event handlers
    play_pause_button.on_click(
        lambda b: on_play_pause_clicked(
            playing,
            current_time,
            play_thread,
            stop_event,
            play_pause_button,
            lambda start_time: play_segment(audio_segment, start_time, stop_event, sr),
            lambda: update_line(
                playing, current_time, duration, current_time_lock, line1, line2, fig
            ),
        )
    )
    reset_button.on_click(
        lambda b: on_reset_clicked(
            playing,
            current_time,
            play_thread,
            stop_event,
            line1,
            line2,
            slider,
            play_pause_button,
            fig,
            current_time_lock,
        )
    )
    slider.observe(
        lambda change: on_slider_changed(
            change,
            playing,
            current_time,
            play_thread,
            stop_event,
            line1,
            line2,
            fig,
            current_time_lock,
        ),
        names="value",
    )

    # Display the UI elements
    slider_label = Label("Drag the slider to navigate through the audio:")
    display(VBox([HBox([play_pause_button, reset_button]), slider_label, slider]))
