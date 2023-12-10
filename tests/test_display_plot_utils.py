import sys
import threading
import time
from unittest.mock import MagicMock, Mock, patch
import pytest
import numpy as np

from soundata import display_plot_utils
import soundata
import simpleaudio as sa  # Alternative library for audio playback


class MockDataset:
    def __init__(self, clip_durations=None, tags=None, subdatasets=None, layers=None):
        self.clip_durations = clip_durations or []
        self.tags = tags or {}
        self.subdatasets = subdatasets or {}
        self.layers = layers or {}

        self._index = {"clips": {i: f"clip_{i}" for i in range(len(clip_durations))}}
        self._metadata = {
            clip_id: {
                "subdataset": self.subdatasets.get(clip_id, None),
                **{
                    f"subdataset_layer_{layer}": self.layers.get(layer, {}).get(
                        clip_id, None
                    )
                    for layer in range(len(self.layers))
                },
            }
            for clip_id in self._index["clips"]
        }

    def clip(self, clip_id):
        return MockClip(self.clip_durations[int(clip_id)], self.tags.get(clip_id, []))


class MockClip:
    def __init__(self, duration, tags):
        self.audio = (np.random.random(int(duration * 44100)), 44100)
        self.tags = MockTags(tags)


class MockTags:
    def __init__(self, labels):
        self.labels = labels


def test_compute_clip_statistics():
    # Test with mock data
    mock_clip_durations = [30, 60, 45, 90, 120]  # durations in seconds
    dataset = MockDataset(mock_clip_durations)  # Assume this is a mock class you define

    stats = display_plot_utils.compute_clip_statistics(dataset)
    assert stats["total_duration"] == sum(mock_clip_durations)
    assert stats["mean_duration"] == np.mean(mock_clip_durations)
    assert stats["median_duration"] == np.median(mock_clip_durations)
    assert stats["std_deviation"] == np.std(mock_clip_durations)
    assert stats["min_duration"] == min(mock_clip_durations)
    assert stats["max_duration"] == max(mock_clip_durations)

    # Test with empty dataset
    empty_dataset = MockDataset([])
    with pytest.raises(ValueError):
        display_plot_utils.compute_clip_statistics(empty_dataset)


def test_perform_dataset_exploration_initialization():
    """Test the initialization and default states of widgets."""
    # Create a mock instance with specific return values for the widget attributes
    exploration_instance = Mock()
    exploration_instance.event_dist_check = Mock(value=True)
    exploration_instance.dataset_analysis_check = Mock(value=False)
    exploration_instance.audio_plot_check = Mock(value=True)
    exploration_instance.output = Mock()
    exploration_instance.loader = Mock()

    # Call the function with the mock instance
    display_plot_utils.perform_dataset_exploration(exploration_instance)
    exploration_instance.on_button_clicked(Mock())
    # Test initial values of widgets
    assert exploration_instance.event_dist_check.value is True
    assert exploration_instance.dataset_analysis_check.value is False
    assert exploration_instance.audio_plot_check.value is True


def test_time_unit_conversion(mocker):
    mock_stats = mocker.patch("soundata.display_plot_utils.compute_clip_statistics")
    # Use values that would trigger conversion to minutes and hours
    mock_stats.return_value = {
        "durations": [120, 180, 240],  # durations in seconds
        "total_duration": 54000,  # total duration in seconds
        "mean_duration": 180,  # mean in seconds
        "median_duration": 180,  # median in seconds
        "std_deviation": 60,
        "min_duration": 120,
        "max_duration": 240,
    }

    mock_show = mocker.patch("matplotlib.pyplot.show")

    dataset = MagicMock()
    dataset._index = {"clips": [1, 2, 3]}

    display_plot_utils.plot_clip_durations(dataset)

    mock_stats.assert_called_once()
    mock_show.assert_called_once()


def test_time_unit_conversion_less_than_120(mocker):
    mock_stats = mocker.patch("soundata.display_plot_utils.compute_clip_statistics")
    # Use values that would trigger conversion to minutes and hours
    mock_stats.return_value = {
        "durations": [120, 180, 240],  # durations in seconds
        "total_duration": 120,  # total duration in seconds
        "mean_duration": 180,  # mean in seconds
        "median_duration": 180,  # median in seconds
        "std_deviation": 60,
        "min_duration": 120,
        "max_duration": 240,
    }

    mock_show = mocker.patch("matplotlib.pyplot.show")

    dataset = MagicMock()
    dataset._index = {"clips": [1, 2, 3]}

    display_plot_utils.plot_clip_durations(dataset)

    mock_stats.assert_called_once()
    mock_show.assert_called_once()


def test_visualize_audio(mocker):
    # Mock audio data and sample rate
    mock_audio = np.random.rand(44100)  # 1 second of random audio data
    mock_sr = 44100  # Sample rate
    expected_duration = len(mock_audio) / mock_sr
    # Mock the clip method to return the mock audio and sample rate
    mock_clip = mocker.MagicMock()
    mock_clip.audio = (mock_audio, mock_sr)
    dataset = soundata.initialize("urbansound8k")
    mocker.patch.object(dataset, "clip", return_value=mock_clip)
    # Create an instance of your class
    instance = dataset
    instance._index = {"clips": {"dummy_clip_id": None}}  # Set up _index for the test

    display_plot_utils.visualize_audio(instance, "dummy_clip_id")

    instance.clip.assert_called_once_with("dummy_clip_id")


def test_visualize_audio_clip_id_none(mocker):
    # Mock the necessary attributes and methods
    mock_index = {"clips": {"clip1": "data1", "clip2": "data2"}}
    dataset = soundata.initialize("urbansound8k")
    dataset._index = mock_index

    # Mock the clip method to return a clip with an audio tuple (audio_data, sample_rate)
    mock_audio_data = np.random.rand(44100)  # Random audio data for testing
    mock_sample_rate = 44100  # Sample rate
    mock_clip = MagicMock()
    mock_clip.audio = (mock_audio_data, mock_sample_rate)
    mocker.patch.object(dataset, "clip", return_value=mock_clip)

    # Mock np.random.choice to control its output
    mocker.patch("numpy.random.choice", return_value="clip1")

    # Call the method with clip_id as None
    display_plot_utils.visualize_audio(dataset, None)

    # Assert that np.random.choice was called with the correct arguments
    np.random.choice.assert_called_once_with(list(mock_index["clips"].keys()))

    # Assert that the 'clip' method was called with the 'clip_id' chosen by np.random.choice
    dataset.clip.assert_called_once_with("clip1")


def test_play_segment(mocker):
    # Prepare the mock data
    mock_clip_id = "clip123"
    mock_audio_data = np.random.rand(44100)  # Random audio data for testing
    mock_sample_rate = 44100  # Sample rate

    # Mock the clip method to return a clip with an audio tuple (audio_data, sample_rate)
    mock_clip = MagicMock()
    mock_clip.audio = (mock_audio_data, mock_sample_rate)
    dataset = soundata.initialize("urbansound8k")
    mocker.patch.object(dataset, "clip", return_value=mock_clip)

    # Mock dependencies used in play_segment
    mocker.patch.object(sa, "play_buffer")
    mocker.patch.object(threading.Thread, "start")

    # Call the visualize_audio function with a specific clip_id
    display_plot_utils.visualize_audio(dataset, mock_clip_id)

    # Assert that the 'clip' method was called with the provided 'clip_id'
    dataset.clip.assert_called_once_with(mock_clip_id)


@pytest.fixture
def mock_output():
    # Mock the output object
    mock = MagicMock()
    mock.clear_output = MagicMock()
    return mock


@pytest.fixture
def mock_loader():
    # Mock the loader object
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_self():
    # Mock the 'self' object if it has dependencies
    mock = MagicMock()
    # Mock the specific methods of the self object
    soundata.display_plot_utils.plot_hierarchical_distribution = MagicMock()
    soundata.display_plot_utils.plot_clip_durations = MagicMock()
    soundata.display_plot_utils.visualize_audio = MagicMock()
    return mock


@pytest.mark.parametrize(
    "event_dist, dataset_analysis, audio_plot",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ],
)
def test_on_button_clicked(
    event_dist, dataset_analysis, audio_plot, mock_output, mock_loader, mock_self
):
    # Mock the checkbox objects
    event_dist_check = MagicMock(value=event_dist)
    dataset_analysis_check = MagicMock(value=dataset_analysis)
    audio_plot_check = MagicMock(value=audio_plot)

    # Mock the time.sleep to speed up the test
    with patch("soundata.display_plot_utils.time.sleep", return_value=None):
        # Call the function
        display_plot_utils.on_button_clicked(
            event_dist_check,
            dataset_analysis_check,
            audio_plot_check,
            mock_output,
            mock_loader,
            mock_self,
            clip_id=123,  # Use an appropriate clip_id
        )

    mock_output.clear_output.assert_called_once_with(wait=True)

    assert (
        mock_loader.value == "<p style='font-size:15px;'>Completed the processes!</p>"
    )


@patch("soundata.display_plot_utils.sa.play_buffer")
@patch("soundata.display_plot_utils.time.sleep", side_effect=lambda x: None)
def test_play_segment(mock_sleep, mock_play_buffer):
    # Mock simpleaudio object
    mock_play_obj = MagicMock()
    mock_play_obj.is_playing.side_effect = [True, True, False]
    mock_play_buffer.return_value = mock_play_obj

    # Create dummy audio segment and stop event
    audio_segment = MagicMock()
    stop_event = threading.Event()

    # Test normal playback
    display_plot_utils.play_segment(
        audio_segment, start_time=0, stop_event=stop_event, sr=44100
    )
    mock_play_buffer.assert_called_once()
    assert not stop_event.is_set()

    # Reset mock objects and event
    mock_play_buffer.reset_mock()
    stop_event.clear()

    # Test stopping playback
    stop_event.set()
    display_plot_utils.play_segment(
        audio_segment, start_time=0, stop_event=stop_event, sr=44100
    )
    assert stop_event.is_set()


@patch("soundata.display_plot_utils.time.sleep", side_effect=lambda x: None)
def test_update_line(mock_sleep):
    # Create mock objects for line1, line2, and fig
    line1 = MagicMock()
    line2 = MagicMock()
    fig = MagicMock()

    # Shared variables
    playing = [True]
    current_time = [0.0]
    duration = 2.0
    current_time_lock = threading.Lock()
    step = 0.1

    # Execute the function in a separate thread
    update_thread = threading.Thread(
        target=display_plot_utils.update_line,
        args=(
            playing,
            current_time,
            duration,
            current_time_lock,
            line1,
            line2,
            fig,
            step,
        ),
    )
    update_thread.start()

    # Wait for the duration of the test
    update_thread.join(duration + step)

    # Assertions
    assert not playing[0]
    assert current_time[0] == 0.0
    line1.set_xdata.assert_called()
    line2.set_xdata.assert_called()
    fig.canvas.draw_idle.assert_called()


@patch("soundata.display_plot_utils.threading.Thread")
def test_on_play_pause_clicked(mock_thread):
    # Prepare mock objects
    playing = [False]
    current_time = [0.0]
    play_thread = [None]
    stop_event = threading.Event()
    play_pause_button = MagicMock()
    play_segment_function = MagicMock()
    update_line_function = MagicMock()

    # Test when playing is False initially
    display_plot_utils.on_play_pause_clicked(
        playing,
        current_time,
        play_thread,
        stop_event,
        play_pause_button,
        play_segment_function,
        update_line_function,
    )

    assert playing[0] == True
    assert not stop_event.is_set()
    play_pause_button.description == "❚❚ Pause"
    mock_thread.assert_called()

    # Reset mock objects
    mock_thread.reset_mock()
    play_pause_button.reset_mock()

    # Set playing to True for the next test
    playing[0] = True

    # Test when playing is True
    display_plot_utils.on_play_pause_clicked(
        playing,
        current_time,
        play_thread,
        stop_event,
        play_pause_button,
        play_segment_function,
        update_line_function,
    )

    assert playing[0] == False
    assert stop_event.is_set()
    play_pause_button.description == "► Play"
    mock_thread.assert_not_called()


def test_on_reset_clicked():
    # Mock objects
    playing = [True]
    current_time = [10.0]
    play_thread = [MagicMock()]
    stop_event = threading.Event()
    line1 = MagicMock()
    line2 = MagicMock()
    slider = MagicMock()
    play_pause_button = MagicMock()
    fig = MagicMock()
    current_time_lock = threading.Lock()

    # Set up the play_thread mock
    play_thread[0].is_alive.return_value = True

    # Call the function
    display_plot_utils.on_reset_clicked(
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

    # Assertions
    assert not playing[0]
    assert current_time[0] == 0.0
    line1.set_xdata.assert_called_with([0, 0])
    line2.set_xdata.assert_called_with([0, 0])
    assert slider.value == 0.0
    assert play_pause_button.description == "► Play"
    fig.canvas.draw_idle.assert_called()


def test_on_slider_changed_with_thread_alive():
    # Setup for when play_thread is alive
    playing = [True]
    current_time = [0.0]
    play_thread = [MagicMock()]
    stop_event = threading.Event()
    line1 = MagicMock()
    line2 = MagicMock()
    fig = MagicMock()
    current_time_lock = threading.Lock()
    change = MagicMock()
    change.new = 5.0

    play_thread[0].is_alive.return_value = True

    # Call the function
    display_plot_utils.on_slider_changed(
        change,
        playing,
        current_time,
        play_thread,
        stop_event,
        line1,
        line2,
        fig,
        current_time_lock,
    )

    # Assertions for when thread is alive
    play_thread[0].join.assert_called()
    assert stop_event.is_set()


def test_on_slider_changed_with_thread_not_alive():
    # Setup for when play_thread is not alive
    playing = [True]
    current_time = [0.0]
    play_thread = [MagicMock()]
    stop_event = threading.Event()
    line1 = MagicMock()
    line2 = MagicMock()
    fig = MagicMock()
    current_time_lock = threading.Lock()
    change = MagicMock()
    change.new = 5.0

    play_thread[0].is_alive.return_value = False

    # Call the function
    display_plot_utils.on_slider_changed(
        change,
        playing,
        current_time,
        play_thread,
        stop_event,
        line1,
        line2,
        fig,
        current_time_lock,
    )

    # Assertions for when thread is not alive
    play_thread[0].join.assert_not_called()


def test_update_line_exception():
    # Mock objects
    playing = [True]
    current_time = [0.0]
    duration = 1.0
    current_time_lock = threading.Lock()
    line1 = MagicMock()
    line2 = MagicMock()
    fig = MagicMock()
    step = 0.1

    # Configure one of the mocks to raise an exception
    line1.set_xdata.side_effect = Exception("Test Exception")

    # Call the function with the mocks
    with patch("soundata.display_plot_utils.time.sleep", side_effect=lambda x: None):
        display_plot_utils.update_line(
            playing, current_time, duration, current_time_lock, line1, line2, fig, step
        )

    # Assertions to check if the exception was caught and handled
    line1.set_xdata.assert_called()


@patch("soundata.display_plot_utils.sa.play_buffer")
def test_play_segment_stop(mock_play_buffer):
    # Mock play_obj and its methods
    play_obj = MagicMock()
    play_obj.is_playing.side_effect = [
        True,
        True,
        True,
        False,
    ]  # Extend the playback simulation
    mock_play_buffer.return_value = play_obj

    # Prepare other mocks and shared variables
    audio_segment = MagicMock()
    audio_segment.__getitem__.return_value.raw_data = b"some raw data"
    stop_event = threading.Event()
    sr = 44100  # Sample rate

    # Start playing in a separate thread
    play_thread = threading.Thread(
        target=display_plot_utils.play_segment, args=(audio_segment, 0, stop_event, sr)
    )
    play_thread.start()

    # Increase the delay before setting the stop event
    time.sleep(0.5)
    stop_event.set()

    # Wait for the thread to finish
    play_thread.join()


@patch("soundata.display_plot_utils.sns.countplot")
@patch("soundata.display_plot_utils.pd.value_counts")
def test_plot_distribution(mock_value_counts, mock_countplot):
    # Prepare the data and parameters
    data = ["A", "B", "A", "C"]
    title = "Test Title"
    x_label = "X Label"
    y_label = "Y Label"
    axes = [MagicMock(), MagicMock()]
    subplot_position = 0

    # Mock value_counts to return a specific order
    mock_value_counts.return_value.index = ["A", "B", "C"]

    # Call the function
    display_plot_utils.plot_distribution(
        data, title, x_label, y_label, axes, subplot_position
    )

    # Assertions
    mock_value_counts.assert_called_with(data)
    mock_countplot.assert_called_with(
        y=data,
        order=["A", "B", "C"],
        palette=["#404040", "#126782", "#C9C9C9"],
        ax=axes[subplot_position],
    )
    axes_subplot = axes[subplot_position]
    axes_subplot.set_title.assert_called_with(title, fontsize=8)
    axes_subplot.set_xlabel.assert_called_with(x_label, fontsize=6)
    axes_subplot.set_ylabel.assert_called_with(y_label, fontsize=6)
    axes_subplot.tick_params.assert_called_with(axis="both", which="major", labelsize=6)
