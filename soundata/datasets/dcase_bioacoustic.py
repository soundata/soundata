"""DCASE-BIOACOUSTIC Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    *DCASE-BIOACOUSTIC*

    *Development set:*

    The development set for task 5 of DCASE 2022 "Few-shot Bioacoustic Event Detection" consists of 192 audio files acquired from different bioacoustic sources. The dataset is split into training and validation sets. 

    Multi-class annotations are provided for the training set with positive (POS), negative (NEG) and unkwown (UNK) values for each class. UNK indicates uncertainty about a class. 

    Single-class (class of interest) annotations are provided for the validation set, with events marked as positive (POS) or unkwown (UNK) provided for the class of interest. 

    this version (3):

    - fixes issues with annotations from HB set


    Folder Structure:

    Development_Set.zip

    |_Development_Set/

        |__Training_Set/

            |___JD/

                |____*.wav

                |____*.csv

            |___HT/

                |____*.wav

                |____*.csv

            |___BV/

                |____*.wav

                |____*.csv

            |___MT/

                |____*.wav

                |____*.csv

            |___WMW/

                |____*.wav

                |____*.csv

    

        |__Validation_Set/

            |___HB/

                |____*.wav

                |____*.csv

            |___PB/

                |____*.wav

                |____*.csv

            |___ME/

                |____*.wav

                |____*.csv

    

    Development_Set_Annotations.zip has the same structure but contains only the *.csv files

    

    *Annotation structure*

    Each line of the annotation csv represents an event in the audio file. The column descriptions are as follows:

    TRAINING SET
    ---------------------
    Audiofilename, Starttime, Endtime, CLASS_1, CLASS_2, ...CLASS_N

    VALIDATION SET
    ---------------------
    Audiofilename, Starttime, Endtime, Q

    

    *Classes*

    DCASE2022_task5_training_set_classes.csv and DCASE2022_task5_validation_set_classes.csv provide a table with class code correspondence to class name for all classes in the Development set.

    DCASE2022_task5_training_set_classes.csv
    ---------------------
    dataset, class_code, class_name

    DCASE2022_task5_validation_set_classes.csv
    ---------------------
    dataset, recording, class_code, class_name

    

    *Evaluation set*

    The evaluation set for task 5 of DCASE 2022 "Few-shot Bioacoustic Event Detection" consists of 46 audio files acquired from different bioacoustic sources. 

    The first 5 annotations are provided for each file, with events marked as positive (POS) for the class of interest. 

    This dataset is to be used for evaluation purposes during the task and the rest of the annotations will be released after the end of the DCASE 2022 challenge (July 1st).

    Folder Structure

    Evaluation_Set.zip

        |___DC/

            |____*.wav

            |____*.csv

        |___CT/

            |____*.wav

            |____*.csv

        |___CHE/

            |____*.wav

            |____*.csv

        |___MGE/

            |____*.wav

            |____*.csv

        |___MS/

            |____*.wav

            |____*.csv

        |___QU/

            |____*.wav

            |____*.csv

    

    Evaluation_Set_5shots.zip has the same structure but contains only the *.wav files.

    Evaluation_Set_5shots_annotations_only.zip has the same structure but contains only the *.csv files

    The subfolders denote different recording sources and there may or may not be overlap between classes of interest from different wav files.

    Annotation structure

    Each line of the annotation csv represents an event in the audio file. The column descriptions are as follows:
    [ Audiofilename, Starttime, Endtime, Q ]


    Open Access:

    This dataset is available under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
    

    Contact info:

    Please send any feedback or questions to:

    Ines Nolasco -  i.dealmeidanolasco@qmul.ac.uk
"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import jams
import glob
import json
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
import pandas as pd
from IPython.display import display, Audio
from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io
import sounddevice as sd
import numpy as np
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Button, VBox, HBox, Checkbox, Label
import threading
import time
from pydub import AudioSegment
from pydub.playback import play

BIBTEX = """
@dataset{nolasco_ines_2022_6482837,
  author       = {Nolasco, Ines and
                  Singh, Shubhr and
                  Strandburg-Peshkin, Ariana and
                  Gill, Lisa and
                  Pamula, Hanna and
                  Morford, Joe and
                  Emmerson, Michael and
                  Jensen, Frants and
                  Whitehead, Helen and
                  Kiskin, Ivan and
                  Vidaña-Vila, Ester and
                  Lostanlen, Vincent and
                  Morfi, Veronica and
                  Stowell, Dan},
  title        = {{DCASE 2022 Task 5: Few-shot Bioacoustic Event 
                   Detection Development Set}},
  month        = mar,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6482837},
  url          = {https://doi.org/10.5281/zenodo.6482837}
}
"""
REMOTES = {
    "dev": download_utils.RemoteFileMetadata(
        filename="Development_Set.zip",
        url="https://zenodo.org/record/6482837/files/Development_Set.zip?download=1",
        checksum="cf4d3540c6c78ac2b3df2026c4f1f7ea",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "train-classes": download_utils.RemoteFileMetadata(
        filename="DCASE2022_task5_Training_set_classes.csv",
        url="https://zenodo.org/record/6482837/files/DCASE2022_task5_Training_set_classes.csv?download=1",
        checksum="abce1818ba10436971bad0b6a3464aa6",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "validation-classes": download_utils.RemoteFileMetadata(
        filename="DCASE2022_task5_Validation_set_classes.csv",
        url="https://zenodo.org/record/6482837/files/DCASE2022_task5_Validation_set_classes.csv?download=1",
        checksum="0c05ff0c9e1662ff8958c4c812abffdb",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "eval": download_utils.RemoteFileMetadata(
        filename="Evaluation_set_5shots.zip",
        url="https://zenodo.org/record/6517414/files/Evaluation_set_5shots.zip?download=1",
        checksum="5212c0e133874bba1ee25c81ced0de99",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Clip(core.Clip):
    """DCASE bioacoustic Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        csv_path (str): path to the csv file
        clip_id (str): clip id
        split (str): subset the clip belongs to (for experiments): train, validate, or test

    Cached properties:
        events_classes (list): list of classes annotated for the file
        events (soundata.annotations.Events): sound events with start time, end time, labels (list for all classes) and confidence
        POSevents (soundata.annotations.Events): sound events for the positive class with start time, end time, label and confidence

    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.csv_path = self.get_path("csv")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def split(self):
        """The data splits (e.g. train)

        Returns
            * str - split

        """
        return self._clip_metadata.get("split")

    @property
    def subdataset(self):
        """The (sub)dataset

        Returns
            * str - subdataset

        """
        return self._clip_metadata.get("subdataset")

    @core.cached_property
    def events_classes(self) -> Optional[list]:
        """The audio events

        Returns
            * list - list of the annotated events

        """
        return load_events_classes(self.csv_path)

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        """The audio events

        Returns
            * annotations.Events - audio event object

        """
        return load_events(self.csv_path)

    @core.cached_property
    def POSevents(self) -> Optional[annotations.Events]:
        """The audio events for POS (positive) class

        Returns
            * annotations.Events - audio event object

        """
        return load_POSevents(self.csv_path)

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            events=self.events,
            metadata={
                "split": self._clip_metadata.get("split"),
                "subdataset": self._clip_metadata.get("subdataset"),
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a DCASE bioacoustic audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate without resampling.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@io.coerce_to_string_io
def load_events(fhandle: TextIO) -> annotations.Events:
    """Load an DCASE bioacoustic sound events annotation file

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        Events: sound events annotation data

    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    for line in reader:
        times.append([float(line[1]), float(line[2])])
        classes = [class_ids[i] for i, l in enumerate(line[3:])]
        labels.append(",".join(classes))
        confidence.append(1.0)
    events_data = annotations.Events(
        intervals=np.array(times),
        intervals_unit="seconds",
        labels=labels,
        labels_unit="open",
        confidence=np.array(confidence),
    )
    return events_data


@io.coerce_to_string_io
def load_POSevents(fhandle: TextIO) -> annotations.Events:
    """Load an DCASE bioacoustic sound events annotation file, just for POS labels

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        Events: sound events annotation data

    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    for line in reader:
        times.append([float(line[1]), float(line[2])])
        classes = [class_ids[i] for i, l in enumerate(line[3:]) if l == "POS"]
        labels.append(",".join(classes))
        confidence.append(1.0)
    events_data = annotations.Events(
        intervals=np.array(times),
        intervals_unit="seconds",
        labels=labels,
        labels_unit="open",
        confidence=np.array(confidence),
    )
    return events_data


@io.coerce_to_string_io
def load_events_classes(fhandle: TextIO) -> list:
    """Load an DCASE bioacoustic sound events annotation file

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file
        positive (bool): False get all labels, True get just POS labels

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        class_ids: list of events classes

    """
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    return class_ids


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The DCASE bioacoustic dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="dcase_bioacoustic",
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
        metadata_index = {
            clip_id: {
                "subdataset": os.path.normpath(v["csv"][0])
                .split(clip_id)[0]
                .split(os.path.sep)[-2],
                "split": "train"
                if "Training" in os.path.normpath(v["csv"][0]).split(clip_id)[0]
                else "validation"
                if "Validation" in os.path.normpath(v["csv"][0]).split(clip_id)[0]
                else "evaluation",
            }
            for clip_id, v in self._index["clips"].items()
        }

        metadata_paths = {
            "train": os.path.join(
                self.data_home, "DCASE2022_task5_Training_set_classes.csv"
            ),
            "validation": os.path.join(
                self.data_home, "DCASE2022_task5_Validation_set_classes.csv"
            ),
        }

        metadata_index["class_codes"] = {}
        metadata_index["subdatasets"] = {}

        for split, metadata_path in metadata_paths.items():
            metadata_path = os.path.normpath(metadata_path)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError("Metadata not found. Did you run .download()?")

            with open(metadata_path, "r") as fhandle:
                reader = csv.reader(fhandle, delimiter=",")

                headers = next(reader)
                class_code_id = headers.index("class_code")
                class_name_id = headers.index("class_name")
                dataset_id = headers.index("dataset")

                for line in reader:
                    metadata_index["class_codes"][line[class_code_id]] = {
                        "subdataset": line[dataset_id],
                        "class_name": line[class_name_id],
                        "split": split,
                    }
                    if line[dataset_id] not in metadata_index["subdatasets"]:
                        metadata_index["subdatasets"][line[dataset_id]] = [
                            line[class_code_id]
                        ]
                    else:
                        metadata_index["subdatasets"][line[dataset_id]].append(
                            line[class_code_id]
                        )

        return metadata_index
    
    def loading_spinner(self, duration=5):
        end_time = time.time() + duration
        symbols = ['|', '/', '-', '\\']

        while time.time() < end_time:
            for symbol in symbols:
                sys.stdout.write(f'\rLoading {symbol}')
                sys.stdout.flush()
                time.sleep(0.2)
        sys.stdout.write('\rDone!     \n')

    def event_distribution(self):
        print("Event Distribution:")
        print("\nPlotting the distribution of different events across the dataset...")
        events = [label for clip_id in self._index["clips"] for label in self.clip(clip_id).events.labels]

        plt.figure(figsize=(10, 6))
        sns.countplot(y=events, order=pd.value_counts(events).index)
        plt.title('Event distribution in the dataset')
        plt.xlabel('Count')
        plt.ylabel('Event')
        plt.tight_layout()
        plt.show()
        print("\n")

    def dataset_analysis(self):
        print("Dataset Analysis:")

        durations = [len(self.clip(c_id).audio[0])/self.clip(c_id).audio[1] for c_id in self._index["clips"].keys()]
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)

        # Print analysis results
        print("\nShowing statistics about the dataset...")
        print(f"Mean duration for the entire dataset: {mean_duration:.2f} seconds")
        print(f"Median duration for the entire dataset: {median_duration:.2f} seconds")
        print(f"Total number of clips: {len(self._index['clips'])}")

        unique_classes = set([label for clip_id in self._index["clips"] for label in self.clip(clip_id).events.labels])
        print(f"Total number of unique classes: {len(unique_classes)}")
        print(f"Subclasses: {self._metadata.get('subdatasets', 'None provided')}")

        print("\nPlotting the distribution of clip durations...")
        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=30)
        plt.title('Distribution of Clip Durations')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Number of Clips')
        plt.show()

        print("\nPlotting the distribution of subclasses (if available)...")
        subclasses = [self._metadata[clip_id]['subdataset'] for clip_id in self._index["clips"]]
        plt.figure(figsize=(10, 6))
        sns.countplot(y=subclasses, order=pd.value_counts(subclasses).index)
        plt.title('Subclass distribution in the dataset')
        plt.xlabel('Count')
        plt.ylabel('Subclass')
        plt.tight_layout()
        plt.show()
        print("\n")

    def class_distribution(self):
        print("Class Distribution:")
        print("\nShowing counts for each class and subclass (if available)...")

        classes = [label for clip_id in self._index["clips"] for label in self.clip(clip_id).events.labels]
        unique_classes = set(classes)

        for cls in unique_classes:
            print(f"Class: {cls}, Count: {classes.count(cls)}")

            if "subdatasets" in self._metadata and cls in self._metadata["subdatasets"]:
                for subclass in self._metadata["subdatasets"][cls]:
                    print(f"    Subclass: {subclass}, Count: {classes.count(subclass)}")

    """DCASE-BIOACOUSTIC Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    *DCASE-BIOACOUSTIC*

    *Development set:*

    The development set for task 5 of DCASE 2022 "Few-shot Bioacoustic Event Detection" consists of 192 audio files acquired from different bioacoustic sources. The dataset is split into training and validation sets. 

    Multi-class annotations are provided for the training set with positive (POS), negative (NEG) and unkwown (UNK) values for each class. UNK indicates uncertainty about a class. 

    Single-class (class of interest) annotations are provided for the validation set, with events marked as positive (POS) or unkwown (UNK) provided for the class of interest. 

    this version (3):

    - fixes issues with annotations from HB set


    Folder Structure:

    Development_Set.zip

    |_Development_Set/

        |__Training_Set/

            |___JD/

                |____*.wav

                |____*.csv

            |___HT/

                |____*.wav

                |____*.csv

            |___BV/

                |____*.wav

                |____*.csv

            |___MT/

                |____*.wav

                |____*.csv

            |___WMW/

                |____*.wav

                |____*.csv

    

        |__Validation_Set/

            |___HB/

                |____*.wav

                |____*.csv

            |___PB/

                |____*.wav

                |____*.csv

            |___ME/

                |____*.wav

                |____*.csv

    

    Development_Set_Annotations.zip has the same structure but contains only the *.csv files

    

    *Annotation structure*

    Each line of the annotation csv represents an event in the audio file. The column descriptions are as follows:

    TRAINING SET
    ---------------------
    Audiofilename, Starttime, Endtime, CLASS_1, CLASS_2, ...CLASS_N

    VALIDATION SET
    ---------------------
    Audiofilename, Starttime, Endtime, Q

    

    *Classes*

    DCASE2022_task5_training_set_classes.csv and DCASE2022_task5_validation_set_classes.csv provide a table with class code correspondence to class name for all classes in the Development set.

    DCASE2022_task5_training_set_classes.csv
    ---------------------
    dataset, class_code, class_name

    DCASE2022_task5_validation_set_classes.csv
    ---------------------
    dataset, recording, class_code, class_name

    

    *Evaluation set*

    The evaluation set for task 5 of DCASE 2022 "Few-shot Bioacoustic Event Detection" consists of 46 audio files acquired from different bioacoustic sources. 

    The first 5 annotations are provided for each file, with events marked as positive (POS) for the class of interest. 

    This dataset is to be used for evaluation purposes during the task and the rest of the annotations will be released after the end of the DCASE 2022 challenge (July 1st).

    Folder Structure

    Evaluation_Set.zip

        |___DC/

            |____*.wav

            |____*.csv

        |___CT/

            |____*.wav

            |____*.csv

        |___CHE/

            |____*.wav

            |____*.csv

        |___MGE/

            |____*.wav

            |____*.csv

        |___MS/

            |____*.wav

            |____*.csv

        |___QU/

            |____*.wav

            |____*.csv

    

    Evaluation_Set_5shots.zip has the same structure but contains only the *.wav files.

    Evaluation_Set_5shots_annotations_only.zip has the same structure but contains only the *.csv files

    The subfolders denote different recording sources and there may or may not be overlap between classes of interest from different wav files.

    Annotation structure

    Each line of the annotation csv represents an event in the audio file. The column descriptions are as follows:
    [ Audiofilename, Starttime, Endtime, Q ]


    Open Access:

    This dataset is available under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
    

    Contact info:

    Please send any feedback or questions to:

    Ines Nolasco -  i.dealmeidanolasco@qmul.ac.uk
"""

import os
import sys
import csv
import glob
import json
import time
import threading
from typing import BinaryIO, Optional, TextIO, Tuple
import simpleaudio as sa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import jams
import pandas as pd
from IPython.display import display, Audio, clear_output
import ipywidgets as widgets
from ipywidgets import FloatSlider, Button, VBox, HBox, Layout, Output
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, Button, VBox, HBox, Output
import time
from soundata import core, annotations, io, download_utils, jams_utils
BIBTEX = """
@dataset{nolasco_ines_2022_6482837,
  author       = {Nolasco, Ines and
                  Singh, Shubhr and
                  Strandburg-Peshkin, Ariana and
                  Gill, Lisa and
                  Pamula, Hanna and
                  Morford, Joe and
                  Emmerson, Michael and
                  Jensen, Frants and
                  Whitehead, Helen and
                  Kiskin, Ivan and
                  Vidaña-Vila, Ester and
                  Lostanlen, Vincent and
                  Morfi, Veronica and
                  Stowell, Dan},
  title        = {{DCASE 2022 Task 5: Few-shot Bioacoustic Event 
                   Detection Development Set}},
  month        = mar,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6482837},
  url          = {https://doi.org/10.5281/zenodo.6482837}
}
"""
REMOTES = {
    "dev": download_utils.RemoteFileMetadata(
        filename="Development_Set.zip",
        url="https://zenodo.org/record/6482837/files/Development_Set.zip?download=1",
        checksum="cf4d3540c6c78ac2b3df2026c4f1f7ea",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "train-classes": download_utils.RemoteFileMetadata(
        filename="DCASE2022_task5_Training_set_classes.csv",
        url="https://zenodo.org/record/6482837/files/DCASE2022_task5_Training_set_classes.csv?download=1",
        checksum="abce1818ba10436971bad0b6a3464aa6",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "validation-classes": download_utils.RemoteFileMetadata(
        filename="DCASE2022_task5_Validation_set_classes.csv",
        url="https://zenodo.org/record/6482837/files/DCASE2022_task5_Validation_set_classes.csv?download=1",
        checksum="0c05ff0c9e1662ff8958c4c812abffdb",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
    "eval": download_utils.RemoteFileMetadata(
        filename="Evaluation_set_5shots.zip",
        url="https://zenodo.org/record/6517414/files/Evaluation_set_5shots.zip?download=1",
        checksum="5212c0e133874bba1ee25c81ced0de99",
        # unpack_directories=["URBAN-SED_v2.0.0"],
    ),
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Clip(core.Clip):
    """DCASE bioacoustic Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        csv_path (str): path to the csv file
        clip_id (str): clip id
        split (str): subset the clip belongs to (for experiments): train, validate, or test

    Cached properties:
        events_classes (list): list of classes annotated for the file
        events (soundata.annotations.Events): sound events with start time, end time, labels (list for all classes) and confidence
        POSevents (soundata.annotations.Events): sound events for the positive class with start time, end time, label and confidence

    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.csv_path = self.get_path("csv")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The clip's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def split(self):
        """The data splits (e.g. train)

        Returns
            * str - split

        """
        return self._clip_metadata.get("split")

    @property
    def subdataset(self):
        """The (sub)dataset

        Returns
            * str - subdataset

        """
        return self._clip_metadata.get("subdataset")

    @core.cached_property
    def events_classes(self) -> Optional[list]:
        """The audio events

        Returns
            * list - list of the annotated events

        """
        return load_events_classes(self.csv_path)

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        """The audio events

        Returns
            * annotations.Events - audio event object

        """
        return load_events(self.csv_path)

    @core.cached_property
    def POSevents(self) -> Optional[annotations.Events]:
        """The audio events for POS (positive) class

        Returns
            * annotations.Events - audio event object

        """
        return load_POSevents(self.csv_path)

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        return jams_utils.jams_converter(
            audio_path=self.audio_path,
            events=self.events,
            metadata={
                "split": self._clip_metadata.get("split"),
                "subdataset": self._clip_metadata.get("subdataset"),
            },
        )


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a DCASE bioacoustic audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate without resampling.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@io.coerce_to_string_io
def load_events(fhandle: TextIO) -> annotations.Events:
    """Load an DCASE bioacoustic sound events annotation file

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        Events: sound events annotation data

    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    for line in reader:
        times.append([float(line[1]), float(line[2])])
        classes = [class_ids[i] for i, l in enumerate(line[3:])]
        labels.append(",".join(classes))
        confidence.append(1.0)
    events_data = annotations.Events(
        intervals=np.array(times),
        intervals_unit="seconds",
        labels=labels,
        labels_unit="open",
        confidence=np.array(confidence),
    )
    return events_data


@io.coerce_to_string_io
def load_POSevents(fhandle: TextIO) -> annotations.Events:
    """Load an DCASE bioacoustic sound events annotation file, just for POS labels

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        Events: sound events annotation data

    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    for line in reader:
        times.append([float(line[1]), float(line[2])])
        classes = [class_ids[i] for i, l in enumerate(line[3:]) if l == "POS"]
        labels.append(",".join(classes))
        confidence.append(1.0)
    events_data = annotations.Events(
        intervals=np.array(times),
        intervals_unit="seconds",
        labels=labels,
        labels_unit="open",
        confidence=np.array(confidence),
    )
    return events_data


@io.coerce_to_string_io
def load_events_classes(fhandle: TextIO) -> list:
    """Load an DCASE bioacoustic sound events annotation file

    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file
        positive (bool): False get all labels, True get just POS labels

    Raises:
        IOError: if csv_path doesn't exist

    Returns:
        class_ids: list of events classes

    """
    reader = csv.reader(fhandle, delimiter=",")
    headers = next(reader)
    class_ids = headers[3:]
    return class_ids


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The DCASE bioacoustic dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="dcase_bioacoustic",
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
        metadata_index = {
            clip_id: {
                "subdataset": os.path.normpath(v["csv"][0])
                .split(clip_id)[0]
                .split(os.path.sep)[-2],
                "split": "train"
                if "Training" in os.path.normpath(v["csv"][0]).split(clip_id)[0]
                else "validation"
                if "Validation" in os.path.normpath(v["csv"][0]).split(clip_id)[0]
                else "evaluation",
            }
            for clip_id, v in self._index["clips"].items()
        }

        metadata_paths = {
            "train": os.path.join(
                self.data_home, "DCASE2022_task5_Training_set_classes.csv"
            ),
            "validation": os.path.join(
                self.data_home, "DCASE2022_task5_Validation_set_classes.csv"
            ),
        }

        metadata_index["class_codes"] = {}
        metadata_index["subdatasets"] = {}

        for split, metadata_path in metadata_paths.items():
            metadata_path = os.path.normpath(metadata_path)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError("Metadata not found. Did you run .download()?")

            with open(metadata_path, "r") as fhandle:
                reader = csv.reader(fhandle, delimiter=",")

                headers = next(reader)
                class_code_id = headers.index("class_code")
                class_name_id = headers.index("class_name")
                dataset_id = headers.index("dataset")

                for line in reader:
                    metadata_index["class_codes"][line[class_code_id]] = {
                        "subdataset": line[dataset_id],
                        "class_name": line[class_name_id],
                        "split": split,
                    }
                    if line[dataset_id] not in metadata_index["subdatasets"]:
                        metadata_index["subdatasets"][line[dataset_id]] = [
                            line[class_code_id]
                        ]
                    else:
                        metadata_index["subdatasets"][line[dataset_id]].append(
                            line[class_code_id]
                        )

        return metadata_index

    def plot_clip_durations(self):
        print("Clip Durations Analysis:")

        durations = [len(self.clip(c_id).audio[0]) / self.clip(c_id).audio[1] for c_id in self._index["clips"].keys()]

        # Calculating statistics
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)

        # Determine unit conversion (seconds or minutes)
        convert_to_minutes = mean_duration > 60 or median_duration > 60
        conversion_factor = 60 if convert_to_minutes else 1
        unit = "minutes" if convert_to_minutes else "seconds"

        durations = [d / conversion_factor for d in durations]
        mean_duration /= conversion_factor
        median_duration /= conversion_factor

        # Continue with statistics calculation
        std_deviation = np.std(durations)
        variance = np.var(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)
        q25, q75 = np.percentile(durations, [25, 75])
        range_duration = max_duration - min_duration
        iqr = q75 - q25

        # Create the main figure and the two axes
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, frame_on=False)
        ax2.axis('off')

        # Histogram
        n, bins, patches = ax1.hist(durations, bins=30, color='lightblue', edgecolor='black')

        mean_bin = np.digitize(mean_duration, bins)
        median_bin = np.digitize(median_duration, bins)
        patches[mean_bin-1].set_fc('red')
        patches[median_bin-1].set_fc('green')

        ax1.axvline(mean_duration, color='red', linestyle='dashed', linewidth=1)
        ax1.text(mean_duration + 0.2, max(n) * 0.9, 'Mean', color='red')
        ax1.axvline(median_duration, color='green', linestyle='dashed', linewidth=1)
        ax1.text(median_duration + 0.2, max(n) * 0.8, 'Median', color='green')

        ax1.set_title('Distribution of Clip Durations', fontsize = 8)
        ax1.set_xlabel(f'Duration ({unit})', fontsize = 8)
        ax1.set_ylabel('Number of Clips', fontsize = 8)
        ax1.grid(axis='y', alpha=0.75)

        analysis_results = (
            r"$\bf{Mean\ duration:}$" + f" {mean_duration:.2f} {unit}\n"
            r"$\bf{Median\ duration:}$" + f" {median_duration:.2f} {unit}\n"
            r"$\bf{Standard\ Deviation:}$" + f" {std_deviation:.2f} {unit}\n"
            r"$\bf{Variance:}$" + f" {variance:.2f}\n"
            r"$\bf{Min\ Duration:}$" + f" {min_duration:.2f} {unit}\n"
            r"$\bf{Max\ Duration:}$" + f" {max_duration:.2f} {unit}\n"
            r"$\bf{25th\ Percentile:}$" + f" {q25:.2f} {unit}\n"
            r"$\bf{75th\ Percentile:}$" + f" {q75:.2f} {unit}\n"
            r"$\bf{Range:}$" + f" {range_duration:.2f} {unit}\n"
            r"$\bf{IQR:}$" + f" {iqr:.2f} {unit}\n"
            r"$\bf{Total\ Clips:}$" + f" {len(self._index['clips'])}")
        ax2.text(0.1, 0.4, analysis_results, transform=ax2.transAxes, fontsize=10)
        
        plt.tight_layout()
        plt.show()

       
    def plot_subclasses_distribution(self):
        print("\nSubclasses Distribution Analysis:")

        if 'subdatasets' not in self._metadata:
            print("Subclass information not available.")
            return

        subclasses = [self._metadata[clip_id]['subdataset'] for clip_id in self._index["clips"]]
        unique_classes = set([label for clip_id in self._index["clips"] for label in self.clip(clip_id).events.labels])
        print(f"Total number of unique classes: {len(unique_classes)}")
        print(f"Subclasses: {self._metadata.get('subdatasets', 'None provided')}")

        # Plot
        plt.figure(figsize=(10, 6))
        sns.countplot(y=subclasses, order=pd.value_counts(subclasses).index)
        plt.title('Subclass distribution in the dataset')
        plt.xlabel('Count')
        plt.ylabel('Subclass')

        ax = plt.gca()
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height()/2),
                        ha='left', va='center', xytext=(5, 0), textcoords='offset points')

        plt.tight_layout()
        plt.show()
        print("\n")

    def plot_hierarchical_distribution(self):

        def plot_distribution(data, title, x_label, y_label, subplot_position):
            sns.countplot(y=data, order=pd.value_counts(data).index, palette="viridis", ax=axes[subplot_position])
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
        dataset_analysis_check = Checkbox(value=True, description='Dataset Statistics')
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
        line1, = ax1.plot([0, 0], [min(audio), max(audio)], color='r')

        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontsize(8)

        # Plotting the Mel spectrogram
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_title('Mel spectrogram', fontsize=8)
        ax2.set_xlim(0, duration)
        line2, = ax2.plot([0, 0], ax2.get_ylim(), color='white', linestyle='--')

        # Reduce font size for time and mel axis labels
        ax2.set_xlabel('Time (s)', fontsize=8)
        ax2.set_ylabel('Hz', fontsize=8)

        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontsize(8)

        plt.tight_layout()

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