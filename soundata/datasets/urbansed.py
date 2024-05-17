"""URBAN-SED Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    **URBAN-SED**

        | URBAN-SED (c) by Justin Salamon, Duncan MacConnell, Mark Cartwright, Peter Li, and Juan Pablo Bello.
        | URBAN-SED is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).
        | You should have received a copy of the license along with this work. If not, see <http://creativecommons.org/licenses/by/4.0/>.

    *Created By:*

        | Justin Salamon*^, Duncan MacConnell*, Mark Cartwright*, Peter Li*, and Juan Pablo Bello*.
        | * Music and Audio Research Lab (MARL), New York University, USA
        | ^ Center for Urban Science and Progress (CUSP), New York University, USA
        | http://urbansed.weebly.co
        | http://steinhardt.nyu.edu/marl/
        | http://cusp.nyu.edu/

    *Version 2.0.0*
        * Audio files generated with scaper v0.1.0 (identical to audio in URBAN-SED 1.0)
        * Jams annotation files generated with scaper v0.1.0 and updated to comply with scaper v1.0.0 (namespace changed from "sound_event" to "scaper")
        * NOTE: due to updates to the scaper library, regenerating the audio from the jams annotations using scaper >=1.0.0 will result in audio files that are highly similar, but not identical, to the audio files provided. This is because the provided audio files were generated with scaper v0.1.0 and have been purposely kept the same as in URBAN-SED v1.0 to ensure comparability to previously published results.


    *Description*

        URBAN-SED is a dataset of 10,000 soundscapes with sound event annotations generated using scaper (github.com/justinsalamon/scaper).

        A detailed description of the dataset is provided in the following article:

        .. code-block:: latex

            J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello. "Scaper: A Library for Soundscape Synthesis and Augmentation", 
            In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017.

        A summary is provided here:

        * The dataset includes 10,000 soundscapes, totals almost 30 hours and includes close to 50,000 annotated sound events
        * Complete annotations are provided in JAMS format, and simplified annotations are provided as tab-separated text files
        * Every soundscape is 10 seconds long and has a background of Brownian noise resembling the typical "hum" often heard in urban environments
        * Every soundscape contains between 1-9 sound events from the following classes:
            * air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren and street_music
        * The source material for the sound events are the clips from the UrbanSound8K dataset
        * URBAN-SED comes pre-sorted into three sets: train, validate and test:
            * There are 6000 soundscapes in the training set, generated using clips from folds 1-6 in UrbanSound8K
            * There are 2000 soundscapes in the validation set,  generated using clips from folds 7-8 in UrbanSound8K
            * There are 2000 soundscapes in the test set, generated using clips from folds 9-10 in UrbanSound8K
        * Further details about how the soundscapes were generated including the distribution of sound event start times, durations, signal-to-noise ratios, pitch shifting, time stretching, and the range of sound event polyphony (overlap) can be found in Section 3 of the aforementioned scaper paper 
        * The scripts used to generated URBAN-SED using scaper can be found here: https://github.com/justinsalamon/scaper_waspaa2017/tree/master/notebooks


    *Audio Files Included*
        * 10,000 synthesized soundscapes in single channel (mono), 44100Hz, 16-bit, WAV format.
        * The files are split into a training set (6000), validation set (2000) and test set (2000).


    *Annotation Files Included*
        The annotations list the sound events that occur in every soundscape. The annotations are "strong", meaning for every 
        sound event the annotations include (at least) the start time, end time, and label of the sound event. Sound events 
        come from the following 10 labels (categories): air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling,
        gun_shot, jackhammer, siren, street_music

        There are two types of annotations: full annotations in JAMS format, and simplified annotations in 
        tab-separated txt format.

    *JAMS Annotations*
        * The full annotations are distributed in JAMS format (https://github.com/marl/jams).
        * There are 10,000 JAMS annotation files, each one corresponding to a single soundscape with the same filename (other than the extension)
        * Each JAMS file contains a single annotation in the scaper namespace format - jams >=v0.3.2 is required in order to load the annotation into python with jams:

        import jams
        jam = jams.load('soundscape_train_bimodal0.jams').

        * The value of each observation (sound event) is a dictionary storing all scaper-related sound event parameters:
            * label, source_file, source_time, event_time, event_duration, snr, role, pitch_shift, time_stretch.
            * Note: the event_duration stored in the value dictionary represents the specified duration prior to any time stretching. The actual event durtation in the soundscape is stored in the duration field of the JAMS observation.
        * The observations (sound events) in the JAMS annotation include both foreground sound events and the background(s).
        * The probabilistic scaper foreground and background event specifications are stored in the annotation's sandbox, allowing a complete reconstruction of the soundscape audio from the JAMS annotation (assuming access to the original source material) using scaper.generate_from_jams('soundscape_train_bimodal0.jams').
        * The annotation sandbox also includes additional metadata such as the total number of foreground sound events, the maximum polyphony (sound event overlap) of the soundscape and its gini coefficient (a measure of soundscape complexity).

    *Simplified Annotations*
        * The simplified annotations are distributed as tab-separated text files.
        * There are 10,000 simplified annotation files, each one corresponding to a single soundscape with the same filename (other than the extension)
        * Each simplified annotation has a 3-column format (no header): start_time, end_time, label.
        * Background sounds are NOT included in the simplified annotations (only foreground sound events)
        * No additional information is stored in the simplified events (see the JAMS annotations for more details).


    *Please Acknowledge URBAN-SED in Academic Research*
        When URBAN-SED is used for academic research, we would highly appreciate it if scientific publications of works 
        partly based on the URBAN-SED dataset cite the following publication:

        .. code-block:: latex

            J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello. "Scaper: A Library for Soundscape Synthesis and Augmentation", 
            In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017.

        The creation of this dataset was supported by NSF award 1544753.


    *Conditions of Use*
        Dataset created by J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello. Audio files contain excerpts of 
        recordings uploaded to www.freesound.org. Please see FREESOUNDCREDITS.txt for an attribution list.
        
        The URBAN-SED dataset is offered free of charge under the terms of the Creative Commons
        Attribution 4.0 International License (CC BY 4.0): http://creativecommons.org/licenses/by/4.0/
        
        The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including 
        without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or 
        completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, NYU is not 
        liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of 
        the URBAN-SED dataset or any part of it.


    *Feedback*
        | Please help us improve URBAN-SED by sending your feedback to: justin.salamon@nyu.edu
        | In case of a problem report please include as many details as possible.

"""

import os
from typing import BinaryIO, Optional, TextIO, Tuple

import librosa
import numpy as np
import csv
import jams
import glob

from soundata import download_utils
from soundata import jams_utils
from soundata import core
from soundata import annotations
from soundata import io


BIBTEX = """
@inproceedings{Salamon:Scaper:WASPAA:17,
	Address = {New Paltz, NY, USA},
	Author = {Salamon, J. and MacConnell, D. and Cartwright, M. and Li, P. and Bello, J.~P.},
	Booktitle = {IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
	Month = {Oct.},
	Pages = {344--348},
	Title = {Scaper: A Library for Soundscape Synthesis and Augmentation},
	Year = {2017}}
"""

INDEXES = {
    "default": "2.0",
    "test": "sample",
    "2.0": core.Index(
        filename="urbansed_index_2.0.json",
        url="https://zenodo.org/records/11176925/files/urbansed_index_2.0.json?download=1",
        checksum="95e7a152729ebdb386356e7d21b342b7",
    ),
    "sample": core.Index(filename="urbansed_index_2.0_sample.json"),
}

REMOTES = {
    "all": download_utils.RemoteFileMetadata(
        filename="URBAN-SED_v2.0.0.tar.gz",
        url="https://zenodo.org/record/1324404/files/URBAN-SED_v2.0.0.tar.gz?download=1",
        checksum="a2d24a2148ece7c021fcc079ee87c2dc",
        unpack_directories=["URBAN-SED_v2.0.0"],
    )
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International"


class Clip(core.Clip):
    """URBAN-SED Clip class

    Args:
        clip_id (str): id of the clip

    Attributes:
        audio (np.ndarray, float): path to the audio file
        audio_path (str): path to the audio file
        clip_id (str): clip id
        events (soundata.annotations.Events): sound events with start time, end time, label and confidence
        split (str): subset the clip belongs to (for experiments): train, validate, or test
    """

    def __init__(self, clip_id, data_home, dataset_name, index, metadata):
        super().__init__(clip_id, data_home, dataset_name, index, metadata)

        self.audio_path = self.get_path("audio")
        self.jams_path = self.get_path("jams")
        self.txt_path = self.get_path("txt")

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

    @core.cached_property
    def events(self) -> Optional[annotations.Events]:
        """The audio events

        Returns
            * annotations.Events - audio event object

        """
        return load_events(self.txt_path)

    def to_jams(self):
        """Get the clip's data in jams format

        Returns:
            jams.JAMS: the clip's data in jams format

        """
        jam = jams.load(self.jams_path)
        jam.annotations[0].annotation_metadata.data_source = "soundata"
        return jam


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO, sr=None) -> Tuple[np.ndarray, float]:
    """Load a UrbanSound8K audio file.

    Args:
        fhandle (str or file-like): File-like object or path to audio file
        sr (int or None): sample rate for loaded audio, None by default, which
            uses the file's original sample rate of 44100 without resampling.

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    audio, sr = librosa.load(fhandle, sr=sr, mono=True)
    return audio, sr


@io.coerce_to_string_io
def load_events(fhandle: TextIO) -> annotations.Events:
    """Load an URBAN-SED sound events annotation file
    Args:
        fhandle (str or file-like): File-like object or path to the sound events annotation file
    Raises:
        IOError: if txt_path doesn't exist
    Returns:
        Events: sound events annotation data
    """

    times = []
    labels = []
    confidence = []
    reader = csv.reader(fhandle, delimiter="\t")
    for line in reader:
        times.append([float(line[0]), float(line[1])])
        labels.append(line[2])
        confidence.append(1.0)

    events_data = annotations.Events(
        np.array(times), "seconds", labels, "open", np.array(confidence)
    )
    return events_data


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The URBAN-SED dataset
    """

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="urbansed",
            clip_class=Clip,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_audio)
    def load_audio(self, *args, **kwargs):
        return load_audio(*args, **kwargs)

    @core.cached_property
    def _metadata(self):
        splits = ["train", "validate", "test"]
        expected_sizes = [6000, 2000, 2000]
        metadata_index = {}

        for split, es in zip(splits, expected_sizes):
            annotation_folder = os.path.join(self.data_home, "annotations", split)
            txtfiles = sorted(glob.glob(os.path.join(annotation_folder, "*.txt")))

            for tf in txtfiles:
                clip_id = os.path.basename(tf).replace(".txt", "")
                metadata_index[clip_id] = {"split": split}

        return metadata_index
