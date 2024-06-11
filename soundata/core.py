"""Core soundata classes
"""

import json
import os
import sys
import random
import types
from typing import Any, List, Optional

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
    """Cached property decorator

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
        version (str): dataset version
        name (str): the identifier of the dataset
        bibtex (str or None): dataset citation/s in bibtex format
        indexes (dict or None): indexes to be downloaded
        remotes (dict or None): data to be downloaded
        readme (str): information about the dataset
        clip (function): a function mapping a clip_id to a soundata.core.Clip
        clipgroup (function): a function mapping a clipgroup_id to a soundata.core.Clipgroup

    """

    def __init__(
        self,
        data_home=None,
        version="default",
        name=None,
        clip_class=None,
        clipgroup_class=None,
        bibtex=None,
        indexes=None,
        remotes=None,
        download_info=None,
        license_info=None,
    ):
        """Dataset init method

        Args:
            data_home (str or None): path where soundata will look for the dataset
            version (str): dataset version
            name (str or None): the identifier of the dataset
            clip_class (soundata.core.Clip or None): a Clip class
            clipgroup_class (soundata.core.Clipgroup or None): a Clipgroup class
            bibtex (str or None): dataset citation/s in bibtex format
            remotes (dict or None): data to be downloaded
            download_info (str or None): download instructions or caveats
            license_info (str or None): license of the dataset

        """
        self.name = name
        self.data_home = self.default_path if data_home is None else data_home

        if version not in indexes:
            raise ValueError(
                "Invalid version {}. Must be one of {}.".format(version, indexes.keys())
            )
        if isinstance(indexes[version], str):
            self.version = indexes[version]
        else:
            self.version = version
        self._index_data = indexes[self.version]
        self.index_path = self._index_data.get_path()
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
        try:
            with open(self.index_path, encoding="utf-8") as fhandle:
                index = json.load(fhandle)
        except FileNotFoundError:
            if self._index_data.remote:
                raise FileNotFoundError(
                    "This dataset's index must be downloaded. Did you run .download()?"
                )
            raise FileNotFoundError(
                f"Dataset index for {self.name} was expected "
                + "but not found. Make sure your sample indexes for testing are in soundata/tests/indexes/"
            )

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
            index=self._index_data,
            partial_download=partial_download,
            info_message=self._download_info,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )

    def explore_dataset(self, clip_id=None):  # pragma: no cover
        """Explore the dataset for a given clip_id or a random clip if clip_id is None.

        Args:
            clip_id (str or None):
                The identifier of the clip to explore. If None, a random clip will be chosen.

        """
        try:
            from soundata import display_plot_utils

            display_plot_utils.perform_dataset_exploration(self, clip_id)
        except ModuleNotFoundError:
            sys.exit(
                """Dependencies for display utils not found. Did you install plotting optional dependencies? Please run pip install soundata"[plots]" """
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


class Index(object):
    """Class for storing information about dataset indexes.

    Args:
        filename (str): The index filename (not path), e.g. "example_dataset_index_1.2.json"
        url (str or None): None if index is not remote, or a url to download from
        checksum (str or None): None if index is not remote, or the md5 checksum of the file
        partial_download (list or None): if provided, specifies a subset of Dataset.remotes
            corresponding to this index to be downloaded. If None, all Dataset.remotes will
            be downloaded when calling Dataset.download()

    Attributes:
        remote (download_utils.RemoteFileMetadata or None): None if index is not remote, or
            a RemoteFileMetadata object
        partial_download (list or None): a list of keys to partially download, or None

    """

    def __init__(
        self,
        filename: str,
        url: Optional[str] = None,
        checksum: Optional[str] = None,
        partial_download: Optional[List[str]] = None,
    ):
        self.filename = filename
        self.remote: Optional[download_utils.RemoteFileMetadata]
        self.indexes_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "datasets",
            "indexes",
        )
        if url and checksum:
            self.remote = download_utils.RemoteFileMetadata(
                filename=filename,
                url=url,
                checksum=checksum,
                destination_dir=self.indexes_dir,
            )
        elif url or checksum:
            raise ValueError(
                "Remote indexes must have both a url and a checksum specified."
            )
        else:
            self.indexes_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                "tests",
                "indexes",
            )
            self.remote = None

        self.partial_download = partial_download

    def get_path(self) -> str:
        """Get the absolute path to the index file

        Returns:
            str: absolute path to the index file
        """
        return os.path.join(self.indexes_dir, self.filename)
