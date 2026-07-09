import importlib
import os
import pkgutil

from .version import version as __version__
from .dcase_challenges import dcase_challenges as DCASE_CHALLENGES
from .core import DCASEWrapper

DATASETS = [
    d.name
    for d in pkgutil.iter_modules(
        [os.path.dirname(os.path.abspath(__file__)) + "/datasets"]
    )
]


def list_datasets():
    """Get a list of all soundata dataset names

    Returns:
        list: list of dataset names as strings
    """
    return DATASETS


def list_dataset_versions(dataset_name):
    """List the available versions of a dataset

    Returns:
        list: a list of available versions

    """
    if dataset_name not in DATASETS:
        raise ValueError("Invalid dataset {}".format(dataset_name))
    module = importlib.import_module("soundata.datasets.{}".format(dataset_name))
    return "Available versions for {}: {}. Default version: {}".format(
        dataset_name,
        [
            x
            for x in list(module.INDEXES.keys())
            if x not in ["default", "sample", "test"]
        ],
        module.INDEXES["default"],
    )


def initialize(dataset_name, data_home=None, version="default"):
    """Load a soundata dataset by name

    Example:
        .. code-block:: python

            urbansound8k = soundata.initialize('urbansound8k')  # get the urbansound8k dataset
            urbansound8k.download()  # download orchset
            urbansound8k.validate()  # validate orchset
            clip = urbansound8k.choice_clip()  # load a random clip
            print(clip)  # see what data a clip contains
            urbansound8k.clip_ids()  # load all clip ids

    Args:
        dataset_name (str): the dataset's name
            see soundata.DATASETS for a complete list of possibilities
        data_home (str or None): path where the data lives. If None
            uses the default location.
        version (str or None): which version of the dataset to load.
            If None, the default version is loaded.

    Returns:
        Dataset: a soundata.core.Dataset object

    """
    if dataset_name not in DATASETS:
        raise ValueError("Invalid dataset {}".format(dataset_name))

    module = importlib.import_module("soundata.datasets.{}".format(dataset_name))
    return module.Dataset(data_home=data_home, version=version)


def dcase_challenge(challenge_name, dictionary=DCASE_CHALLENGES, data_home=None):
    """
    Function to create DCASE wrapper

    Args:
        challenge_name (str): Name of the DCASE challenge
        dictionary (dict): Dictionary containing challenge configurations.

    Returns:
        DCASEWrapper: Initialized wrapper instance
    """

    if challenge_name not in dictionary:
        raise ValueError(f"Challenge '{challenge_name}' not found in dictionary")

    return DCASEWrapper(challenge_name, dictionary, data_home)
