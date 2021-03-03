import importlib
import os
import pkgutil

from .version import version as __version__


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


def initialize(dataset_name, data_home=None):
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

    Returns:
        Dataset: a soundata.core.Dataset object

    """
    if dataset_name not in DATASETS:
        raise ValueError("Invalid dataset {}".format(dataset_name))

    module = importlib.import_module("soundata.datasets.{}".format(dataset_name))
    return module.Dataset(data_home=data_home)
