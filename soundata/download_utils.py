"""utilities for downloading from the web.
"""

import glob
import logging
import os
import shutil
import tarfile
import urllib
import zipfile
import subprocess
import py7zr
from tqdm import tqdm

from soundata.validate import md5

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)


class RemoteFileMetadata(object):
    """The metadata for a remote file

    Attributes:
        filename (str): the remote file's basename
        url (str): the remote file's url
        checksum (str): the remote file's md5 checksum
        destination_dir (str or None): the relative path for where to save the file
        unpack_directories (list or None): list of relative directories. For each directory
            the contents will be moved to destination_dir (or data_home if not provided)

    """

    def __init__(
        self, filename, url, checksum, destination_dir=None, unpack_directories=None
    ):
        self.filename = filename
        self.url = url
        self.checksum = checksum
        self.destination_dir = destination_dir
        self.unpack_directories = unpack_directories


def downloader(
    save_dir,
    remotes=None,
    index=None,
    partial_download=None,
    info_message=None,
    force_overwrite=False,
    cleanup=False,
):
    """Download data to `save_dir` and optionally log a message

    Args:
        save_dir (str):
            The directory to download the data
        remotes (dict or None):
            A dictionary of RemoteFileMetadata tuples of data in zip format.
            If an element of the dictionary is a list of RemoteFileMetadata, it is handled as a multipart zip file
            If None, there is no data to download
        index (core.Index):
            A soundata Index class, which contains a remote index to be downloaded
            or a subset of remotes to download by default.
        partial_download (list or None):
            A list of keys to partially download the remote objects of the download dict.
            If None, all data specified by the index is downloaded
        info_message (str or None):
            A string of info to log when this function is called.
            If None, no string is logged.
        force_overwrite (bool):
            If True, existing files are overwritten by the downloaded files.
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not index:
        raise ValueError("Index must be specified.")

    if cleanup:
        logging.warning(
            "Zip and tar files will be deleted after they are uncompressed. "
            + "If you download this dataset again, it will overwrite existing files, even if force_overwrite=False"
        )

    if index.remote:
        if remotes is None:
            remotes = {}
        remotes["index"] = index.remote

    # if partial download is specified, use it. Otherwise, use the
    # partial download specified by the index.
    partial_download = partial_download if partial_download else index.partial_download

    if remotes is not None:
        if partial_download is not None:
            # check the keys in partial_download are in the download dict
            if not isinstance(partial_download, list) or any(
                [k not in remotes for k in partial_download]
            ):
                raise ValueError(
                    "partial_download must be a list which is a subset of {}, but got {}".format(
                        list(remotes.keys()), partial_download
                    )
                )
            objs_to_download = partial_download
        else:
            objs_to_download = list(remotes.keys())

        if "index" in objs_to_download and len(objs_to_download) > 1:
            logging.info(
                "Downloading {}. Index is being stored in {}, and the rest of files in {}".format(
                    objs_to_download, index.indexes_dir, save_dir
                )
            )
        elif "index" in objs_to_download and len(objs_to_download) == 1:
            logging.info(
                "Downloading {}. Index is being stored in {}".format(
                    objs_to_download, index.indexes_dir
                )
            )
        else:
            logging.info("Downloading {} to {}".format(objs_to_download, save_dir))

        for k in objs_to_download:
            if isinstance(remotes[k], list):
                if all([remote.filename[-4:-2] == ".z" for remote in remotes[k]]):
                    download_multipart_zip(
                        remotes[k], save_dir, force_overwrite, cleanup
                    )
                else:
                    raise NotImplementedError("Only multipart zip supported.")

            else:
                logging.info("[{}] downloading {}".format(k, remotes[k].filename))
                extension = os.path.splitext(remotes[k].filename)[-1]
                if ".zip" in extension:
                    download_zip_file(remotes[k], save_dir, force_overwrite, cleanup)
                elif ".gz" in extension or ".tar" in extension or ".bz2" in extension:
                    download_tar_file(remotes[k], save_dir, force_overwrite, cleanup)
                elif ".7z" in extension:
                    download_7z_file(remotes[k], save_dir, force_overwrite, cleanup)
                else:
                    download_from_remote(remotes[k], save_dir, force_overwrite)

                if remotes[k].unpack_directories:
                    for src_dir in remotes[k].unpack_directories:
                        # path to destination directory
                        destination_dir = (
                            os.path.join(save_dir, remotes[k].destination_dir)
                            if remotes[k].destination_dir
                            else save_dir
                        )
                        # path to directory to unpack
                        source_dir = os.path.join(destination_dir, src_dir)

                        if not os.path.exists(source_dir):
                            logging.info(
                                "Data not downloaded, because it probably already exists on your computer. "
                                + "Run .validate() to check, or rerun with force_overwrite=True to delete any "
                                + "existing files and download from scratch"
                            )
                            return

                        move_directory_contents(source_dir, destination_dir)

    if info_message is not None:
        logging.info(info_message.format(save_dir))


class DownloadProgressBar(tqdm):
    """Wrap tqdm to show download progress"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_multipart_zip(zip_remotes, save_dir, force_overwrite, cleanup):
    """Download and unzip a multipart zip file.

    Args:
        zip_remotes (list):
            A list of RemoteFileMetadata Objects
            containing download information
        save_dir (str):
            Path to save downloaded file
        force_overwrite (bool):
            If True, overwrites existing files
        cleanup (bool):
            If True, remove zipfile after unziping

    """
    for l in range(len(zip_remotes)):
        download_from_remote(zip_remotes[l], save_dir, force_overwrite)
    zip_path = os.path.join(
        save_dir,
        next((part.filename for part in zip_remotes if ".zip" in part.filename), None),
    )
    out_path = zip_path.replace(".zip", "_single.zip")
    subprocess.run(["zip", "-s", "0", zip_path, "--out", out_path], shell=True)
    if cleanup:
        for l in range(len(zip_remotes)):
            zip_path = os.path.join(save_dir, zip_remotes[l].filename)
            os.remove(zip_path)
    unzip(out_path, cleanup=cleanup)


def download_from_remote(remote, save_dir, force_overwrite):
    """Download a remote dataset into path

    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the MD5 Checksum of the
    downloaded file.

    Adapted from scikit-learn's sklearn.datasets.base._fetch_remote.

    Args:
        remote (RemoteFileMetadata): Named tuple containing remote dataset
            meta information: url, filename and checksum
        save_dir (str): Directory to save the file to. Usually `data_home`
        force_overwrite  (bool):
            If True, overwrite existing file with the downloaded file.
            If False, does not overwrite, but checks that checksum is consistent.

    Returns:
        str: Full path of the created file.

    """
    if remote.destination_dir is None:
        download_dir = save_dir
    else:
        download_dir = os.path.join(save_dir, remote.destination_dir)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    download_path = os.path.join(download_dir, remote.filename)

    if not os.path.exists(download_path) or force_overwrite:
        # if we got here, we want to overwrite any existing file
        if os.path.exists(download_path):
            os.remove(download_path)

        # If file doesn't exist or we want to overwrite, download it
        with DownloadProgressBar(
            unit="B", unit_scale=True, unit_divisor=1024, miniters=1
        ) as t:
            try:
                urllib.request.urlretrieve(
                    remote.url,
                    filename=download_path,
                    reporthook=t.update_to,
                    data=None,
                )
            except Exception as exc:
                error_msg = """
                            soundata failed to download the dataset from {}!
                            Please try again in a few minutes.
                            If this error persists, please raise an issue at
                            https://github.com/soundata/soundata,
                            and tag it with 'broken-link'.
                            """.format(
                    remote.url
                )
                logging.error(error_msg)
                raise exc
    else:
        logging.info(
            "{} already exists and will not be downloaded. ".format(download_path)
            + "Rerun with force_overwrite=True to delete this file and force the download."
        )

    checksum = md5(download_path)
    if remote.checksum != checksum:
        raise IOError(
            "{} has an MD5 checksum ({}) "
            "differing from expected ({}), "
            "file may be corrupted.".format(download_path, checksum, remote.checksum)
        )
    return download_path


def download_zip_file(zip_remote, save_dir, force_overwrite, cleanup):
    """Download and unzip a zip file.

    Args:
        zip_remote (RemoteFileMetadata):
            Object containing download information
        save_dir (str):
            Path to save downloaded file
        force_overwrite (bool):
            If True, overwrites existing files
        cleanup (bool):
            If True, remove zipfile after unziping

    """
    zip_download_path = download_from_remote(zip_remote, save_dir, force_overwrite)
    unzip(zip_download_path, cleanup=cleanup)


def extractall_unicode(zfile, out_dir):
    """Extract all files inside a zip archive to a output directory.

    In comparison to the zipfile, it checks for correct file name encoding

    Args:
        zfile (obj): Zip file object created with zipfile.ZipFile
        out_dir (str): Output folder

    """
    for m in zfile.infolist():
        data = zfile.read(m)  # extract zipped data into memory

        try:
            decoded_name = m.filename.encode("cp437").decode()
        except UnicodeEncodeError:
            decoded_name = m.filename

        disk_file_name = os.path.join(out_dir, decoded_name)

        dir_name = os.path.dirname(disk_file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not os.path.isdir(disk_file_name):
            with open(disk_file_name, "wb") as fd:
                fd.write(data)


def unzip(zip_path, cleanup):
    """Unzip a zip file inside it's current directory.

    Args:
        zip_path (str): Path to zip file
        cleanup (bool): If True, remove zipfile after unzipping

    """
    zfile = zipfile.ZipFile(zip_path, "r")
    extractall_unicode(zfile, os.path.dirname(zip_path))
    zfile.close()
    if cleanup:
        os.remove(zip_path)


def download_7z_file(tar_remote, save_dir, force_overwrite, cleanup):
    """Download and untar a tar file.

    Args:
        tar_remote (RemoteFileMetadata): Object containing download information
        save_dir (str): Path to save downloaded file
        force_overwrite (bool): If True, overwrites existing files
        cleanup (bool): If True, remove tarfile after untarring

    """
    _7z_download_path = download_from_remote(tar_remote, save_dir, force_overwrite)
    un7z(_7z_download_path, cleanup=cleanup)


def un7z(sevenz_path, cleanup):
    """Unzip a 7z file inside its current directory.

    Args:
        sevenz_path (str): Path to the 7z file
        cleanup (bool): If True, remove 7z file after extraction

    """
    with py7zr.SevenZipFile(sevenz_path, mode="r") as z:
        z.extractall(path=os.path.dirname(sevenz_path))
    if cleanup:
        os.remove(sevenz_path)


def download_tar_file(tar_remote, save_dir, force_overwrite, cleanup):
    """Download and untar a tar file.

    Args:
        tar_remote (RemoteFileMetadata): Object containing download information
        save_dir (str): Path to save downloaded file
        force_overwrite (bool): If True, overwrites existing files
        cleanup (bool): If True, remove tarfile after untarring

    """
    tar_download_path = download_from_remote(tar_remote, save_dir, force_overwrite)
    untar(tar_download_path, cleanup=cleanup)


def untar(tar_path, cleanup):
    """Untar a tar file inside it's current directory.

    Args:
        tar_path (str): Path to tar file
        cleanup (bool): If True, remove tarfile after untarring

    """
    tfile = tarfile.open(tar_path, "r")
    tfile.extractall(os.path.dirname(tar_path))
    tfile.close()
    if cleanup:
        os.remove(tar_path)


def move_directory_contents(source_dir, target_dir):
    """Move the contents of source_dir into target_dir, and delete source_dir

    Args:
        source_dir (str): path to source directory
        target_dir (str): path to target directory

    """
    directory_contents = glob.glob(os.path.join(source_dir, "*"))
    for fpath in directory_contents:
        target_path = os.path.join(target_dir, os.path.basename(fpath))
        if os.path.exists(target_path):
            logging.info(
                "{} already exists. Run with force_overwrite=True to download from scratch".format(
                    target_path
                )
            )
            continue
        shutil.move(fpath, target_dir)

    shutil.rmtree(source_dir)
