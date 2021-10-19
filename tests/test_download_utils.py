import os
from pathlib import Path
import shutil
import zipfile
import re

from soundata import download_utils
from soundata.datasets import esc50

import pytest


@pytest.fixture
def mock_download_from_remote(mocker):
    return mocker.patch.object(download_utils, "download_from_remote")


@pytest.fixture
def mock_downloader(mocker):
    return mocker.patch.object(download_utils, "downloader")


@pytest.fixture
def mock_untar(mocker):
    return mocker.patch.object(download_utils, "untar")


@pytest.fixture
def mock_unzip(mocker):
    return mocker.patch.object(download_utils, "unzip")


@pytest.fixture
def mock_path(mocker, mock_download_from_remote):
    return mocker.patch.object(Path, "mkdir")


def test_downloader(mocker, mock_path):
    mock_zip = mocker.patch.object(download_utils, "download_zip_file")
    mock_tar = mocker.patch.object(download_utils, "download_tar_file")
    mock_download_from_remote = mocker.patch.object(
        download_utils, "download_from_remote"
    )
    mock_multipart_zip = mocker.patch.object(download_utils, "download_multipart_zip")

    zip_remote = download_utils.RemoteFileMetadata(
        filename="remote.zip", url="a", checksum=("1234")
    )
    multipart_zip_remote = [
        download_utils.RemoteFileMetadata(
            filename="remote.zip", url="a", checksum=("1234")
        ),
        download_utils.RemoteFileMetadata(
            filename="remote.z01", url="b", checksum=("2345")
        ),
    ]
    tar_remote = download_utils.RemoteFileMetadata(
        filename="remote.tar.gz", url="a", checksum=("1234")
    )

    file_remote = download_utils.RemoteFileMetadata(
        filename="remote.txt", url="a", checksum=("1234")
    )

    # Zip only
    download_utils.downloader("a", remotes={"b": zip_remote})
    mock_zip.assert_called_once_with(zip_remote, "a", False, False)
    mocker.resetall()

    # tar only
    download_utils.downloader("a", remotes={"b": tar_remote})
    mock_tar.assert_called_once_with(tar_remote, "a", False, False)
    mocker.resetall()

    # file only
    download_utils.downloader("a", remotes={"b": file_remote})
    mock_download_from_remote.assert_called_once_with(file_remote, "a", False)
    mocker.resetall()

    # zip and tar
    download_utils.downloader("a", remotes={"b": zip_remote, "c": tar_remote})
    mock_zip.assert_called_once_with(zip_remote, "a", False, False)
    mock_tar.assert_called_once_with(tar_remote, "a", False, False)
    mocker.resetall()

    # zip and file
    download_utils.downloader("a", remotes={"b": zip_remote, "c": file_remote})
    mock_zip.assert_called_once_with(zip_remote, "a", False, False)
    mock_download_from_remote.assert_called_once_with(file_remote, "a", False)
    mocker.resetall()

    # tar and file
    download_utils.downloader("a", remotes={"b": tar_remote, "c": file_remote})
    mock_tar.assert_called_once_with(tar_remote, "a", False, False)
    mock_download_from_remote.assert_called_once_with(file_remote, "a", False)
    mocker.resetall()

    # zip and tar and file
    download_utils.downloader(
        "a", remotes={"b": zip_remote, "c": tar_remote, "d": file_remote}
    )
    mock_zip.assert_called_once_with(zip_remote, "a", False, False)
    mock_download_from_remote.assert_called_once_with(file_remote, "a", False)
    mock_tar.assert_called_once_with(tar_remote, "a", False, False)
    mocker.resetall()

    # Zip multipart
    download_utils.downloader("a", remotes={"b": multipart_zip_remote})
    mock_multipart_zip.assert_called_once_with(multipart_zip_remote, "a", False, False)
    mocker.resetall()

    # test partial download
    download_utils.downloader(
        "a",
        remotes={"b": zip_remote, "c": tar_remote, "d": file_remote},
        partial_download=["b", "d"],
    )
    mock_zip.assert_called_once_with(zip_remote, "a", False, False)
    mock_download_from_remote.assert_called_once_with(file_remote, "a", False)
    mocker.resetall()

    # test bad type partial download
    with pytest.raises(ValueError):
        download_utils.downloader(
            "a",
            remotes={"b": zip_remote, "c": tar_remote, "d": file_remote},
            partial_download="b",
        )

    with pytest.raises(ValueError):
        download_utils.downloader(
            "a",
            remotes={"b": zip_remote, "c": tar_remote, "d": file_remote},
            partial_download=["d", "e"],
        )
    # test only multipart zip supported
    with pytest.raises(NotImplementedError):
        download_utils.downloader(
            "a",
            remotes={"b": [zip_remote, tar_remote, file_remote]},
        )

    # test info message
    download_utils.downloader("a", info_message="I am a message!")
    mocker.resetall()

    # test download twice - defaults
    download_utils.downloader(
        "a", remotes={"b": zip_remote, "c": tar_remote, "d": file_remote}
    )
    download_utils.downloader(
        "a", remotes={"b": zip_remote, "c": tar_remote, "d": file_remote}
    )

    # test download twice - cleanup=True
    download_utils.downloader(
        "a", remotes={"b": zip_remote, "c": tar_remote, "d": file_remote}, cleanup=True
    )
    download_utils.downloader(
        "a", remotes={"b": zip_remote, "c": tar_remote, "d": file_remote}
    )


def _clean(fpath):
    if os.path.exists(fpath):
        shutil.rmtree(fpath)


def test_downloader_with_server_file(httpserver):

    httpserver.serve_content(open("tests/resources/remote.wav").read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename="remote.wav",
        url=httpserver.url,
        checksum=("3f77d0d69dc41b3696f074ad6bf2852f"),
    )

    save_dir = "tests/resources/tmp_download_test"

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})
    # test downloading twice
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE}, cleanup=True)
    # test downloading twice
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})
    # test downloading twice
    download_utils.downloader(
        save_dir, remotes={"b": TEST_REMOTE}, force_overwrite=True
    )

    _clean(save_dir)


def test_downloader_with_server_zip(httpserver):

    httpserver.serve_content(open("tests/resources/remote.zip", "rb").read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename="remote.zip",
        url=httpserver.url,
        checksum=("7a31ccfa28bfa3fb112d16c96e9d9a89"),
    )

    save_dir = "tests/resources/_tmp_test_download_utils"

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})
    # test downloading twice
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE}, cleanup=True)
    # test downloading twice
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})
    # test downloading twice
    download_utils.downloader(
        save_dir, remotes={"b": TEST_REMOTE}, force_overwrite=True
    )

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE}, cleanup=True)
    # test downloading twice
    download_utils.downloader(
        save_dir, remotes={"b": TEST_REMOTE}, force_overwrite=True
    )

    _clean(save_dir)


def test_downloader_with_server_tar(httpserver):

    httpserver.serve_content(open("tests/resources/remote.tar.gz", "rb").read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename="remote.tar.gz",
        url=httpserver.url,
        checksum=("9042f5eebdcd0b94aa7a3c9bf12dc51d"),
    )

    save_dir = "tests/resources/_tmp_test_download_utils"

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})
    # test downloading twice
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE}, cleanup=True)
    # test downloading twice
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})

    _clean(save_dir)
    download_utils.downloader(save_dir, remotes={"b": TEST_REMOTE})
    # test downloading twice
    download_utils.downloader(
        save_dir, remotes={"b": TEST_REMOTE}, force_overwrite=True
    )

    _clean(save_dir)


def test_download_from_remote(httpserver, tmpdir):

    httpserver.serve_content(open("tests/resources/remote.wav").read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename="remote.wav",
        url=httpserver.url,
        checksum=("3f77d0d69dc41b3696f074ad6bf2852f"),
    )

    download_path = download_utils.download_from_remote(TEST_REMOTE, str(tmpdir), False)


def test_download_from_remote_destdir(httpserver, tmpdir):
    httpserver.serve_content(open("tests/resources/remote.wav").read())

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename="remote.wav",
        url=httpserver.url,
        checksum=("3f77d0d69dc41b3696f074ad6bf2852f"),
        destination_dir="subfolder",
    )

    download_path = download_utils.download_from_remote(TEST_REMOTE, str(tmpdir), False)
    expected_download_path = os.path.join(str(tmpdir), "subfolder", "remote.wav")
    assert expected_download_path == download_path


def test_download_from_remote_raises_IOError(httpserver, tmpdir):
    httpserver.serve_content("File not found!", 404)

    TEST_REMOTE = download_utils.RemoteFileMetadata(
        filename="remote.wav", url=httpserver.url, checksum=("1234")
    )

    with pytest.raises(IOError):
        download_utils.download_from_remote(TEST_REMOTE, str(tmpdir), False)


def test_unpackdir(httpserver):

    data_home = "tests/resources/sound_datasets/esc50_download"
    if os.path.exists(data_home):
        shutil.rmtree(data_home)

    # download the full dataset
    httpserver.serve_content(
        open("tests/resources/download/ESC-50-master.zip", "rb").read()
    )

    remotes = {
        "all": download_utils.RemoteFileMetadata(
            filename="ESC-50-master.zip",
            url=httpserver.url,
            checksum=("91281932a1c5ff0473f103c628a53d77"),
            unpack_directories=["ESC-50-master"],
        )
    }
    dataset = esc50.Dataset(data_home)
    dataset.remotes = remotes
    dataset.download(None, False, False)

    assert os.path.exists(data_home)
    assert not os.path.exists(os.path.join(data_home, "ESC-50-master"))

    assert os.path.exists(os.path.join(data_home, "requirements.txt"))
    assert os.path.exists(
        os.path.join(
            data_home,
            "audio/1-137-A-32.wav",
        )
    )


def test_unzip():
    download_utils.unzip("tests/resources/file.zip", cleanup=False)
    expected_file_location = os.path.join("tests", "resources", "file.txt")
    assert os.path.exists(expected_file_location)
    os.remove(expected_file_location)


def test_untar():
    download_utils.untar("tests/resources/file.tar.gz", cleanup=False)
    expected_file_location = os.path.join("tests", "resources", "file", "file.txt")
    assert os.path.exists(expected_file_location)
    os.remove(expected_file_location)


def test_download_zip_file(mocker, mock_download_from_remote, mock_unzip):
    mock_download_from_remote.return_value = "foo"
    download_utils.download_zip_file("a", "b", False, False)

    mock_download_from_remote.assert_called_once_with("a", "b", False)
    mock_unzip.assert_called_once_with("foo", cleanup=False)
    _clean("a")


def test_download_multipart_zip(mocker, mock_download_from_remote, mock_unzip):
    Path("tests/resources/foo.zip").touch()
    Path("tests/resources/foo.z01").touch()
    multipart_zip_remote = {
        "foo": [
            download_utils.RemoteFileMetadata(
                filename="foo.zip", url="a", checksum=("1234")
            ),
            download_utils.RemoteFileMetadata(
                filename="foo.z01", url="b", checksum=("2345")
            ),
        ]
    }
    download_utils.downloader(
        "tests/resources", multipart_zip_remote, force_overwrite=False, cleanup=True
    )
    mock_download_from_remote.assert_has_calls(
        [
            mocker.call(multipart_zip_remote["foo"][0], "tests/resources", False),
            mocker.call(multipart_zip_remote["foo"][1], "tests/resources", False),
        ]
    )
    mock_unzip.assert_called_once_with("tests/resources/foo_single.zip", cleanup=True)


def test_download_tar_file(mocker, mock_download_from_remote, mock_untar):
    mock_download_from_remote.return_value = "foo"
    download_utils.download_tar_file("a", "b", False, False)

    mock_download_from_remote.assert_called_once_with("a", "b", False)
    mock_untar.assert_called_once_with("foo", cleanup=False)
    _clean("a")


def test_extractall_unicode(mocker, mock_download_from_remote, mock_unzip):
    zfile = zipfile.ZipFile("tests/resources/utfissue.zip", "r")
    download_utils.extractall_unicode(zfile, os.path.dirname("tests/resources/"))
    zfile.close()
    expected_files = ["pic👨‍👩‍👧‍👦🎂.jpg", "Benoît.txt"]
    for expected_file in expected_files:
        expected_file_location = os.path.join("tests", "resources", expected_file)
        assert os.path.exists(expected_file_location)
        os.remove(expected_file_location)


def test_extractall_cp437(mocker, mock_download_from_remote, mock_unzip):
    zfile = zipfile.ZipFile("tests/resources/utfissue.zip", "r")
    zfile.extractall(os.path.dirname("tests/resources/"))
    zfile.close()
    expected_files = ["pic👨‍👩‍👧‍👦🎂.jpg", "Benoît.txt"]
    for expected_file in expected_files:
        expected_file_location = os.path.join("tests", "resources", expected_file)
        assert not os.path.exists(expected_file_location)
    true_files = [
        file
        for file in os.listdir(os.path.join("tests", "resources"))
        if re.match(r"(pic.*).jpg", file) or re.match(r"(Beno.*).txt", file)
    ]
    for true_file in true_files:
        true_file_location = os.path.join("tests", "resources", true_file)
        os.remove(true_file_location)
    shutil.rmtree(os.path.join("tests", "resources", "__MACOSX"))
