import argparse
import glob
import json
import os
from soundata.validate import md5
from tqdm import tqdm

DATASET_INDEX_PATH = "../soundata/datasets/indexes/singapura_index.json"


def make_dataset_index(dataset_data_path):
    audio_dir = os.path.join(dataset_data_path, "labelled")
    audio_files = sorted(glob.glob(os.path.join(audio_dir, "*", "*.flac")))

    label_dir = os.path.join(dataset_data_path, "labels_public")

    track_ids = [os.path.normpath(f).split(os.sep)[-2:] for f in audio_files]
    label_files = [tid[-1].replace(".flac", ".csv") for tid in track_ids]
    #will be fixed in the next version
    label_files = [
        "[b827eb7d576e][2020-08-03T23-32-11Z][manual][---][565a40f866f3d2804332ca7896a4c77d][93.csv"
        if "565a40f866f3d2804332ca7896a4c77d" in lf
        else lf
        for lf in label_files
    ]
    audio_files = [os.sep.join(tid) for tid in track_ids]

    # top-key level metadata

    metadata_path = "labelled_metadata_public.csv"
    metadata_checksum = md5(os.path.join(dataset_data_path, metadata_path))

    index_metadata = {
        "metadata": {
            "spatiotemporal_metadata": (metadata_path, metadata_checksum),
        }
    }

    # top-key level tracks
    index_tracks = {}
    for i, tid in enumerate(tqdm(track_ids)):

        assert os.path.exists(os.path.join(audio_dir, audio_files[i]))
        assert os.path.exists(os.path.join(label_dir, label_files[i]))

        audio_checksum = md5(os.path.join(audio_dir, audio_files[i]))

        label_checksum = md5(os.path.join(label_dir, label_files[i]))

        index_tracks[tid[-1].replace(".flac", "")] = {
            "audio": (f"labelled/{audio_files[i]}", audio_checksum),
            "annotation": (f"labels_public/{label_files[i]}", label_checksum),
        }

    # top-key level version
    dataset_index = {"version": "1.0a"}

    # combine all in dataset index
    dataset_index.update(index_metadata)
    dataset_index.update({"clips": index_tracks})

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(dataset_index, fhandle, indent=2)


def main(args):
    make_dataset_index(args.dataset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make dataset index file.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to dataset data folder."
    )

    main(PARSER.parse_args())
