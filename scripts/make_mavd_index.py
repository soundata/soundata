import argparse
from natsort import natsorted, ns
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/mavd_index.json"


def make_index(data_path):

    index = {
        "version": "0.1.0",
        "clips": {}
    }

    splits = ["train", "validate", "test"]
    expected_sizes = [24, 7, 16]
    
    for split, es in zip(splits, expected_sizes):

        audio_split_dir = os.path.join(data_path, "audio_" + split)
        annotations_split_dir = os.path.join(data_path, "annotations_" + split)

        audiofiles = natsorted(glob.glob(os.path.join(audio_split_dir, "*.flac")), alg=ns.IGNORECASE)

        assert len(audiofiles) == es

        for af in audiofiles:

            txtfile = os.path.join(annotations_split_dir, os.path.basename(af).replace(".flac", ".txt"))

            assert os.path.isfile(txtfile)

            clip_id = os.path.basename(af).replace(".flac", "")
            index["clips"][clip_id] = {
                "audio": [
                    os.path.join("audio_" + split, os.path.basename(af)),
                    md5(af),
                ],
                "txt": [os.path.join("annotations_" + split, os.path.basename(txtfile)), md5(txtfile)]
            }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate MAVD index file.")
    PARSER.add_argument("data_path", type=str, help="Path to MAVD data folder.")

    main(PARSER.parse_args())
