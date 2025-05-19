import argparse
import hashlib
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/eigenscape_raw_index.json"


def make_index(data_path):

    index = {
        "version": "raw",
        "clips": {},
        "metadata": {
            "Metadata-EigenScape" : [
                "Metadata-EigenScape.csv",
                md5(os.path.join(data_path, "Metadata-EigenScape.csv")),
            ]
        }
    }

    # audio folder
    clips = glob.glob(os.path.join(data_path, "*.wav"))

    # Store clips
    for clip in clips:
        clip_id = os.path.basename(clip).replace(".wav", "")
        index["clips"][clip_id] = {
            "audio": [
                os.path.join(os.path.basename(clip)),
                md5(clip),
            ]
        }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate EigenScape Raw index file.")
    PARSER.add_argument("data_path", type=str, help="Path to EigenScape Raw data folder.")

    main(PARSER.parse_args())
