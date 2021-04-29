import argparse
import glob
import json
import os
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/fsdnoisy18k_index.json"


def make_index(data_path):

    metadata_folder = "FSDnoisy18k.meta"
    audio_train_folder = "FSDnoisy18k.audio_train"
    audio_test_folder = "FSDnoisy18k.audio_test"

    index = {
        "version": "1.0",
        "clips": {},
        "metadata": {
            # Groundtruth files
            "train": [
                os.path.join(metadata_folder, "train.csv"),
                md5(os.path.join(data_path, metadata_folder, "train.csv")),
            ],
            "test": [
                os.path.join(metadata_folder, "eval.csv"),
                md5(os.path.join(data_path, metadata_folder, "test.csv")),
            ],
        },
    }
    
    train_clips = glob.glob(os.path.join(data_path, audio_train_folder, "*.wav"))
    test_clips = glob.glob(os.path.join(data_path, audio_test_folder, "*.wav"))

    # Store train clips
    for clip in train_clips:
        clip_id = os.path.basename(clip).replace(".wav", "")
        index["clips"][clip_id] = {
            "audio": [
                os.path.join(audio_train_folder, os.path.basename(clip)),
                md5(clip),
            ]
        }

    # Store test clips
    for clip in test_clips:
        clip_id = os.path.basename(clip).replace(".wav", "")
        index["clips"][clip_id] = {
            "audio": [
                os.path.join(audio_test_folder, os.path.basename(clip)),
                md5(clip),
            ]
        }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate FSDnoisy18K index file.")
    PARSER.add_argument("data_path", type=str, help="Path to FSDnoisy18K data folder.")

    main(PARSER.parse_args())
