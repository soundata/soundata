import argparse
import glob
import json
import os
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/warblrb10k_index.json"


def make_index(data_path):

    audio_train_folder = "wav"
    audio_test_folder = "wabrlrb10k_test_wav"

    # Create Warblrb10k index
    index = {
        "version": "1.0",
        "clips": {},
        "metadata": {
            "Warblrb10k" : [
                "warblrb10k_public_metadata.csv",
                md5(os.path.join(data_path, "warblrb10k_public_metadata.csv")),
            ]
        }
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
    PARSER = argparse.ArgumentParser(description="Generate Warblrb10k index file.")
    PARSER.add_argument("data_path", type=str, help="Path to Warblrb10k data folder.")

    main(PARSER.parse_args())
