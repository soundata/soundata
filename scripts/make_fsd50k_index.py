import argparse
import hashlib
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/fsd50k_index.json"


def make_index(data_path):

    metadata_folder = "FSD50K.metadata"
    ground_truth_folder = "FSD50K.ground_truth"

    index = {
        "version": "1.0",
        "clips": {},
        "metadata": {
            # Groundtruth files
            "dev_ground_truth": [
                os.path.join(ground_truth_folder, "dev.csv"),
                md5(os.path.join(data_path, ground_truth_folder, "dev.csv")),
            ],
            "eval_ground_truth": [
                os.path.join(ground_truth_folder, "eval.csv"),
                md5(os.path.join(data_path, ground_truth_folder, "eval.csv")),
            ],
            # List of FSD50K sound classes
            "vocabulary": [
                os.path.join(ground_truth_folder, "vocabulary.csv"),
                md5(os.path.join(data_path, ground_truth_folder, "vocabulary.csv")),
            ],
            # Additional metadata
            "dev_clips_info": [
                os.path.join(metadata_folder, "dev_clips_info_FSD50K.json"),
                md5(
                    os.path.join(
                        data_path, metadata_folder, "dev_clips_info_FSD50K.json"
                    )
                ),
            ],
            "eval_clips_info": [
                os.path.join(metadata_folder, "eval_clips_info_FSD50K.json"),
                md5(
                    os.path.join(
                        data_path, metadata_folder, "eval_clips_info_FSD50K.json"
                    )
                ),
            ],
            # Relevant info about the labels
            "class_info": [
                os.path.join(metadata_folder, "class_info_FSD50K.json"),
                md5(os.path.join(data_path, metadata_folder, "class_info_FSD50K.json")),
            ],
            # PP/PNP ratings
            "pp_pnp_ratings": [
                os.path.join(metadata_folder, "pp_pnp_ratings_FSD50K.json"),
                md5(
                    os.path.join(
                        data_path, metadata_folder, "pp_pnp_ratings_FSD50K.json"
                    )
                ),
            ],
        },
    }

    # Development audio folder
    dev_audio_dir = "FSD50K.dev_audio"
    eval_audio_dir = "FSD50K.eval_audio"
    dev_clips = glob.glob(os.path.join(data_path, dev_audio_dir, "*.wav"))
    eval_clips = glob.glob(os.path.join(data_path, eval_audio_dir, "*.wav"))

    # Store development clips
    for clip in dev_clips:
        clip_id = os.path.basename(clip).replace(".wav", "")
        index["clips"][clip_id] = {
            "audio": [
                os.path.join(dev_audio_dir, os.path.basename(clip)),
                md5(clip),
            ]
        }

    # Store evaluation clips
    for clip in eval_clips:
        clip_id = os.path.basename(clip).replace(".wav", "")
        index["clips"][clip_id] = {
            "audio": [
                os.path.join(eval_audio_dir, os.path.basename(clip)),
                md5(clip),
            ]
        }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate FSD50K index file.")
    PARSER.add_argument("data_path", type=str, help="Path to FSD50K data folder.")

    main(PARSER.parse_args())
