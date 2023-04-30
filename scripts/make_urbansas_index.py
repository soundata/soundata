import argparse
import hashlib
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/urbansas_index.json"


def make_index(data_path):

    audio_ann_rel_path = os.path.join("./", "audio_annotations.csv")
    video_ann_rel_path = os.path.join("./", "video_annotations.csv")

    index = {
        "version": "1.0",
        "clips": {},
        "metadata": {
            "audio_annotations.csv": [
                audio_ann_rel_path,
                md5(os.path.join(data_path, audio_ann_rel_path)),
            ],
            "video_annotations.csv": [
                video_ann_rel_path,
                md5(os.path.join(data_path, video_ann_rel_path)),
            ]
        },
    }

    audio_dir = os.path.join(data_path, "audio")
    video_dir = os.path.join(data_path, "video/video_2fps")

    mp4files = glob.glob(os.path.join(video_dir, "*.mp4"))

    for video_file in mp4files:
        clip_id = os.path.basename(video_file).replace(".mp4", "")
        audio_file = os.path.basename(video_file).replace(".mp4", ".wav")
        index["clips"][clip_id] = {
            "audio": [
                os.path.join("audio", audio_file),
                md5(os.path.join(audio_dir, audio_file)),
            ],
            "video": [
                os.path.join("video/video_2fps", os.path.basename(video_file)),
                md5(video_file),
            ]
        }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate urbansas index file.")
    PARSER.add_argument("data_path", type=str, help="Path to urbansas data folder.")

    main(PARSER.parse_args())
