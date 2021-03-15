import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/sonyc_ust_index.json"

def make_index(data_path):

    metadata_rel_path = os.path.join("annotations.csv")

    taxonomy_rel_path = os.path.join("dcase-ust-taxonomy.yaml")

    index = {
        "version": "2.3.0",
        "clips": {},
        "metadata": {
            "annotations.csv": [
                metadata_rel_path,
                md5(os.path.join(data_path, metadata_rel_path)),
            ],
            "dcase-ust-taxonomy.yaml": [
                taxonomy_rel_path,
                md5(os.path.join(data_path, taxonomy_rel_path)),
            ]
        },
    }

    audio_dir = os.path.join(data_path, "audio")
    wavfiles = glob.glob(os.path.join(audio_dir, "*.wav"))

    for wf in wavfiles:
        clip_id = os.path.basename(wf).replace(".wav", "")
        index["clips"][clip_id] = {
            "audio": [
                os.path.join("audio", os.path.basename(wf)),
                md5(wf),
            ]
        }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate sonyc_ust index file.")
    PARSER.add_argument("data_path", type=str, help="Path to sonyc_ust data folder.")

    main(PARSER.parse_args())
