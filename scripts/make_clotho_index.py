import argparse
import glob
import json
import os
from soundata.validate import md5

DATASET_INDEX_PATH = "../soundata/datasets/indexes/clotho_index_2.1.json"


def make_dataset_index(dataset_data_path):
    splits = ["development", "validation", "evaluation"]
    index_clips = {}

    for split in splits:
        audio_dir = os.path.join(dataset_data_path, f"clotho_audio_{split}")
        captions_file = os.path.join(dataset_data_path, f"clotho_captions_{split}.csv")
        metadata_file = os.path.join(dataset_data_path, f"clotho_metadata_{split}.csv")

        # checksum
        captions_checksum = md5(captions_file)
        metadata_checksum = md5(metadata_file)

        audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))

        for audio_path in audio_files:
            clip_id = os.path.splitext(os.path.basename(audio_path))[0]

            audio_rel_path = os.path.join(f"clotho_audio_{split}", os.path.basename(audio_path))

            index_clips[clip_id] = {
                "audio": [audio_rel_path, md5(audio_path)],
                "captions": [f"clotho_captions_{split}.csv", captions_checksum],
                "metadata": [f"clotho_metadata_{split}.csv", metadata_checksum],
            }

    dataset_index = {
        "version": "2.1",
        "clips": index_clips,
    }

    os.makedirs(os.path.dirname(DATASET_INDEX_PATH), exist_ok=True)
    with open(DATASET_INDEX_PATH, "w") as f:
        json.dump(dataset_index, f, indent=2)


def main(args):
    make_dataset_index(args.dataset_data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make Clotho dataset index.")
    PARSER.add_argument(
        "dataset_data_path", type=str, help="Path to the root of the Clotho dataset"
    )
    main(PARSER.parse_args())