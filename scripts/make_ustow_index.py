import argparse
import json
import os

import numpy as np
from tqdm import tqdm
from soundata.validate import md5

MAX_INDEX = 133  # as of 18 May 2022
EXCLUDED_TRACKS = [
    21,
    77,
    86,
    93,
    100,
    102,
]  # these tracks are not listed on the website

DATASET_INDEX_PATH = "../soundata/datasets/indexes/usotw_index.json"

def make_dataset_index(dataset_data_path):
    track_ids = np.delete(np.arange(1, MAX_INDEX + 1), EXCLUDED_TRACKS)

    # top-key level tracks
    index_tracks = {}
    for track_id in tqdm(track_ids):

        foa_filename = "audio/ambisonics/R{:04d}_segment_ambisonics.wav".format(
            track_id
        )
        bin_filename = "audio/binaural/R{:04d}_segment_binaural.wav".format(track_id)

        foa_checksum = md5(os.path.join(dataset_data_path, foa_filename))

        bin_checksum = md5(
            os.path.join(
                dataset_data_path,
                bin_filename,
            )
        )

        video_filename = "video/spherical/R{:04d}_segment_ambisonics_headphones_highres.360.mono.mov".format(
            track_id
        )
        video_checksum = md5(
            os.path.join(
                dataset_data_path,
                video_filename,
            )
        )

        index_tracks[f"R{track_id:04d}"] = {
            "audio/binaural": (bin_filename, bin_checksum),
            "audio/ambisonics": (foa_filename, foa_checksum),
        }

        index_tracks[f"R{track_id:04d}"]["video"] = (
            video_filename,
            video_checksum,
        )

    # top-key level version
    dataset_index = {"version": "20220517"}

    dataset_index.update({"clips": index_tracks})
    
    dataset_index["metadata"] = {
        "spl": ("SotW_LAeq_binaural_average_LR.xlsx", "d001b32ad3ed7c7954f59784690e0875")
    }

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
