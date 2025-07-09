import argparse
import glob
import json
import os
from soundata.validate import md5

INDEX_PATH = "soundata/datasets/indexes/birdsongs_index.json"


def make_index(data_path):

    audio_folder = "wavfiles"
    # Create Warblrb10k index
    index = {
        "version": "1.0",
        "clips": {},
        "metadata": {
            "BirdSongs" : [
                "bird_songs_metadata.csv",
                md5(os.path.join(data_path, "bird_songs_metadata.csv")),
            ]
        }
    }       
    
    clips = glob.glob(os.path.join(data_path, audio_folder, "*.wav"))



    # Store train clips
    for clip in clips:
        clip_id = os.path.basename(clip).replace(".wav", "")
        index["clips"][clip_id] = {
            "audio": [
                os.path.join(audio_folder, os.path.basename(clip)),
                md5(clip),
            ]
        }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate BirdSongs index file.")
    PARSER.add_argument("data_path", type=str, help="Path to BirdSongs data folder.")

    main(PARSER.parse_args())
