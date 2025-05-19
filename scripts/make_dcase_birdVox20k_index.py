import argparse
import json
import os
import glob
from soundata.validate import md5
from tqdm import tqdm
import logging

INDEX_PATH = "../soundata/datasets/indexes/dcase_birdVox20k_index.json"

def make_index(data_path):
    # Create BirdVox20k index
    index = {
        "version": "1.0",
        "clips": {},
        "metadata": {
            "BirdVoxDCASE20k_csvpublic" : [
                "BirdVoxDCASE20k_csvpublic.csv",
                md5(os.path.join(data_path, "BirdVoxDCASE20k_csvpublic.csv")),
            ]
        }
    }

    # audio folder
    clips = glob.glob(os.path.join(data_path, "*.wav"))

    # store clips with loader
    for clip in tqdm(clips):
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
    if os.path.exists(args.data_path):
        make_index(args.data_path)
    else:
        logging.error("Invalid data_path: %s does not exist.", args.data_path)
    

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate BirdVox20k index file.")
    PARSER.add_argument("data_path", type=str, help="Path to BirdVox20k data folder.")
    main(PARSER.parse_args())
