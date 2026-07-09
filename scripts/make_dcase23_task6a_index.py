import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/dcase23_task6a_index_1.0.json"

def make_index(data_path):

    rel_paths = {
        'test': 'test',
    }

    metadata_files = {
        'test': "clotho_metadata_test.csv",
    }

    index = {"version": "1.0", "clips": {}, "metadata": {}}

    for subset, relative_path in rel_paths.items():
        audio_path = os.path.join(data_path, relative_path)
        wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))

        for wf in wavfiles:
            clip_id = "{}".format(os.path.basename(wf).replace(".wav", ""))
            index["clips"][clip_id] = {
                "audio": [os.path.join(relative_path, os.path.basename(wf)), md5(wf)],
            }

        metadata_doc_id = "{}".format(os.path.basename(metadata_files[subset]).replace(".csv", ""))
        metadata_path = os.path.join(data_path, metadata_files[subset])
        index["metadata"][metadata_doc_id] = [
            os.path.join(metadata_files[subset]),
            md5(metadata_path)
        ]

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)

    print("index created in", INDEX_PATH)
    
def main(args):
    make_index(args.data_path)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate DCASE'23 Task 6A dataset index file.")
    PARSER.add_argument("data_path", type=str, help="Path to DCASE'23 Task 6A dataset folder.")
    main(PARSER.parse_args())
