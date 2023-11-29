import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/dcase23_task6b_index.json"

def make_index(data_path):

    rel_paths = {
        'development': "development",
        'evaluation': "evaluation",
        'validation': "validation",
        'test': 'test',
    }

    metadata_files = {
        'development': "clotho_metadata_development.csv",
        'evaluation': "clotho_metadata_evaluation.csv",
        'validation': "clotho_metadata_validation.csv",
        'test': "retrieval_audio_metadata.csv",
    }

    captions_files = {
        'development': "clotho_captions_development.csv",
        'evaluation': "clotho_captions_evaluation.csv",
        'validation': "clotho_captions_validation.csv",
        'test': "retrieval_captions.csv",
    }

    index = {"version": "1.0", "clips": {}, "metadata": {}, "captions": {}}

    for subset, relative_path in rel_paths.items():
        audio_path = os.path.join(data_path, relative_path)
        wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))

        for wf in wavfiles:
            clip_id = "{}/{}".format(subset, os.path.basename(wf).replace(".wav", ""))
            index["clips"][clip_id] = {
                "audio": [os.path.join(relative_path, os.path.basename(wf)), md5(wf)],
            }

        metadata_doc_id = "{}".format(os.path.basename(metadata_files[subset]).replace(".csv", ""))
        metadata_path = os.path.join(data_path, metadata_files[subset])
        index["metadata"][metadata_doc_id] = [
            os.path.join(metadata_files[subset]),
            md5(metadata_path)
        ]

        caption_doc_id = "{}".format(os.path.basename(captions_files[subset]).replace(".csv", ""))
        captions_path = os.path.join(data_path, captions_files[subset])
        index["captions"][caption_doc_id] = [
            os.path.join(captions_files[subset]),
            md5(captions_path)
        ]

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)

def main(args):
    make_index(args.data_path)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate DCASE'23 Task 6B dataset index file.")
    PARSER.add_argument("data_path", type=str, help="Path to DCASE'23 Task 6B dataset folder.")
    main(PARSER.parse_args())
