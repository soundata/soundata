import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/dcase23_task4b_index.json"

def make_index(data_path):
    categories = [
        'cafe_restaurant', 'city_center', 'grocery_store', 'metro_station', 'residential_area']
    
    index = {
        "version": "1.0",
        "clips": {},
    }

    for category in categories:
        audio_path = os.path.join(data_path, 'development_audio', category)
        ann_path = os.path.join(data_path, 'development_annotation', 'soft_labels_' + category)

        wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))
        txtfiles = glob.glob(os.path.join(ann_path, "*.txt"))

        for wf, tf in zip(wavfiles, txtfiles):
            clip_id = os.path.basename(wf).replace(".wav", "")

            assert os.path.isfile(tf)

            index["clips"][clip_id] = {
                "audio": [
                    os.path.join('development_audio', category, os.path.basename(wf)),
                    md5(wf),
                ],
                "annotations": [
                    os.path.join('development_annotation', 'soft_labels_' + category, os.path.basename(tf)),
                    md5(tf),
                ]
            }

    # Indexing evaluation files
    for category in categories:
        audio_path = os.path.join(data_path, 'Evaluation_audio', category)
        wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))

        for wf in wavfiles:
            clip_id = os.path.basename(wf).replace(".wav", "")
            index["clips"][clip_id] = {
                "audio": [
                    os.path.join('Evaluation_audio', category, os.path.basename(wf)),
                    md5(wf),
                ],
                "annotations": [
                    None, None
                ]
            }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate dcase23_task4b index file.")
    PARSER.add_argument("data_path", type=str, help="Path to dcase23_task4b data folder.")
    args = PARSER.parse_args()
    make_index(args.data_path)