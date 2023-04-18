import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/dcase_bioacoustic_index.json"


dev_audio_folder = "Development_Set"
eval_audio_folder = "Evaluation_set_5shots"

splits = {"train":[dev_audio_folder,'Training_Set',['BV','HT','JD','MT','WMV']],
        "val":[dev_audio_folder,'Validation_Set',['HB','ME','PB']],
        "eval":[eval_audio_folder,'',['CHE','CT','DC','MGE','MS','QU']]}

def make_index(data_path):

    index = {
        "version": "3.0.0",
        "clips": {},
        "metadata": {
            # Groundtruth files
            "dev_classes": [
               "DCASE2022_task5_Training_set_classes.csv",
                md5(os.path.join(data_path, "DCASE2022_task5_Training_set_classes.csv")),
            ],
            "val_classes": [
                "DCASE2022_task5_Validation_set_classes.csv",
                md5(os.path.join(data_path, "DCASE2022_task5_Validation_set_classes.csv")),
            ]
        }
    }

    for split,d in splits.items():
        data_dir = os.path.join(data_path, d[0],d[1])
        for c in d[2]:
            wavfiles = glob.glob(os.path.join(data_dir, c, "*.wav"))
            for wf in wavfiles:

                csvfile = os.path.join(data_dir, c, os.path.basename(wf).replace(".wav", ".csv"))

                assert os.path.isfile(csvfile)

                clip_id = os.path.basename(wf).replace(".wav", "")
                index["clips"][clip_id] = {
                    "audio": [
                        os.path.join(d[0],d[1], c, os.path.basename(wf)),
                        md5(wf),
                    ],
                    "csv": [os.path.join(d[0],d[1], c, os.path.basename(csvfile)), md5(csvfile)]
                }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate DCASE-BIOACOUSTIC index file.")
    PARSER.add_argument("data_path", type=str, help="Path to dcase bioacoustic data folder.")

    main(PARSER.parse_args())
