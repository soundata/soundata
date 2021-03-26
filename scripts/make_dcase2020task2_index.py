import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/dcase2020task2_index.json"


def make_index(data_path):

    subsets = ["development", "additional_training", "evaluation"]

    splits = [
        "development.train",
        "development.test",
        "additional_training.train",
        "evaluation.test",
    ]

    machine_types = ["fan", "pump", "slider", "ToyCar", "ToyConveyor", "valve"]

    index = {
        "version": "1.0",
        "clips": {},
    }

    for split in splits:

        subset, fold = split.split(".")

        for mt in machine_types:

            audio_path = os.path.join(data_path, subset, mt, fold)

            wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))

            for wf in wavfiles:

                clip_id = "{}.{}/{}/{}".format(
                    subset, fold, mt, os.path.basename(wf).replace(".wav", "")
                )

                index["clips"][clip_id] = {
                    "audio": [
                        os.path.join(subset, mt, fold, os.path.basename(wf)),
                        md5(wf),
                    ]
                }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate dcase2020task2 index file.")
    PARSER.add_argument(
        "data_path", type=str, help="Path to dcase2020task2 data folder."
    )

    main(PARSER.parse_args())
