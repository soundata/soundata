import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/tau2022uas_mobile_index.json"


def make_index(data_path):

    rel_paths = {
        "development": "TAU-urban-acoustic-scenes-2022-mobile-development",
        "evaluation": "TAU-urban-acoustic-scenes-2023-mobile-evaluation",
    }

    metadata_rel_path = os.path.join(rel_paths["development"], "meta.csv")

    setup_paths = {}
    for dataset_type in rel_paths.keys():
        setup_paths[dataset_type] = os.path.join(
            rel_paths[dataset_type], "evaluation_setup"
        )

    index = {
        "version": "3.0",
        "clips": {},
        "metadata": {
            "meta.csv": [
                metadata_rel_path,
                md5(os.path.join(data_path, metadata_rel_path)),
            ],
            "fold1_evaluate.csv": [
                os.path.join(setup_paths["development"], "fold1_evaluate.csv"),
                md5(
                    os.path.join(
                        data_path, setup_paths["development"], "fold1_evaluate.csv"
                    )
                ),
            ],
            "fold1_test.csv": [
                os.path.join(setup_paths["development"], "fold1_test.csv"),
                md5(
                    os.path.join(
                        data_path, setup_paths["development"], "fold1_test.csv"
                    )
                ),
            ],
            "fold1_train.csv": [
                os.path.join(setup_paths["development"], "fold1_train.csv"),
                md5(
                    os.path.join(
                        data_path, setup_paths["development"], "fold1_train.csv"
                    )
                ),
            ],
            "evaluation/fold1_test.csv": [
                os.path.join(setup_paths["evaluation"], "fold1_test.csv"),
                md5(
                    os.path.join(data_path, setup_paths["evaluation"], "fold1_test.csv")
                ),
            ],
        },
    }

    for relative_path in rel_paths.values():

        audio_path = os.path.join(data_path, relative_path, "audio")

        wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))

        for wf in wavfiles:

            clip_id = os.path.basename(wf).replace(".wav", "")

            index["clips"][clip_id] = {
                "audio": [
                    os.path.join(relative_path, "audio", os.path.basename(wf)),
                    md5(wf),
                ]
            }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Generate tau2022uas_mobile index file."
    )
    PARSER.add_argument(
        "data_path", type=str, help="Path to tau2022uas_mobile data folder."
    )

    main(PARSER.parse_args())
