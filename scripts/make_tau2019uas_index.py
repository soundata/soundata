import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/tau2019uas_index.json"


def make_index(data_path):

    rel_paths = {
        'development': "TAU-urban-acoustic-scenes-2019-development",
        'evaluation': "TAU-urban-acoustic-scenes-2019-evaluation",
        'leaderboard': "TAU-urban-acoustic-scenes-2019-leaderboard"
    }

    metadata_rel_path = os.path.join(rel_paths["development"], "meta.csv")

    setup_paths = {}
    for dataset_type in rel_paths.keys():
        setup_paths[dataset_type] = os.path.join(
            rel_paths[dataset_type], "evaluation_setup"
        )

    index = {
        "version": "1.0",
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
                        data_path,
                        setup_paths["development"],
                        "fold1_evaluate.csv"
                    )
                ),
            ],
            "fold1_test.csv": [
                os.path.join(setup_paths["development"], "fold1_test.csv"),
                md5(
                    os.path.join(
                        data_path,
                        setup_paths["development"],
                        "fold1_test.csv"
                    )
                ),
            ],
            "fold1_train.csv": [
                os.path.join(setup_paths["development"], "fold1_train.csv"),
                md5(
                    os.path.join(
                        data_path,
                        setup_paths["development"],
                        "fold1_train.csv"
                    )
                ),
            ],
            "evaluation/test.csv": [
                os.path.join(setup_paths["evaluation"], "test.csv"),
                md5(
                    os.path.join(
                        data_path,
                        setup_paths["evaluation"],
                        "test.csv"
                    )
                ),
            ],
            "leaderboard/test.csv": [
                os.path.join(setup_paths["leaderboard"], "test.csv"),
                md5(
                    os.path.join(
                        data_path,
                        setup_paths["leaderboard"],
                        "test.csv"
                    )
                ),
            ],
        }
    }

    for subset, relative_path in rel_paths.items():

        audio_path = os.path.join(data_path, relative_path, "audio")

        wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))

        for wf in wavfiles:

            clip_id = "{}/{}".format(
                subset,
                os.path.basename(wf).replace(".wav", "")
            )

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
        description="Generate tau2019uas index file."
    )
    PARSER.add_argument(
        "data_path", type=str,
        help="Path to tau2019uas data folder."
    )

    main(PARSER.parse_args())
