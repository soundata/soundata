import argparse
import json
import os
from soundata.validate import md5
import glob
import csv

DATASET_INDEX_PATH = "../soundata/datasets/indexes/tau_nigens_sse_2021_index.json"


def _index_wav(index, data_path, formt, subset, split):

    if subset == "development":
        audio_path = os.path.join(
                data_path,
                formt+"_"+subset,
                subset+"_"+split,
        )
    elif subset == "evaluation":
        audio_path = os.path.join(
                data_path,
                formt+"_"+subset,
        )

    wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))
    wavfiles.sort()

    for wf in wavfiles:

        if subset == "development":
            clip_id = '{}/{}/{}'.format(
                    formt+"_"+subset,
                    subset+"_"+split,
                    path.basename(wf).replace(".wav","")
            )

            index["clips"][clip_id] = {
                "audio": [
                    os.path.join(formt+"_"+subset, subset+"_"+split, os.path.basename(wf)),
                    md5(wf),
                ],
            }
        elif subset == "evaluation":
            clip_id = '{}/{}'.format(
                    formt+"_"+subset,
                    path.basename(wf).replace(".wav","")
            )

            index["clips"][clip_id] = {
                "audio": [
                    os.path.join(formt+"_"+subset, os.path.basename(wf)),
                    md5(wf),
                ],
            }

    return index

def _index_event(index, data_path, formt, annotation_subset, subset, split):

    if subset == "development":
        annotation_path = os.path.join(
                data_path,
                annotation_subset,
                subset+"_"+split,
        )
    elif subset == "evaluation":
        annotation_path = os.path.join(
                data_path,
                annotation_subset,
        )

    annotationfiles = glob.glob(os.path.join(annotation_path, "*.csv"))
    annotationfiles.sort()

    for af in annotationfiles:

        if subset == "development":
            clip_id = '{}/{}/{}'.format(
                    formt+"_"+subset,
                    subset+"_"+split,
                    os.path.basename(af).replace(".csv", "")
            )

            index["clips"][clip_id]["events"] = [
                    os.path.join(annotation_subset, subset+"_"+split, os.path.basename(af)),
                    md5(af),
                ]
        elif subset == "evaluation":
            clip_id = '{}/{}'.format(
                    formt+"_"+subset,
                    os.path.basename(af).replace(".csv", "")
            )

            index["clips"][clip_id]["events"] = [
                    os.path.join(annotation_subset, os.path.basename(af)),
                    md5(af),
                ]

    return index

def make_index(data_path):

    subsets = [ 
        "development",
        "evaluation",
    ]

    split = [
        "train",
        "val",
        "test"
    ]

    annotations = {
        "development": "metadata_dev",
        "evaluation": "metadata_eval",
    }

    formats = [
        "foa",
        "mic",
    ]

    index = {
        "version": "1.2.0",
        "clips": {},
        "metadata": {},
    }

    # add recordings and annotations to index
    for subset in subsets:

        for formt in formats:

            for s in split:

                if subset == "development":

                    index = _index_wav(index, data_path, formt, "dev", s)

                    index = _index_event(index, data_path, formt, annotations[subset], "dev", s)

                elif subset == "evaluation":

                    index = _index_wav(index, data_path, formt, "eval", s)
                    index = _index_event(index, data_path, formt, annotations[subset], "eval", s)


    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate TAU-NIGENS Spatial Sound Events 2021 index file.")
    PARSER.add_argument("data_path", type=str, help="Path to TAU-NIGENS Spatial Sound Events 2021 data directory.")

    main(PARSER.parse_args())
