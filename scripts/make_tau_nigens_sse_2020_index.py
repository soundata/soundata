import argparse
import json
import os
from soundata.validate import md5
import glob
import csv

DATASET_INDEX_PATH = "../soundata/datasets/indexes/tau_nigens_sse_2020_index.json"


def _get_long_eval_filenames(data_path):
    short2long_eval = {}
    with open(os.path.join(data_path, "metadata_eval_info.txt")) as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader, None) # skip header
        for row in csv_reader:
            mix_name = "mix"+str(row[0]).zfill(3)
            short2long_eval[mix_name] = "fold"+str(row[1])+"_"+mix_name+"_ov"+str(row[2])
    return short2long_eval


def _index_wav(index, data_path, formt, subset):

    eval_long_filenames = _get_long_eval_filenames(data_path)

    audio_path = os.path.join(
            data_path,
            formt+"_"+subset,
    )

    wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))
    wavfiles.sort()

    for wf in wavfiles:

        clip_id = '{}/{}'.format(
                formt+"_"+subset,
                eval_long_filenames[os.path.basename(wf).replace(".wav","")] if subset=="eval" else os.path.basename(wf).replace(".wav","")
        )

        index["clips"][clip_id] = {
            "audio": [
                os.path.join(formt+"_"+subset, os.path.basename(wf)),
                md5(wf),
            ],
        }

    return index

def _index_event(index, data_path, formt, annotation_subset, subset):

    eval_long_filenames = _get_long_eval_filenames(data_path)

    annotation_path = os.path.join(
            data_path,
            annotation_subset,
    )

    annotationfiles = glob.glob(os.path.join(annotation_path, "*.csv"))
    annotationfiles.sort()

    for af in annotationfiles:

        clip_id = '{}/{}'.format(
                formt+"_"+subset,
                eval_long_filenames[os.path.basename(af).replace(".csv","")] if subset=="eval" else os.path.basename(af).replace(".csv", "")
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

            if subset == "development":

                index = _index_wav(index, data_path, formt, "dev")

                index = _index_event(index, data_path, formt, annotations[subset], "dev")

            elif subset == "evaluation":

                index = _index_wav(index, data_path, formt, "eval")
                index = _index_event(index, data_path, formt, annotations[subset], "eval")


    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate TAU-NIGENS Spatial Sound Events 2020 index file.")
    PARSER.add_argument("data_path", type=str, help="Path to TAU-NIGENS Spatial Sound Events 2020 data directory.")

    main(PARSER.parse_args())
