import argparse
import json
import os
from soundata.validate import md5
import glob
import csv

DATASET_INDEX_PATH = "../soundata/datasets/indexes/starss2022_index.json"


def _index_wav(index, data_path, formt, subset, split, site):

    audio_path = os.path.join(
            data_path,
            "{}_{}".format(formt,subset),
            "{}-{}-{}".format(subset,split,site),
    )

    wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))
    wavfiles.sort()

    for wf in wavfiles:

        clip_id = '{}_{}/{}-{}-{}/{}'.format(
                formt,subset,
                subset,split,site,
                os.path.basename(wf).replace(".wav","")
        )

        index["clips"][clip_id] = {
            "audio": [
                os.path.join("{}_{}".format(formt,subset), "{}-{}-{}".format(subset,split,site), os.path.basename(wf)),
                md5(wf),
            ],
        }

    return index

def _index_event(index, data_path, formt, annotation_subset, subset, split, site):

    if subset == "dev":
        annotation_path = os.path.join(
                data_path,
                annotation_subset,
                "{}-{}-{}".format(subset,split,site)
        )
    elif subset == "eval":
        annotation_path = os.path.join(
                data_path,
                annotation_subset,
        )

    annotationfiles = glob.glob(os.path.join(annotation_path, "*.csv"))
    annotationfiles.sort()

    for af in annotationfiles:

        clip_id = '{}_{}/{}-{}-{}/{}'.format(
                formt,subset,
                subset,split,site,
                os.path.basename(af).replace(".csv", "")
        )
        if subset == "dev":

            index["clips"][clip_id]["events"] = [
                    os.path.join(annotation_subset, "{}-{}-{}".format(subset,split,site), os.path.basename(af)),
                    md5(af),
                ]
        elif subset == "eval":

            index["clips"][clip_id]["events"] = [
                    os.path.join(annotation_subset, os.path.basename(af)),
                    md5(af),
                ]

    return index

def make_index(data_path):

    subsets = [ 
        "development",
        # "evaluation", not yet available. Come back later
    ]

    split = [
        "train",
        "test"
    ]

    rec_sites = [
        "sony",
        "tau",
    ]

    annotations = {
        "development": "metadata_dev",
        # "evaluation": "metadata_eval", not yet available. Come back later
    }

    formats = [
        "foa",
        "mic",
    ]

    index = {
        "version": "1.0.0",
        "clips": {},
        "metadata": {},
    }

    # add recordings and annotations to index
    for subset in subsets:

        for formt in formats:

            if subset == "development":

                for s in split:

                    for r in rec_sites:

                        index = _index_wav(index, data_path, formt, "dev", s, r)

                        index = _index_event(index, data_path, formt, annotations[subset], "dev", s, r)

            elif subset == "evaluation":

                index = _index_wav(index, data_path, formt, "eval", "test")
                index = _index_event(index, data_path, formt, annotations[subset], "eval", "test")


    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate STARSS 2022 index file.")
    PARSER.add_argument("data_path", type=str, help="Path to STARSS 2022 data directory.")

    main(PARSER.parse_args())
