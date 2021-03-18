import argparse
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/tut_sound_events_2017_index.json"


def make_index(data_path):

    rel_paths = {
        'development': "TUT-sound-events-2017-development",
        'evaluation': "TUT-sound-events-2017-evaluation"
    }

    index = {
        "version": "2.0",
        "clips": {},
    }

    for dataset_type, relative_path in rel_paths.items():

        audio_path = os.path.join(data_path, relative_path, "audio/street")

        wavfiles = glob.glob(os.path.join(audio_path, "*.wav"))

        for wf in wavfiles:

            clip_id = os.path.basename(wf).replace(".wav", "")

            ann_file = wf.replace("audio", "meta").replace(".wav", ".ann")

            assert os.path.isfile(ann_file)

            if dataset_type == 'development':
                non_verified_ann_file = ann_file.replace(
                    "meta", "non_verified/meta")
                assert os.path.isfile(non_verified_ann_file)
            else:
                non_verified_ann_file = None

            index["clips"][clip_id] = {
                "audio": [
                    os.path.join(
                        relative_path, "audio/street",
                        os.path.basename(wf)),
                    md5(wf),
                ],
                "annotations": [
                    os.path.join(
                        relative_path, "meta/street",
                        os.path.basename(ann_file)),
                    md5(ann_file),
                ],
                "non_verified_annotations": [
                    os.path.join(
                        relative_path, "non_verified/meta/street",
                        os.path.basename(ann_file)),
                    md5(non_verified_ann_file),
                ] if non_verified_ann_file is not None else [None, None]
            }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Generate tut_sound_events_2017 index file."
    )
    PARSER.add_argument(
        "data_path", type=str,
        help="Path to tut_sound_events_2017 data folder."
    )

    main(PARSER.parse_args())
