import argparse
from natsort import natsorted, ns
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/urbansed_index.json"


def make_index(data_path):

    index = {
        "version": "2.0.0",
        "clips": {}
    }

    for split in ["train", "validate", "test"]:

        audio_split_dir = os.path.join(data_path, "audio", split)
        annotations_split_dir = os.path.join(data_path, "annotations", split)

        wavfiles = natsorted(glob.glob(os.path.join(audio_split_dir, "*.wav")), alg=ns.IGNORECASE)

        for wf in wavfiles:

            jamsfile = os.path.join(annotations_split_dir, os.path.basename(wf).replace(".wav", ".jams"))
            txtfile = os.path.join(annotations_split_dir, os.path.basename(wf).replace(".wav", ".txt"))

            assert os.path.isfile(jamsfile)
            assert os.path.isfile(txtfile)

            clip_id = os.path.basename(wf).replace(".wav", "")
            index["clips"][clip_id] = {
                "audio": [
                    os.path.join("audio", split, os.path.basename(wf)),
                    md5(wf),
                ],
                "jams": [os.path.join("annotations", split, os.path.basename(jamsfile)), md5(jamsfile)],
                "txt": [os.path.join("annotations", split, os.path.basename(txtfile)), md5(txtfile)]
            }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate URBAN-SED index file.")
    PARSER.add_argument("data_path", type=str, help="Path to urbansed data folder.")

    main(PARSER.parse_args())
