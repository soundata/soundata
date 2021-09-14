import argparse
import hashlib
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/marco_index.json"


def make_index(data_path):

    rel_paths = {
        'impulse_response': [
            "3D-MARCo\ Impulse\ Responses/01_Speaker_+90deg_3m",
            "3D-MARCo\ Impulse\ Responses/02_Speaker_+75deg_4m",
            "3D-MARCo\ Impulse\ Responses/03_Speaker_+60deg_3m",
            "3D-MARCo\ Impulse\ Responses/04_Speaker_+45deg_4m",
            "3D-MARCo\ Impulse\ Responses/05_Speaker_+30deg_3m",
            "3D-MARCo\ Impulse\ Responses/06_Speaker_+15deg_4m",
            "3D-MARCo\ Impulse\ Responses/07_Speaker_0deg_3m",
            "3D-MARCo\ Impulse\ Responses/08_Speaker_-15deg_4m",
            "3D-MARCo\ Impulse\ Responses/09_Speaker_-30deg_3m",
            "3D-MARCo\ Impulse\ Responses/10_Speaker_-45deg_4m",
            "3D-MARCo\ Impulse\ Responses/11_Speaker_-60deg_3m",
            "3D-MARCo\ Impulse\ Responses/12_Speaker_-75deg_4m",
            "3D-MARCo\ Impulse\ Responses/13_Speaker_-90deg_3m",
        ],
        'acapella': ["Acapella"],
        'organ': ["Organ"],
        'piano_solo_1': ["Piano\ Solo\ 1"], 
        'piano_solo_2': ["Piano\ Solo\ 2"],
        'quartet': ["Quartet"],
        'single_sources': [
            "Single\ sources\ at\ different\ positions/01_0deg",
            "Single\ sources\ at\ different\ positions/02_-15deg",
            "Single\ sources\ at\ different\ positions/03_-30deg",
            "Single\ sources\ at\ different\ positions/04_-45deg",
            "Single\ sources\ at\ different\ positions/05_-60deg",
            "Single\ sources\ at\ different\ positions/06_-75deg",
            "Single\ sources\ at\ different\ positions/07_-90deg",
        ],
        'trio': ["Trio"],
    }

    index = {
        "version": "1.0.1",
        "clips": {},
        }
    }

    for source_type, paths in rel_paths.items():

        for path in paths:

            audio_path = os.path.join(data_path, path)
            wavfiles = glob.glob(os.path.join(audio_path, "*.wav")

            for wf in wavfiles:
                
                clip_id = "{}/{}".format(
                    source_type,
                    os.path.basename(wf).replace(".wav","")
                )

                index["clips"][clip_id] = {
                    "audio": [
                        os.path.join(path, os.path.basename(wf)),
                        md5(wf)
                    ]
                }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate 3D-MARCo index file.")
    PARSER.add_argument("data_path", type=str, help="Path to 3D-MARCo data folder.")

    main(PARSER.parse_args())
