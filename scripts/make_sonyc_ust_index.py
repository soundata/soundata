import argparse
import hashlib
import json
import os
import glob
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/sonyc_ust_index.json"


def make_index(data_path):
    index = {
        "version": "1.0",
        "clips": {}
    },


    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)


def main(args):
    make_index(args.data_path)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate SONYC-UST index file.")
    PARSER.add_argument("data_path", type=str, help="Path to SONYC-UST data folder.")

    main(PARSER.parse_args())
