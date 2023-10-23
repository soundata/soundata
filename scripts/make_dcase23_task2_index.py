import argparse
import glob
import json
import os
from soundata.validate import md5

INDEX_PATH = "../soundata/datasets/indexes/dcase23_task2_index.json"

def make_index(data_path):
    machines = ["fan", "gearbox", "bearing", "slider", "ToyCar", "ToyTrain", "valve"]
    machines_additional = ["Vacuum", "ToyTank", "ToyNscale", "ToyDrone", "bandsaw", "grinder", "shaker"]
    machines_eval = ["Vacuum", "ToyTank", "ToyNscale", "ToyDrone", "bandsaw", "grinder", "shaker"]
    
    index = {
        "version": "1.0",
        "clips": {},
        "metadata": {}
    }
    
    # Loop for machines (Development Set)
    for machine in machines:
        audio_dev_train_folder = os.path.join("7882613", machine, "train")
        audio_dev_test_folder = os.path.join("7882613", machine, "test")
        metadata_dev_file = os.path.join("7882613", machine, "attributes_00.csv")

        if "dev" not in index["metadata"]:
            index["metadata"]["dev"] = {}

        index["metadata"]["dev"][machine] = [
            metadata_dev_file,
            md5(os.path.join(data_path, metadata_dev_file))
        ]

        # Dev Train clips
        for clip in glob.glob(os.path.join(data_path, audio_dev_train_folder, "*.wav")):
            clip_id = os.path.basename(clip).replace(".wav", "")
            index["clips"][clip_id] = {
                "audio": [
                    os.path.join(audio_dev_train_folder, os.path.basename(clip)),
                    md5(clip)
                ]            
            }

        # Dev Test clips
        for clip in glob.glob(os.path.join(data_path, audio_dev_test_folder, "*.wav")):
            clip_id = os.path.basename(clip).replace(".wav", "")
            index["clips"][clip_id] = {
                "audio": [
                    os.path.join(audio_dev_test_folder, os.path.basename(clip)),
                    md5(clip)
                ]
            }

    # Loop for machines_additional (Additional Training Set)
    for machine in machines_additional:
        audio_add_train_folder = os.path.join("7830345", machine, "train")
        metadata_add_file = os.path.join("7830345", machine, "attributes_00.csv")

        if "add_train" not in index["metadata"]:
            index["metadata"]["add_train"] = {}

        index["metadata"]["add_train"][machine] = [
            metadata_add_file,
            md5(os.path.join(data_path, metadata_add_file))
        ]

        # Additional Train clips
        for clip in glob.glob(os.path.join(data_path, audio_add_train_folder, "*.wav")):
            clip_id = os.path.basename(clip).replace(".wav", "")
            index["clips"][clip_id] = {
                "audio": [
                    os.path.join(audio_add_train_folder, os.path.basename(clip)),
                    md5(clip)
                ]
            }

    # Loop for machines_eval (Evaluation Set)
    for machine in machines_eval:
        audio_eval_folder = os.path.join("7860847", machine, "test")
        
        # Eval clips (assuming there's no CSV file for eval set)
        for clip in glob.glob(os.path.join(data_path, audio_eval_folder, "*.wav")):
            clip_id = os.path.basename(clip).replace(".wav", "")
            index["clips"][clip_id] = {
                "audio": [
                    os.path.join(audio_eval_folder, os.path.basename(clip)),
                    md5(clip)
                ]
            }

    with open(INDEX_PATH, "w") as fhandle:
        json.dump(index, fhandle, indent=2)

def main(args):
    make_index(args.data_path)

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Generate DCASE23_Task2 index file.")
    PARSER.add_argument("data_path", type=str, help="Path to DCASE23_Task2 folder.")
    main(PARSER.parse_args())
