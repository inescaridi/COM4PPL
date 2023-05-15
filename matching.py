import json
import os
import sys

import pandas as pd

import com4ppl

import time
import psutil


def main(interactive, verbose):
    with open("config.json") as json_data_file:
        data = json.load(json_data_file)

    if interactive:
        # ask for database paths
        base1_path = input("Please insert database 1 path (leave empty to use default):")
        while base1_path != '' and not os.path.isfile(base1_path):
            print("File does not exists")
            base1_path = input("Please insert database 1 path (leave empty to use default):")

        base2_path = input("Please insert database 2 path (leave empty to use default):")
        while base2_path != '' and not os.path.isfile(base2_path):
            print("File does not exists")
            base2_path = input("Please insert database 2 path (leave empty to use default):")

        if base1_path != '':
            data['base1Path'] = base1_path
        if base2_path != '':
            data['base2Path'] = base2_path

        res = input(f"Do you want to change output folder (from {data['outputPath']})? (y/n):")
        if 'y' in res:
            output_path = input("Please insert new output path:")
            success = False
            while not success:
                try:
                    os.makedirs(output_path, exist_ok=True)
                except OSError as e:
                    print(f"Failed to find/create output folder with error: {e}")
                    output_path = input("Please insert new output path:")
                else:
                    success = True
            data['outputPath'] = output_path

        output_file_name = input("Please insert output file name (leave empty to use default):")
        if output_file_name != '':
            data['outputFileName'] = output_file_name

    com4ppl.load_config(data)
    make_matching(data, verbose)


def make_matching(data, verbose):
    max_used_mem = 0
    # output folder
    os.makedirs(data['outputPath'], exist_ok=True)

    # load databases
    print(f"Loading databases ")
    base1 = pd.read_csv(data['base1Path'])
    base2 = pd.read_csv(data['base2Path'])
    print(f"\tdb1: {data['base1Path']}, size: {base1.shape[0]}")
    print(f"\tdb2: {data['base2Path']}, size: {base2.shape[0]}")

    # TODO check if this way of getting the used mem is correct
    max_used_mem = max(max_used_mem, round(python_process.memory_info().rss / 1e+9, 4))

    # find compatible rows ------------------------------------------------------------- step 1
    print("Finding compatible pairs")
    sub_timer_start = time.time()

    compatibles_df = com4ppl.find_compatibles(base1, base2)

    print(f"\tFound {compatibles_df.shape[0]} compatible rows in {round(time.time() - sub_timer_start, 4)} seconds")
    max_used_mem = max(max_used_mem, round(python_process.memory_info().rss / 1e+9, 4))

    # run configured schemes on compatible pairs  -------------------------------------- step 2
    print("Running schemes on compatible pairs")
    sub_timer_start = time.time()

    candidates_df = com4ppl.find_candidates(compatibles_df, verbose)

    print(f"\tFound {candidates_df.shape[0]} candidates in {round(time.time() - sub_timer_start, 4)} seconds")
    max_used_mem = max(max_used_mem, round(python_process.memory_info().rss / 1e+9, 4))

    # output result -------------------------------------------------------------------- step 3
    if candidates_df.shape[0] > 0:
        candidates_df.to_csv(f"{data['outputPath']}/{data['outputFileName']}.csv", index=False)
        print(f"Finished sorting candidates, results are in {data['outputPath']}")
    else:
        print("No candidates found")

    print(f"Maximum used memory: {max_used_mem} GB")


if __name__ == '__main__':
    pid = os.getpid()
    python_process = psutil.Process(pid)

    interactiveOp = verboseOp = False

    # options must be placed after script name: "python matching.py options"
    if len(sys.argv) > 1:
        if '-i' in sys.argv[1:]:
            interactiveOp = True
        if '-v' in sys.argv[1:]:
            verboseOp = True

        # TODO add an argument to set up extra config and/or databases paths

    timer_start = time.time()
    main(interactiveOp, verboseOp)
    print(f"Finished in {round(time.time() - timer_start, 4)} seconds")
