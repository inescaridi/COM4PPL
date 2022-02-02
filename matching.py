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
        base1Path = input("Please insert database 1 path (leave empty to use default):")
        while base1Path != '' and not os.path.isfile(base1Path):
            print("File does not exists")
            base1Path = input("Please insert database 1 path (leave empty to use default):")

        base2Path = input("Please insert database 2 path (leave empty to use default):")
        while base2Path != '' and not os.path.isfile(base2Path):
            print("File does not exists")
            base2Path = input("Please insert database 2 path (leave empty to use default):")

        if base1Path != '':
            data['base1Path'] = base1Path
        if base2Path != '':
            data['base2Path'] = base2Path

        res = input(f"Do you want to change output folder (from {data['outputPath']})? (y/n):")
        if 'y' in res:
            outputPath = input("Please insert new output path:")
            success = False
            while not success:
                try:
                    os.makedirs(outputPath, exist_ok=True)
                except OSError as e:
                    print(f"Failed to find/create output folder with error: {e}")
                    outputPath = input("Please insert new output path:")
                else:
                    success = True
            data['outputPath'] = outputPath

        outputFileName = input("Please insert output file name (leave empty to use default):")
        if outputFileName != '':
            data['outputFileName'] = outputFileName

    com4ppl.loadConfig(data)
    makeMatching(data, verbose)


def makeMatching(data, verbose):
    maxUsedMem = 0
    # output folder
    os.makedirs(data['outputPath'], exist_ok=True)

    # load databases
    print(f"Loading databases ")
    base1 = pd.read_csv(data['base1Path'])
    base2 = pd.read_csv(data['base2Path'])
    print(f"\tdb1: {data['base1Path']}, size: {base1.shape[0]}")
    print(f"\tdb2: {data['base2Path']}, size: {base2.shape[0]}")

    # TODO check if this way of getting the used mem is correct
    maxUsedMem = max(maxUsedMem, round(python_process.memory_info().rss / 1e+9, 4))

    # find compatible rows ------------------------------------------------------------- step 1
    print("Finding compatible pairs")
    sub_timer_start = time.time()

    compatiblesDF = com4ppl.findCompatibles(base1, base2)

    print(f"\tFound {compatiblesDF.shape[0]} compatible rows in {round(time.time() - sub_timer_start, 4)} seconds")
    maxUsedMem = max(maxUsedMem, round(python_process.memory_info().rss / 1e+9, 4))

    # run configured schemes on compatible pairs  -------------------------------------- step 2
    print("Running schemes on compatible pairs")
    sub_timer_start = time.time()

    candidatesDF = com4ppl.findCandidates(compatiblesDF, verbose)

    print(f"\tFound {candidatesDF.shape[0]} candidates in {round(time.time() - sub_timer_start, 4)} seconds")
    maxUsedMem = max(maxUsedMem, round(python_process.memory_info().rss / 1e+9, 4))

    # output result -------------------------------------------------------------------- step 3
    if candidatesDF.shape[0] > 0:
        candidatesDF.to_csv(f"{data['outputPath']}/{data['outputFileName']}.csv", index=False)
        print(f"Finished sorting candidates, results are in {data['outputPath']}")
    else:
        print("No candidates found")

    print(f"Maximum used memory: {maxUsedMem} GB")


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
