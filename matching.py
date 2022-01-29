import json
import os
import sys

import pandas as pd

import com4ppl

import time
import psutil


def main(interactive, verbose, debug):
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
    makeMatching(data, verbose, debug)


def makeMatching(data, verbose, debug):
    maxUsedMem = 0
    # output folder
    os.makedirs(data['outputPath'], exist_ok=True)

    # load databases
    print(f"Loading databases ")
    base1 = pd.read_csv(data['base1Path'])
    base2 = pd.read_csv(data['base2Path'])
    print(f"\tdb1: {data['base1Path']}, size: {base1.shape[0]}")
    print(f"\tdb2: {data['base2Path']}, size: {base2.shape[0]}")

    if debug:
        mem = round(python_process.memory_info().rss / 1e+9, 4)  # TODO check if this is correct
        maxUsedMem = max(maxUsedMem, mem)
        print(f"\tDEBUG>> Used memory: {mem} GB")

    # find compatible rows
    print("Finding compatible rows")
    sub_timer_start = time.time()
    compatiblesDF = base1.apply(lambda r1: base2.apply(lambda r2: com4ppl.areCompatibles(r1, r2), axis=1), axis=1)
    i1, i2 = compatiblesDF.values.nonzero()
    print(f"\tFound {len(i1)} compatible rows in {round(time.time() - sub_timer_start, 4)} seconds")

    if debug:
        mem = round(python_process.memory_info().rss / 1e+9, 4)
        maxUsedMem = max(maxUsedMem, mem)
        print(f"\tDEBUG>> Used memory: {mem} GB")

    # run configured schemes on compatible pairs
    print("Running schemes on compatible pairs")

    candidatesList = []
    sub_timer_start = time.time()

    for i1, i2 in zip(i1, i2):
        row1 = base1.iloc[i1]
        row2 = base2.iloc[i2]

        candidates, information = com4ppl.areCandidates(row1, row2, base1, base2, verbose)

        if candidates:
            # rename columns as base1_ and base2_
            row1.set_axis(['base1_' + x for x in base1.columns], inplace=True)
            row2.set_axis(['base2_' + x for x in base2.columns], inplace=True)

            candidatesList.append(pd.concat([row1, row2, information]))

    if debug:
        mem = round(python_process.memory_info().rss / 1e+9, 4)
        maxUsedMem = max(maxUsedMem, mem)
        print(f"\tDEBUG>> Used memory: {mem} GB")

    print(f"\tFound {len(candidatesList)} candidates in {round(time.time() - sub_timer_start, 4)} seconds")

    # output result
    if len(candidatesList) > 0:
        candidatesDF = com4ppl.getSortedCandidatesDF(candidatesList)
        candidatesDF.to_csv(f"{data['outputPath']}/{data['outputFileName']}.csv", index=False)
        print(f"Finished sorting candidates, results are in {data['outputPath']}")
    else:
        print("No candidates found")

    if debug:
        print(f"DEBUG>> Maximum used memory: {maxUsedMem} GB")


if __name__ == '__main__':
    pid = os.getpid()
    python_process = psutil.Process(pid)

    interactiveOp = verboseOp = debugOp = False

    # options must be placed after script name: "python matching.py options"
    if len(sys.argv) > 1:
        if '-i' in sys.argv[1:]:
            interactiveOp = True
        if '-v' in sys.argv[1:]:
            verboseOp = True
        if '-d' in sys.argv[1:]:
            debugOp = True

        # TODO add an argument to set up extra config and/or databases paths

    timer_start = time.time()
    main(interactiveOp, verboseOp, debugOp)
    print(f"Finished in {round(time.time() - timer_start, 4)} seconds")
