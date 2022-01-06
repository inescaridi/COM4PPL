import os, sys
import com4ppl
import pandas as pd
import json


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

    com4ppl.loadConfig(data)
    makeMatching(data, verbose)


def makeMatching(data, verbose):
    # output folder
    os.makedirs(data['outputPath'], exist_ok=True)

    # load databases
    if verbose:
        print(f"Loading databases, db1: {data['base1Path']}, db2: {data['base2Path']}")
    base1 = pd.read_csv(data['base1Path'])
    base2 = pd.read_csv(data['base2Path'])

    # find compatible rows
    if verbose:
        print("Finding compatible rows")
    compatiblesDF = base1.apply(lambda r1: base2.apply(lambda r2: com4ppl.areCompatibles(r1, r2), axis=1), axis=1)
    i1, i2 = compatiblesDF.values.nonzero()
    compatibles = list(zip(i1, i2))

    # run configured schemes on compatible pairs
    if verbose:
        print("Running schemes on compatible pairs")

    candidatesList = []

    for i1, i2 in compatibles:
        row1 = base1.iloc[i1]
        row2 = base2.iloc[i2]

        candidates, information = com4ppl.areCandidates(row1, row2, base1, base2, verbose)

        if candidates:
            # rename columns as base1_ and base2_
            row1.set_axis(['base1_' + x for x in base1.columns], inplace=True)
            row2.set_axis(['base2_' + x for x in base2.columns], inplace=True)

            candidatesList.append(pd.concat([row1, row2, information]))

    # output result
    candidatesDF = pd.concat(candidatesList, axis=1).T
    candidatesDF.to_csv(f"{data['outputPath']}/{data['outputFileName']}.csv", index=False)

    print(f"END, results are in {data['outputPath']}")


if __name__ == '__main__':
    interactive = verbose = False

    if len(sys.argv) > 1:
        if '-i' in sys.argv[1:]:
            interactive = True
        if '-v' in sys.argv[1:]:
            verbose = True

    main(interactive, verbose)
