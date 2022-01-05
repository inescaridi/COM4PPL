import os
from com4ppl import *
import json


def main():
    with open("config.json") as json_data_file:
        data = json.load(json_data_file)
    loadConfig(data)

    # load databases
    base1 = pd.read_csv(data['base1Path'])
    base2 = pd.read_csv(data['base2Path'])

    # output folder
    outputPath = data['outputPath']
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    # MAIN SCRIPT
    # find compatible rows
    compatiblesDF = base1.apply(lambda r1: base2.apply(lambda r2: areCompatibles(r1, r2), axis=1), axis=1)
    i1, i2 = compatiblesDF.values.nonzero()
    compatibles = list(zip(i1, i2))

    # run configured schemes on compatible pairs
    candidatesList = []

    for i1, i2 in compatibles:
        row1 = base1.iloc[i1]
        row2 = base2.iloc[i2]

        candidates, information = areCandidates(row1, row2, base1, base2)

        if candidates:
            # rename columns as base1_ and base2_
            row1.set_axis(['base1_' + x for x in base1.columns], inplace=True)
            row2.set_axis(['base2_' + x for x in base2.columns], inplace=True)

            candidatesList.append(pd.concat([row1, row2, information]))

    # output result
    candidatesDF = pd.concat(candidatesList, axis=1).T
    candidatesDF.to_csv(f"{data['outputPath']}/{data['outputFileName']}.csv", index=False)


if __name__ == '__main__':
    main()
