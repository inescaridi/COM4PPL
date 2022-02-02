import numpy as np
import pandas as pd
import textdistance
from thefuzz import fuzz

# Valid algorithms
comparisonAlgorithms = {
    # Edit based
    'Levenshtein': textdistance.levenshtein.normalized_similarity,
    'D_L': textdistance.damerau_levenshtein.normalized_similarity,
    'MLIPNS': textdistance.mlipns.normalized_similarity,
    'Hamming': textdistance.hamming.normalized_similarity,
    'Jaro-Winkler': textdistance.jaro_winkler.normalized_similarity,
    'Strcmp95': textdistance.strcmp95.normalized_similarity,
    'Needleman-Wunsch': textdistance.needleman_wunsch.normalized_similarity,
    'Gotoh': textdistance.gotoh.normalized_similarity,
    'Smith-Waterman': textdistance.smith_waterman.normalized_similarity,

    # From The Fuzz
    'Ratio': (lambda x, y: fuzz.ratio(x, y) / 100),
    'Partial_ratio': (lambda x, y: fuzz.partial_ratio(x, y) / 100),
    'Token_sort_ratio': (lambda x, y: fuzz.token_sort_ratio(x, y) / 100),
    'WRATIO': (lambda x, y: fuzz.WRatio(x, y) / 100),

    # Token based
    'Jaccard_index': textdistance.jaccard.normalized_similarity,
    'Sorensen-Dice': textdistance.sorensen_dice.normalized_similarity,
    'Tversky_index': textdistance.tversky.normalized_similarity,
    'Overlap_coefficient': textdistance.overlap.normalized_similarity,
    'Tanimoto_distance': textdistance.tanimoto.normalized_similarity,
    'Cosine_similarity': textdistance.cosine.normalized_similarity,
    'Monge-Elkan': textdistance.monge_elkan.normalized_similarity,
    'Bag_distance': textdistance.bag.normalized_similarity,
}

# CONFIGURATION
config = {}


def loadConfig(data):
    # files
    config['configFilesPath'] = data['configFilesPath']

    # compatible rows config
    variableFields = pd.read_csv(f"{data['configFilesPath']}/{data['variableFields']}").set_index('variable')
    compatibleData = pd.read_csv(f"{data['configFilesPath']}/{data['compatibleData']}").set_index('variable')
    compatibleData['equivalences'] = compatibleData[compatibleData.type == 'categorical'].parameter.apply(
        loadEquivalences)
    config['compatibleSettings'] = compatibleData.join(variableFields).to_dict(orient='index')

    # schemes
    config['schemesToUse'] = pd.read_csv(f"{data['configFilesPath']}/{data['schemesToUse']}", names=['schemes'])[
        'schemes'].to_list()
    # TODO add option for "fast scheme"
    schemesConfig = pd.read_csv(f"{data['configFilesPath']}/{data['schemesConfig']}")
    schemesConfig.rename(columns={'listA.column': 'key1', 'listB.column': 'key2'}, inplace=True)
    if schemesConfig.dtypes['threshold'] == str:
        schemesConfig.threshold = schemesConfig.threshold.str.replace(',', '.').astype(float)
    config['schemesConfig'] = schemesConfig

    # algorithms
    algorithmsConfig = pd.read_csv(f"{data['configFilesPath']}/{data['algorithmsConfig']}")
    algorithmsConfig['function'] = algorithmsConfig.scores.map(comparisonAlgorithms)
    config['algorithmsConfig'] = algorithmsConfig


def loadEquivalences(filename):
    if not isNaN(filename):
        equivalencesDF = pd.read_csv(f"{config['configFilesPath']}/{filename}")
        return list(equivalencesDF.itertuples(index=False, name=None))
    else:
        return np.nan


def isNaN(value):
    if type(value) == str:
        return value.upper() == 'NAN'
    elif type(value) == list:
        return len(value) == 0
    else:
        return np.isnan(value)


def findCompatibleRows(row1, db2):
    compatibles = db2

    for variable, options in config['compatibleSettings'].items():
        if options['consider'].lower() == 'yes':
            allowNa = options['na.action'].upper() == 'ALL'

            for field in options['fields'].split(' '):
                if field not in row1 or field not in db2.columns:
                    continue

                # If the value on the current row is NaN and matching with NaN is allowed,
                # we should not filter anything, since anything will match with it
                if isNaN(row1[field]) and allowNa:
                    continue

                filtered = pd.DataFrame()

                # Type: Range
                if options['type'] == 'range':
                    filtered = compatibles[compatibles[field].between(
                        row1[field] - int(options['parameter']),
                        row1[field] + int(options['parameter']))]
                # Type: Categorical
                elif options['type'] == 'categorical':
                    filtered = compatibles[compatibles[field] == row1[field]]
                    equivalences = options['equivalences']

                    # TODO find another way to represent equivalences or "move" to preprocessing.
                    if not isNaN(equivalences):
                        for v1, v2 in equivalences:
                            equivalent = pd.DataFrame()

                            if v1 == row1[field]:
                                equivalent = compatibles[compatibles[field] == v2]
                            elif v2 == row1[field]:
                                equivalent = compatibles[compatibles[field] == v1]

                            filtered = pd.concat([filtered, equivalent])

                # No matter the type it is always the same search for NaN values if allowed
                if allowNa:
                    nan = compatibles[compatibles[field].isna()]
                    compatibles = pd.concat([filtered, nan])
                else:
                    compatibles = filtered

    return compatibles.index


def findCompatibles(base1, base2):
    compatibles = base1.apply(lambda r1: findCompatibleRows(r1, base2), axis=1)
    # Convert compatibles index series to complete dataframe with row information
    compatiblesList = []
    for i1, c in compatibles.iteritems():
        for i2 in c:
            row1 = base1.iloc[i1]
            row2 = base2.iloc[i2]

            row1.set_axis(['base1_' + x for x in base1.columns], inplace=True)
            row2.set_axis(['base2_' + x for x in base2.columns], inplace=True)

            compatiblesList.append(pd.concat([row1, row2]))

    return pd.DataFrame(compatiblesList)


def findCandidates(compatiblesDF, verbose=False):
    for scheme, key1, key2, threshold in config['schemesConfig'].itertuples(index=False):
        key1 = 'base1_' + key1
        key2 = 'base2_' + key2

        # check for valid scheme and keys
        if scheme not in config['schemesToUse']:
            continue
        if verbose:
            print(f"Running scheme {scheme}")
        if key1 not in compatiblesDF.columns:
            print(f"Error: Invalid key on database 1 {key1}, skipping")
            continue
        if key2 not in compatiblesDF.columns:
            print(f"Error: Invalid key on database 2 {key2}, skipping")
            continue

        medianCount = 0
        compatiblesDF[f"sum_median_{scheme}"] = 0

        for algorithmName, doCalculate, addToMedian, function in config['algorithmsConfig'].itertuples(index=False):
            if doCalculate:
                compatiblesDF[f"{algorithmName}_{scheme}"] = compatiblesDF.apply(
                    lambda x: round(function(x[key1], x[key2]), 4), axis=1)

                if addToMedian:
                    compatiblesDF[f"sum_median_{scheme}"] += compatiblesDF[f"{algorithmName}_{scheme}"]
                    medianCount += 1

        compatiblesDF[f"median_{scheme}"] = round(compatiblesDF[f"sum_median_{scheme}"] / medianCount, 4)
        compatiblesDF.drop(f"sum_median_{scheme}", axis=1, inplace=True)
        compatiblesDF = compatiblesDF[compatiblesDF[f"median_{scheme}"] >= threshold]

    return sortCandidatesDF(compatiblesDF)


def sortCandidatesDF(candidatesDF):
    candidatesDF.to_csv(f"prueba.csv", index=False)
    sortPriority = [f"median_{scheme}" for scheme in config['schemesToUse']]
    return candidatesDF.sort_values(
        sortPriority, ascending=False, kind='stable', ignore_index=True)
