import numpy as np
import pandas as pd
import textdistance
from thefuzz import fuzz

# CONFIGURATION
config = {
    "comparisonAlgorithms": {
        'Levenshtein': textdistance.levenshtein.normalized_similarity,
        'DL': textdistance.damerau_levenshtein.normalized_similarity,
        'fuzzy_ratio': fuzz.ratio
    }  # TODO add the rest/more
}


def loadConfig(data):
    variableFields = pd.read_csv(f"{data['configFilesPath']}/{data['variableFields']}").set_index('variable')
    compatibleData = pd.read_csv(f"{data['configFilesPath']}/{data['compatibleData']}").set_index('variable')
    config['comparisonSettings'] = compatibleData.join(variableFields).to_dict(orient='index')

    # schemes
    config['schemesAllowed'] = pd.read_csv(f"{data['configFilesPath']}/{data['schemesAllowed']}", names=['schemes'])[
        'schemes'].to_list()
    # TODO add option for "fast scheme"
    schemesConfig = pd.read_csv(f"{data['configFilesPath']}/{data['schemesConfig']}")
    schemesConfig.rename(columns={'listA.column': 'key1', 'listB.column': 'key2'}, inplace=True)
    if schemesConfig.dtypes['threshold'] == str:
        schemesConfig.threshold = schemesConfig.threshold.str.replace(',', '.').astype(float)
    config['schemesConfig'] = schemesConfig

    config['configFilesPath'] = data['configFilesPath']


def isNaN(value):
    if type(value) == str:
        return value.upper() == 'NAN'
    else:
        return np.isnan(value)


def areEquivalentValues(val1, val2, equivalences):
    return (val1 == val2 or (val1, val2) in equivalences) or ((val2, val1) in equivalences)


def compatibleRanges(row1, row2, options, verbose=False):
    compatible = True  # need to be compatible on all available fields (can be changed for 'or')
    allowNa = options['na.action'] == 'all'

    # for debugging
    val1 = val2 = 'NAN'

    for field in options['fields'].split(' '):
        if field not in row1 or field not in row2:
            continue

        if isNaN(row1[field]) or isNaN(row2[field]):
            compatible &= allowNa
        else:
            compatible &= abs(int(row1[field]) - int(row2[field])) <= int(options['parameter'])
            val1 = int(row1[field])
            val2 = int(row2[field])

    # for debugging
    if verbose:
        print(f"\tval1:{val1}, val2:{val2}, allowNa: {allowNa}, range:{int(options['parameter'])}")

    return compatible


def compatibleCategory(row1, row2, options, verbose=False):
    compatible = True  # need to be compatible on all available fields (can be changed for 'or')
    allowNa = options['na.action'].upper() == 'ALL'
    equivalences = None

    # for debugging
    val1 = val2 = 'NAN'

    if not isNaN(options['parameter']):
        equivalencesDF = pd.read_csv(f"{config['configFilesPath']}/{options['parameter']}")
        equivalences = list(equivalencesDF.itertuples(index=False, name=None))
        # WARNING, there may be a better way to do this other than creating tuples but we should have a scheme
        # to follow, in order to grab the columns without having to specify their names

    for field in options['fields'].split(' '):
        if field not in row1 or field not in row2:
            continue

        if isNaN(row1[field]) or isNaN(row2[field]):
            compatible &= allowNa
        elif equivalences:
            compatible &= areEquivalentValues(row1[field], row2[field], equivalences)
            val1 = row1[field]
            val2 = row2[field]

    # for debugging
    if verbose:
        print(f"\tval1:{val1}, val2:{val2}, allowNa: {allowNa}")

    return compatible


def areCompatibles(row1, row2, verbose=False):
    compatible = True

    for variable, options in config['comparisonSettings'].items():
        if options['consider'].lower() == 'yes':
            if verbose:
                print(f"{variable} compatible?")
            varCompatible = True

            if options['type'] == 'range':
                varCompatible = compatibleRanges(row1, row2, options, verbose)
            elif options['type'] == 'categorical':
                varCompatible = compatibleCategory(row1, row2, options, verbose)

            compatible &= varCompatible
            if verbose:
                print(f"\tresult: {varCompatible}")
    if verbose:
        print(f"Candidates? {compatible} \n\n")
    return compatible


def areCandidates(row1, row2, base1, base2, verbose=False):
    candidates = True

    for scheme, key1, key2, threshold in config['schemesConfig'].itertuples(index=False):
        # check for valid scheme and keys
        if scheme not in config['schemesAllowed']:
            if verbose:
                print(f"scheme not used: {scheme}")
            continue
        if verbose:
            print(f"scheme {scheme}")
        if key1 not in base1.columns:
            print(f"invalid key on database 1 {key1}, skipping")
            continue
        if key2 not in base2.columns:
            print(f"invalid key on database 2 {key2}, skipping")
            continue

        value1 = row1[key1]
        value2 = row2[key2]

        information = {}

        for algorithmName, algorithm in config['comparisonAlgorithms'].items():
            result = algorithm(value1, value2)
            candidates &= (result >= threshold)

            information[f"{algorithmName}_{scheme}"] = result
    return candidates, pd.Series(information)
