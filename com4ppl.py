import numpy as np
import pandas as pd
import textdistance
from numpy import median
from pandas import DataFrame
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


def load_config(data):
    # files
    config['configFilesPath'] = data['configFilesPath']

    # compatible rows config
    variable_fields = pd.read_csv(f"{data['configFilesPath']}/{data['variableFields']}").set_index('variable')
    compatible_data = pd.read_csv(f"{data['configFilesPath']}/{data['compatibleData']}").set_index('variable')
    compatible_data['equivalences'] = compatible_data[compatible_data.type == 'categorical'].parameter.apply(
        load_equivalences)
    config['compatibleSettings'] = compatible_data.join(variable_fields).to_dict(orient='index')

    # schemes
    config['schemesToUse'] = pd.read_csv(f"{data['configFilesPath']}/{data['schemesToUse']}", names=['schemes'])[
        'schemes'].to_list()
    # TODO add option for "fast scheme"
    schemes_config = pd.read_csv(f"{data['configFilesPath']}/{data['schemesConfig']}")
    schemes_config.rename(columns={'listA.column': 'key1', 'listB.column': 'key2'}, inplace=True)
    if schemes_config.dtypes['threshold'] == str:
        schemes_config.threshold = schemes_config.threshold.str.replace(',', '.').astype(float)
    config['schemesConfig'] = schemes_config

    # algorithms
    algorithms_config = pd.read_csv(f"{data['configFilesPath']}/{data['algorithmsConfig']}")
    algorithms_config['function'] = algorithms_config.scores.map(comparisonAlgorithms)
    config['algorithmsConfig'] = algorithms_config


def load_equivalences(filename):
    if not is_nan(filename):
        equivalences_df = pd.read_csv(f"{config['configFilesPath']}/{filename}")
        return list(equivalences_df.itertuples(index=False, name=None))
    else:
        return np.nan


def is_nan(value) -> bool:
    if type(value) == str:
        return value.upper() == 'NAN'
    elif type(value) == list:
        return len(value) == 0
    else:
        return np.isnan(value)


def find_compatible_rows(row1, db2):
    compatibles = db2

    for variable, options in config['compatibleSettings'].items():
        if options['consider'].lower() == 'yes':
            allow_na = options['na.action'].upper() == 'ALL'

            for field in options['fields'].split(' '):
                if field not in row1 or field not in db2.columns:
                    continue

                # If the value on the current row is NaN and matching with NaN is allowed,
                # we should not filter anything, since anything will match with it
                if is_nan(row1[field]) and allow_na:
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
                    if not is_nan(equivalences):
                        for v1, v2 in equivalences:
                            equivalent = pd.DataFrame()

                            if v1 == row1[field]:
                                equivalent = compatibles[compatibles[field] == v2]
                            elif v2 == row1[field]:
                                equivalent = compatibles[compatibles[field] == v1]

                            filtered = pd.concat([filtered, equivalent])

                # No matter the type it is always the same search for NaN values if allowed
                if allow_na:
                    nan = compatibles[compatibles[field].isna()]
                    compatibles = pd.concat([filtered, nan])
                else:
                    compatibles = filtered

    return compatibles.index


def find_compatibles(base1: DataFrame, base2: DataFrame) -> DataFrame:
    compatibles = base1.apply(lambda r1: find_compatible_rows(r1, base2), axis=1)
    # Convert compatibles index series to complete dataframe with row information
    compatibles_list = []
    for i1, c in compatibles.items():
        for i2 in c:
            row1 = base1.iloc[i1].set_axis(['base1_' + x for x in base1.columns])
            row2 = base2.iloc[i2].set_axis(['base2_' + x for x in base2.columns])

            compatibles_list.append(pd.concat([row1, row2]))

    return pd.DataFrame(compatibles_list)


def safe_apply(function, row, key1, key2):
    if is_nan(row[key1]):
        print(f"No value found on base1 with key: {key1[6:]}, DEFAULTING to zero")
        return 0
    if is_nan(row[key2]):
        print(f"No value found on base2 with key: {key2[6:]}, DEFAULTING to zero")
        return 0
    return round(function(row[key1], row[key2]), 4)


def find_candidates(compatibles: DataFrame, verbose=False) -> DataFrame:
    for scheme, key1, key2, threshold in config['schemesConfig'].itertuples(index=False):
        key1 = 'base1_' + key1
        key2 = 'base2_' + key2

        # check for valid scheme and keys
        if scheme not in config['schemesToUse']:
            continue
        if verbose:
            print(f"Running scheme {scheme}")
        if key1 not in compatibles.columns:
            print(f"Error: Invalid key on database 1 {key1}, skipping")
            continue
        if key2 not in compatibles.columns:
            print(f"Error: Invalid key on database 2 {key2}, skipping")
            continue

        compatibles[f"elems_median_{scheme}"] = 0

        for algorithmName, doCalculate, addToMedian, function in config['algorithmsConfig'].itertuples(index=False):
            if 'yes' in doCalculate.lower():
                value = compatibles.apply(lambda row: safe_apply(function, row, key1, key2), axis=1)
                compatibles[f"{algorithmName}_{scheme}"] = value

                if addToMedian:
                    compatibles[f"elems_median_{scheme}"] += value

        compatibles[f"median_{scheme}"] = median(compatibles[f"elems_median_{scheme}"])
        compatibles.drop(f"elems_median_{scheme}", axis=1, inplace=True)
        compatibles = compatibles[compatibles[f"median_{scheme}"] >= threshold]

    return sort_candidates_df(compatibles)


def sort_candidates_df(candidates: DataFrame) -> DataFrame:
    candidates.to_csv(f"prueba.csv", index=False)

    sort_priority = []
    for scheme in config['schemesToUse']:
        median_colname = f"median_{scheme}"
        if median_colname in candidates:
            sort_priority.append(median_colname)
        else:
            print(f"No {median_colname} found in candidates, perhaps there was an error in the calculation")

    return candidates.sort_values(
        sort_priority, ascending=False, kind='stable', ignore_index=True)
