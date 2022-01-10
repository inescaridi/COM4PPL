# COM4PPL
Comparison for People: context-assisted name matching comparison


# For users:
## Set up and running
1- Clone or download this repository.
2- Run `pip install -r requirements.txt` to install required libraries.
3- Run the script `python3 matching.py`, you can also run it in interactive (`-i`) and/or verbose (`-v`) modes.

## Interactive mode
It will guide you through the configuration, allowing you to pick the databases paths, output folder and output file name.

## Verbose mode
It will show information of what the script is doing, for example which scheme is running. Note: this is done for each candidate pair, so it can be a lot of information when using big databases. This option is mostly meant to check the scheme configuration is as the user wants it to.

## Configuration (files are inside `config` folder)
- `variableFields` and `compatible_data` are used to determine compatible rows.
- Only schemes listed in `select_schemes` will be run and it will also determine the final sorting of the candidates.
- Schemes are defined in `info_scheme`, every scheme has a name, the column that should be used to compare and a threshold.
- `info_score` declares which algorithms will be calculated and which of them will be part of the final median calculation. Each of this medians (there's one for each scheme) will be compared with the threshold and only candidates with a higher median will be part of the result.

# For devs:
## TODO: Add script and library explanation


# Folders:
- config: has all the configuration files.
- dicts: has dictionary used to specify common misnomer or synonyms between values. This will be used in preprocessing script, not finished yet.
- docs: for now it only has the old documentation for the R script.
- example: has 'toy' databases to use as an example.
- notebooks: this is deprecated but not deleted because it can be useful to future devs, as a user please run the scripted version instead of the notebook as the latter can be outdated.


# How to set up a virtualenv (optional)
We highly recommend using a virtualenv instead of installing dependencies directly. For more information on how to do this please visit [this guide](https://docs.python-guide.org/dev/virtualenvs/).

# References
This project uses the following libraries for string comparison:
- [the fuzz](https://github.com/seatgeek/thefuzz)
- [textdistance](https://github.com/life4/textdistance)