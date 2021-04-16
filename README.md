# CMPUT466

## DNN

### Classification

`Data.py`
* `Data` class: A wrapper for the data
itself, making grabbing some of the information a little easier (e.g.
`Data.class_weight`). Also contains functions to load and save data from json
objects

* `DataPrep` class: Functions to convert the csv data to `Data` objects, and to
load and save `Data` objects as json files. Also includes functions to shuffle
and split the data into train and test sets.


`DNN.py` Is probably best explained by walking through the code in the `main`
function, and stepping into function calls when necessary.
* The first part of the function with all the `DataPrep` calls is to load and
cache the data. This is important because it allows us to keep the same train /
test splits after hyperparameter tuning and between runs.
* `tune_classification`: Performs hyperparameter tuning of the model using
`cv_tuner`
	* `cv_tuner`: Performs 5-fold cross validated Hyperband tuning of the model,
  parameterized by `classification_model_builder`
	* `classification_model_builder`: Uses kerastuner hyperparamters to allow
	selection of model parameters by the tuner
	* `classification_model_builder_refined`: Fine tunes the results of the
	above
* `train_classification_model`: The tuner output was hardcoded into
`create_classification_model`, which this function uses to create and then train
the model. This function shows plots of the training history of the model and
caches it as `model.json` and `model.h5`.
* `test_classification_model`: Grabs the cached model and tests it on the test
set. Prints out the accuracy and loss.
* `plot_classification_model`: Grabs the cached model and evaluates it on the
test set. The result is outputted as one of several different types of plots.
* `explain_model`: Grabs the cached model and runs a LIME explanation on an
entry from the test set. Shows the explanation as a LIME plot and a feature
overview plot
	* `generate_lime_plot`: A messy function that shows the LIME output as a
	nice pyplot. Kind of thrown together but gets the job done.
	* `generate_feature_overview_plot`: Shows a plot of the 10 most significant
	features as measured by their total change, and their influence on
	prediction of each rank, for the given data point.

### Regression

Our intial strategy was to use regression for the DNN. Most of the code is
roughly the same as the classification code, but using MMR values instead of
integer classes. The classification peformed so much better that we didn't even
include the DNN regression in the report. The code is just here for completness.

## DT

The main file for decision tress is `dtClassifier_CrossValidation.py`. This file 
starts of by reading in the data, shuffling it, then splitting it into test and train set
It then creates a decision tree object for grid search to use.

Within the `gridSeach` function, you can specifiy which hyperparameters to iterate over 
and what values they should take. It then iterates over then with 10 fold cross validation.
It reports the best parameters it found along with the best accuracy and returns a decision 
tree that utilizes those parameters.

The code then takes this decision tree and runs it against the test data that had been set 
aside earlier and reports the accuracy as well as creates a figure for the confusion matrix 
and decision tree.

The second file is `stats.py` which performs permutation testing. It shuffles the y labels 
of the data and runs the model against it recording it's accuracy. It then finds the percentage 
that did better than the true accuracy and reports that value.

## MLR

## Scraping

Contains (what's left of) the code used to scrape data from Ballchasing.com.
Ballchasing actually doesn't offer a complete list of all the entries for a
given rank, so the strategy was to use their "random" button to get as many
entries as possible for a given rank. Here is a breakdown of the functions in
`Fetch Data.main`:

* `initial_download`: Go through each rank and randomly pull games for that
rank and store the csvs in a folder. Tries up to 25 times to find a unique id
from the random selector before moving to the next rank.

* `redownload_data`: It was later found out that we were exceeding the rate
limit of the server, and a lot of the "csv" files were actually 403 error page
html files, which were much larger than the actual csv files. This function
finds the html files based on their size and re-pulls the csvs, with a 5 second
delay between requests.

* `compile_files`: Each csv contained 6 entries for the 6 players in one game.
This function conglomerates all the entries into one csv file.

