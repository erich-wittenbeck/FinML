
# About FinML

*FinML* is a library that was developed over the course and as the subject of my Master's thesis in 2019.
It's aim is to bundle the functionality of the *Scikit-Learn* stack in a single API to enable the writing of
compact code for learning and evaluating machine learning models on financial time series.

As of now, it is to be understood as a prototype, as it can only deal with static data read out from .csv-files.

## Requirements

*FinML* was developed under the following dependencies:

- Pandas 0.24.2
- Numpy 1.16.4
- Matplotlib 3.1.3
- Scikit-Learn 0.21.1

While newer versions of the above *should* also work, this has not been tested yet.

# Basic Usage

A *FinML*-script can be thought of as a pipeline generally consisting of several stages. For more detailed information
on how the below steps work, check the documentation inside the respective modules.

## Reading in data

CSV-data is read into using the *Chart*-class:

````python
from finml import *

history = Chart('/path/to/your/chart/data.csv') \
    .fill_missing_data('min', 'interpolate')\
    .upscale('60min')
````

This reads in a .csv-file, containing the timestamped (Unix-timestamps) price movement
data. Columns must be named 'timestamp', 'open', 'high', 'low', 'close' and 'volume'.

## Creating feature matrices

A Chart-instance may passed to a *Features*-instance to label the data and add technical
indicators as features

````python
features = Features(history) \
        .label_data('ternary', 1, 0, 0)\
        .add_indicators(Indicator('rsi', 'rsi'),
                        Indicator('wpr', 'wpr'),
                        Indicator('uo', 'uo')
                        # Add or remove indicators
                        )

training_data, test_data = features.split(0.75)
````

The *label_data*-method can be passed either a callable or a string referencing
one of the built-in labeling methods, in this case 'ternary' (sell, hold, buy), and additional arguments
for the given labeling function.

Indicators work similarily, their constructors receive a string identifier (the feature name) and a callable or reference
to a built-in technical indicator.

See *lib/utility/labeling.py* and the modules under *lib/indicators* for an overview of available built-ins.

## Creating Models

To create machine learned models, instantiate the *Classification*-class with a string identifier and either a
class constructor that implements *Scikit-Learn's* Predictor-protocol or a string referencing one of the built ins.

````python
randf = Classification('randf', 'randf')\
                    .set_hyper_parameters(n_estimators = 100,
                                          bootstrap = True)\
                    .train(training_data, prune_features=True, rfe_scoring='f1_macro')

logreg = Classification('logreg', 'logreg') \
            .set_hyper_parameters(C=[0.0001, 0.001, 0.01, 0, 10, 100, 1000]) \
            .configure_hpo('exhaustive', 'f1_macro', n_jobs=1, verbose=2) \
            .train(training_data, prune_features=True, rfe_scoring='f1_macro')
````

As of now, *FinML* offers short-cuts to Logistic Regression ('logreg'), Support Vector Machines ('svm') and 
Random Forests ('randf').

## Evaluating Models

Models are evaluated using the *Evaluator*-class:

````python
evaluator = Evaluator().evaluate(test_data, logreg, randf, evaluations=['reports'])
````

This passes a set of test data alongside two trained models to the evaluator and let's it create a Scikit-Learn
classification report.

It can be plotted using matplotlib via

````python
evaluator.plot(save_as='/path/to/plot.png')
````

Available evaluations are 'reports', 'confmats' (confusion matrices) and 'roc_curves' (ROC-analysis).