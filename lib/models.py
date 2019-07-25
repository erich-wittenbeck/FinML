
import collections
import pandas as pd

from numpy.random import choice

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Built-Ins for ML models

ml_algos = {'svm' : SVC,
            'randf' : RFC}
hpo_metrics = {}

# Actual classes

class Manual():

    def __init__(self, name, code, result_var='__signal__'):

        self.__name = name
        self.__result_var = result_var
        self.__code = compile(code, '<string>', 'exec')

    """ Getters """

    def __get_name(self):
        return self.__name

    """ Execute """

    def predict(self, input):

        # Local function to apply strategy on a per row/series basis

        def __predict__(local_dict):
            exec(self.__code, globals(), local_dict)

            return local_dict[self.__result_var]

        if type(input) == pd.Series:
            return __predict__(input.to_dict())
        else:
            X = input if type(input) == pd.DataFrame else input.X
            return X.apply(lambda row: __predict__(row.to_dict()), axis=1)

    """ Properties """

    name = property(__get_name)

class Classification():

    def __init__(self, name, algorithm):

        self.__features = []
        self.__name = name
        self.__algorithm = ml_algos[algorithm]
        self.__classifier = None
        self.__use_hpo = False
        self.__hyper_params = {}
        self.__hpo_config = {}


    """ Setters and Getters """

    def __get_name(self):
        return self.__name

    def __get_params(self):
        if isinstance(self.__classifier, GridSearchCV):
            return self.__classifier.best_params_
        else:
            return self.__classifier.get_params

    def __get_estimation_func(self):
        if hasattr(self.__classifier, 'decision_function'):
            return self.__classifier.decision_function
        elif hasattr(self.__classifier, 'predict_proba'):
            return self.__classifier.predict_proba
        elif self.__classifier is None:
            raise Exception('Classifier has not been trained yet!')
        else:
            raise TypeError('Classifier is expected to provide either decision_function or predict_proba as attributes!')

    def __get_classes(self):
        return self.__classifier.classes_

    def set_hyper_parameters(self, **hyper_params):

        self.__hpo_config = {} # Reset HPO configuration
        self.__hyper_params = hyper_params

        self.__use_hpo = any([isinstance(hyper_param, collections.Iterable) and not isinstance(hyper_param, str) for hyper_param in hyper_params.values()])

        return self

    def configure_hpo(self, scoring_metric, **hpo_configs):

        scoring = hpo_metrics[scoring_metric] \
            if scoring_metric in hpo_metrics \
            else scoring_metric

        self.__hpo_config = {'scoring' : scoring, **hpo_configs}

        return self

    def train(self, training_data, *features):

        self.__features = list(features)

        X = training_data.X[self.__features] if len(self.__features) > 0 else training_data.X
        y = training_data.y

        self.__classifier = GridSearchCV(self.__algorithm(), self.__hyper_params, **self.__hpo_config)\
            if self.__use_hpo \
            else self.__algorithm(**self.__hyper_params)

        self.__classifier.fit(X, y)

        return self

    def predict(self, input):

        X = input if type(input) in [pd.Series, pd.DataFrame] \
            else input.X

        if self.__features:
            X = X[self.__features]

        predictions = self.__classifier.predict(X)

        return pd.Series(predictions, X.index)

    """ Properties """

    name = property(__get_name)
    params = property(__get_params)
    estimation_func = property(__get_estimation_func)
    classes = property(__get_classes)

class Stochastic():

    def __init__(self, name, threshold=1):

        if threshold <= 0.5 or threshold > 1:
            raise ValueError("threshold - value must be in (0.5, 1] but was " + str(threshold) + "!")

        self.__name = name
        self.__threshold = threshold

        self.__distribution = {}

    # Getters/Setters

    def __get_name(self):

        return self.__name

    # Public methods

    def determine_distribution(self, labels):

        self.__distribution = labels.value_counts(True).to_dict()

        return self

    def predict(self, input):

        X = input if type(input) in [pd.Series, pd.DataFrame] \
            else input.X

        index = X.index
        pred_count = len(index)

        keys = list(self.__distribution.keys())
        probs = list(self.__distribution.values())

        if probs[0] > self.__threshold:
            predictions = [keys[0]]*pred_count
        else:
            predictions = choice(keys, pred_count, replace=True, p=probs)

        return pd.Series(predictions, index)

    # Properties

    name = property(__get_name)