
import collections
import pandas as pd

from numpy import array
from numpy.random import choice

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LogReg

# Built-Ins for ML models

ml_algos = {'svm' : SVC,
            'logreg' : LogReg,
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

    def __get_features(self):
        return list(self.__features)

    def __get_classes(self):
        return self.__classifier.classes_

    def set_hyper_parameters(self, **hyper_params):

        self.__hpo_config = {} # Reset HPO configuration
        self.__hyper_params = hyper_params

        self.__use_hpo = any([isinstance(hyper_param, collections.Iterable) and not isinstance(hyper_param, str) for hyper_param in hyper_params.values()])

        return self

    def configure_hpo(self, hpo_method, scoring_metric, **hpo_configs):

        scoring = hpo_metrics[scoring_metric] \
            if scoring_metric in hpo_metrics \
            else scoring_metric

        if hpo_method == 'exhaustive':
            self.__hpo_method = GridSearchCV
        elif hpo_method == 'random':
            self.__hpo_method = RandomizedSearchCV
        else:
            raise ValueError("hpo_method: expected either 'exhaustive' or 'random' but was " + str(hpo_method))

        self.__hpo_config = {'scoring' : scoring, **hpo_configs}

        return self

    def train(self, training_data, *features, prune_features=False, k=None, rfe_folds=None, rfe_scoring=None):

        X, y = training_data.X, training_data.y

        if not prune_features and len(features) > 0:
            X = X[list(features)]

        algorithm = self.__algorithm
        hyper_params = self.__hyper_params

        if self.__use_hpo:

            if prune_features:
                hyper_params = {'estimator__' + key: hyper_params[key] for key in hyper_params}
                rfe = RFE(algorithm(), k) if k is not None else RFECV(algorithm(), cv=rfe_folds, scoring=self.__hpo_config['scoring'])

                gridsearch = self.__hpo_method(rfe, hyper_params, **self.__hpo_config)
                gridsearch.fit(X, y)

                best_rfe = gridsearch.best_estimator_

                self.__classifier = best_rfe.estimator_
                self.__features = X[X.columns[best_rfe.support_]].columns
            else:
                gridsearch = self.__hpo_method(algorithm(), hyper_params, **self.__hpo_config)
                gridsearch.fit(X, y)

                self.__classifier = gridsearch.best_estimator_
                self.__features = X.columns
        else:
            if prune_features:
                rfe = RFE(algorithm(**hyper_params), k) if k is not None else RFECV(algorithm(), cv=rfe_folds, scoring=rfe_scoring)
                rfe.fit(X, y)

                self.__classifier = rfe.estimator_
                self.__features = X[X.columns[rfe.support_]].columns
            else:
                self.__classifier = algorithm(**hyper_params)
                self.__classifier.fit(X, y)

                self.__features = X.columns

        return self

    def predict(self, input):

        X = input if type(input) in [pd.Series, pd.DataFrame] \
            else input.X

        if len(self.__features) > 0:
            X = X[self.__features]

        predictions = self.__classifier.predict(X)

        return pd.Series(predictions, X.index)

    """ Properties """

    name = property(__get_name)
    params = property(__get_params)
    classes = property(__get_classes)
    features = property(__get_features)
    estimation_func = property(__get_estimation_func)

class Stochastic():

    def __init__(self, name):

        self.__distribution = {}

        self.__name = name
        self.__threshold = 1
        self.__classes = []
        self.__features = []
        self.__estimation_func = lambda x: x

    # Getters/Setters

    def __get_name(self):

        return self.__name

    def __get_params(self):

        return {'threshold': self.__threshold}

    def __get_classes(self):

        return self.__classes

    def __get_features(self):

        return self.__features

    def __get_estimation_func(self):

        return self.__estimation_func


    # Public methods

    def determine_distribution(self, train_data, threshold=1):

        if threshold <= 0.5 or threshold > 1:
            raise ValueError("threshold - value must be in (0.5, 1] but was " + str(threshold) + "!")

        self.__distribution = train_data.y.value_counts(True).to_dict()

        self.__threshold = threshold
        self.__classes = list(self.__distribution.keys())
        self.__features = train_data.X.columns

        def predict_proba(data):
            n_samples = len(data.index)

            return array([[self.__distribution[label]]*n_samples for label in self.__classes]).T

        self.__estimation_func = predict_proba


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
    params = property(__get_params)
    classes = property(__get_classes)
    features = property(__get_features)
    estimation_func = property(__get_estimation_func)