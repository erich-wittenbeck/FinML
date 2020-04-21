
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

class Strategy():
    """
    A class implementing predictive models manually created by a human using python code.

    The python code can reference the features of a feature matrix by using their names as variable identifiers

    Attributes:
        name: The string identifier for this model
    """

    def __init__(self, name, code, result_var='__signal__'):
        """
        Constructor method.

        :param name: The string identifier for this model
        :param code: The pythond code, as a string
        :param result_var: Optional. The name of the variable which will store the prediction result inside the code. Default '__signal__'
        """

        self.__name = name
        self.__result_var = result_var
        self.__code = compile(code, '<string>', 'exec')

    """ Getters """

    def __get_name(self):
        return self.__name

    """ Execute """

    def predict(self, input):
        """
        Predict the labels for a given feature matrix

        :param input: Either a Features-instance or the X-attribute of a Features-instance
        :return: A pandas series containing the predictions for each instance in the input. Indexed over the same timestamps as the input
        """

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
    """
    A class implementing predictive models using machine learning algorithms.

    Attributes:
        name: The string identifier for this instance
        params: The parameters of the model, as a dictionary
        classes: The classes/labels predicted by the model, as a numpy array
        features: The features used by the model, as a list of strings
        estimation_func: The underlying estimation function of the model. Either it's 'decision_function' or 'predict_proba'
    """

    def __init__(self, name, algorithm):
        """
        Constructor method.

        :param name: The string identifier of the model
        :param algorithm: The machine learning algorithm. Either a class implementing Scikit-Learn's Predictor-protocol or a string referencing one of the built-in models.
        """

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
        """
        Set the hyper parameters of the learning process.

        :param hyper_params: The hyper parameters of the underlying ml algorithm as keyword arguments. Each argument may be passed either a scalar value or a list of scalars.
        :return: The Classification-instance with hyper parameters set.
        """

        self.__hpo_config = {} # Reset HPO configuration
        self.__hyper_params = hyper_params

        self.__use_hpo = any([isinstance(hyper_param, collections.Iterable) and not isinstance(hyper_param, str) for hyper_param in hyper_params.values()])

        return self

    def configure_hpo(self, hpo_method, scoring_metric, **hpo_configs):
        """
        Configures hyper parameter optimization, if at least one of the parameters in set_hyper_parameters was passed a list of values.

        :param hpo_method: Sets the HPO method. Either 'exhaustive' or 'random'
        :param scoring_metric: The metric with which model performance is evaluated. A legal value for the 'scoring'-parameter of Scikit-Learn's GridSearchCV
        :param hpo_configs: Keyword arguments for Scikit-Learn's GridSearchCV
        :return: The Classification-instance with configured HPO
        """

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

    def train(self, training_data, prune_features=False, k=None, rfe_folds=None, rfe_scoring=None):
        """
        Trains the actual model.

        :param training_data: A Features-instance.
        :param prune_features: Optional. Set if recursive feature elimination is to be performed. Default False
        :param k: Optional. Limits the number of features to be selected to k. Positive integer
        :param rfe_folds: Optional. Number of folds to be used during RFE. Positive integer.
        :param rfe_scoring: Optional. Scoring metric used for RFE. String or callable
        :return: The Classification-instance with learned model
        """

        X, y = training_data.X, training_data.y

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
        """
        Predict labels for a given input feature matrix.

        :param input: A Features-instance or a pandas series or dataframe
        :return: A pandas series containing the predictions for the input instances.
        """

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
    """
    A class for creating predictions based on the label distribution of the training data

    Attributes:
        name: The string identifier for this model.
        params: The parameters of the model. Dictionary with only one entry: 'threshold', the decision threshold.
        classes: The classes predicted by the model
        features: The features found in the input data
        estimation_func: The function predicting labels based on the training data's label distribution.
    """

    def __init__(self, name):
        """
        Constructor method.

        :param name: The string identifier for this instance
        """

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
        """
        Determine the label distribution of the given training data.

        :param train_data: A Features-instance
        :param threshold: Optional. A probability between 0.5 and 1, if any label exceeds this threshold, only that label will be predicted. Default 1
        :return: The Stochastic-instance
        """

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
        """
        Predict labels for a given input

        :param input: A Features-instance or pandas series or dataframe
        :return: A pandas series containing the prediction for input.
        """

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