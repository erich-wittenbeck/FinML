
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from lib.indicators.momentum import \
    moving_average_convergence_divergence as macd, \
    triple_exponential_moving_average as trix, \
    average_directional_index as adx, \
    vortex_indicator as vrtx
from lib.indicators.oscillation import \
    relative_strength_index as rsi, \
    money_flow_index as mfi, \
    williams_percent_range as wpr, \
    awesome_oscillator as ao, \
    ultimate_oscillator as uo, \
    stochastic_oscillator as stoch
from lib.indicators.volatility import \
    average_true_range as atr, \
    mass_index as midx, \
    bollinger_bands as bbands, \
    keltner_channel as kltch
from lib.indicators.volume import \
    on_balance_volume as obv, \
    force_index as fidx, \
    chaikin_money_flow as cmf, \
    accumulation_distribution_line as adl

from lib.utility.labeling import *
from lib.utility.timeseries import *
from lib.utility.statistics import exponential_smoothing as exs, min_max_norm
from lib.utility.visualization import plot_candlesticks



label_funcs =     {'binary': binary_labels,
                   'ternary' : ternary_labels,
                   'quartary' : quartary_labels,
                   'pentary' : pentary_labels}
indicator_funcs = {'macd' : macd,
                   'trix' : trix,
                   'adx' : adx,
                   'vrtx' : vrtx,
                   'rsi' : rsi,
                   'mfi' : mfi,
                   'wpr' : wpr,
                   'ao' : ao,
                   'uo' : uo,
                   'stoch' : stoch,
                   'atr' : atr,
                   'midx' : midx,
                   'bbands' : bbands,
                   'kltch' : kltch,
                   'obv' : obv,
                   'fidx' : fidx,
                   'cmf' : cmf,
                   'adl' : adl}


class Chart():

    def __init__(self, data_source):

        chart = None

        if type(data_source) == str:
            chart = pd.read_csv(data_source)[['timestamp', 'open', 'close', 'high', 'low', 'volume']]
            chart['timestamp'] = pd.to_datetime(chart['timestamp'], unit='s')
            chart.set_index('timestamp', inplace=True)
        elif type(data_source) == pd.DataFrame:
            chart = data_source
        else:
            raise TypeError('history_data: Must either be a filepath-string or DataFrame!')

        self.__data = chart

    # Setters and getters

    def __get_data(self):
        return self.__data

    def __set_data(self, chart_df):
        self.__data = chart_df

    # Transformatory functions

    def fill_missing_data(self, freq, fill_method, **kwargs):
        chart = self.__data

        chart = chart.reindex(pd.date_range(start=chart.index[0], end=chart.index[-1], freq=freq))

        if fill_method == 'ffill':
            chart = chart.ffill(**kwargs)
        elif fill_method == 'interpolate':
            chart = chart.interpolate(**kwargs)
        else:
            raise ValueError(fill_method + ": Unknown fill method!")

        self.__data = chart

        return self

    def upscale(self, scale):
        chart = self.__data

        chart = chart.resample(scale) \
            .agg({'open': 'first',
                  'close': 'last',
                  'high': 'max',
                  'low': 'min',
                  'volume': 'sum'})

        self.__data = chart

        return self

    def smooth(self, alpha, *columns):
        chart = self.__data
        chart[list(columns)] = chart[list(columns)].apply(exs, args=(alpha,))

        self.__data = chart

        return self

    # Generatvie functions

    def shuffle(self, how_many=1, with_replacement=False):
        index = self.__data.index

        return Chart(self.__data.sample(frac=1, replace=with_replacement).set_index(index)) if how_many == 1\
            else tuple(Chart(self.__data.sample(frac=1, replace=with_replacement).set_index(index)) for i in range(how_many))

    # Accessing functions

    def slice(self, start, end=None):
        chart_slice = slice_timeseries(self.__data, start, end)

        return Chart(chart_slice)

    def split(self, chunk_size):
        chart_splits = split_timeseries(self.__data, chunk_size)

        return tuple([Chart(split) for split in chart_splits])

    def copy(self, no_of_copies=1):
        return Chart(self.__data.copy()) if no_of_copies == 1 \
            else tuple(Chart(self.__data.copy()) for i in range(no_of_copies))

    # I/O

    def save_as(self, filename):
        self.__data.to_csv(filename)

    # Plotting

    def plot(self, from_date=None, to_date=None):

        chart_df = None

        if from_date != None and to_date != None:
            chart_df = slice_timeseries(self.__data, from_date, to_date)
        else:
            chart_df = self.__data

        fig, ax = plt.subplots()

        plot_candlesticks(ax, chart_df)

        plt.show()

        plt.clf()

    # Properties

    data = property(__get_data)

class Features():

    def __init__(self, data, feature_matrix=None):
        if type(data) == Chart:
            self.__chart = data.data
        elif type(data) == pd.DataFrame:
            chart_columns = ['open', 'close', 'high', 'low', 'volume']
            self.__chart = data[chart_columns]
        elif type(data) == str:
            chart_columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume']
            data_frame = pd.read_csv(data)
            data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'], unit='s')
            data_frame.set_index('timestamp', inplace=True)

            self.__chart = data_frame[chart_columns]
        else:
            raise TypeError('history: Must either be a History-object, DataFrame or filepath!')

        if feature_matrix is None:
            self.__feature_matrix = pd.DataFrame(index=self.__chart.index.copy(), columns=['labels'])
        else:
            self.__feature_matrix = feature_matrix

        self.__alphas = {}

        self.__means = None
        self.__vars = None

        self.__mins = None
        self.__maxs = None


    # Getters

    def __get_chart(self):
        return self.__chart

    def __get_X(self):
        return self.__feature_matrix.drop('labels', axis=1)

    def __get_y(self):
        return self.__feature_matrix['labels']

    def __get_dist(self):
        return self.__means, self.__vars

    def __get_min_max(self):
        return self.__mins, self.__maxs

    # Public access points for adding labels and features

    def label_data(self, with_func, *args, **kwargs):

        label_func = label_funcs[with_func] \
            if type(with_func) == str \
            else with_func

        self.__feature_matrix['labels'] = label_func(self.__chart, *args, **kwargs)

        self.__feature_matrix['labels'].fillna(method='bfill')

        return self

    def smooth_chart(self, alpha, *chart_columns):

        for chart_column in chart_columns:
            self.__alphas[chart_column] = alpha

        return self

    def add_features(self, *features):

        chart = self.__chart

        for column in self.__alphas:
            chart[column] = chart[[column]].apply(exs, args=(self.__alphas[column],))

        # TODO: Optimize!

        for feature in features:
            if type(feature) == str:
                self.__feature_matrix = self.__feature_matrix.assign(**{feature: chart[feature]})
            elif type(feature) == Indicator:
                self.__feature_matrix = self.__feature_matrix.assign(**feature.compute(chart))

        self.__feature_matrix.fillna(method='bfill', inplace=True)

        return self

    def add_predictions(self, *models):

        X = self.__feature_matrix

        prediction_tuples = [(model.name, model.predict(X)) for model in models]
        prediction_dict = dict(prediction_tuples)

        self.__feature_matrix = self.__feature_matrix.assign(**prediction_dict)

        return self

    # Feature transformation

    def standardize(self, with_params_of=None):
        X = self.__get_X()
        features = X.columns

        if with_params_of is None:

            self.__means, self.__vars = X.mean(), X.std()
        else:
            self.__means, self.__vars = with_params_of.dist

        self.__feature_matrix[features] = ((X - self.__means)/self.__vars).fillna(0)

        return self

    def normalize(self, with_params_of):
        X = self.__get_X()
        features = X.columns

        if with_params_of is None:
            self.__mins, self.__maxs = X.min(), X.max()
        else:
            self.__mins, self.__maxs = with_params_of.minmax

        self.__feature_matrix[features] = ((X - self.__mins)/(self.__maxs - self.__mins)).fillna(0)

        return self

    def rolling_window(self, offset, periods, *features):

        to_be_shifted = self.__get_X()[list(features)] if features != () else self.__get_X()

        shifted = [to_be_shifted.shift(i, axis=0).fillna(0).rename(mapper=lambda s: s + "__" + "t-" + str(i), axis='columns')
                   for i in range(offset+1, offset+periods+1)]

        self.__feature_matrix = pd.concat([self.__feature_matrix, *shifted], axis=1)

        return self

    # Feature selection

    def prune_features(self, k, score_func=f_classif): # TODO Use built_in string!

        X = self.__get_X()
        y = self.__get_y()

        best_k_selector = SelectKBest(score_func, k).fit(X, y)
        mask = best_k_selector.get_support()

        X = X[X.columns[mask]]

        self.__feature_matrix = pd.concat([X, y], axis=1)

        return self

    # Accessing functions

    def slice(self, start, end=None):

        chart_slice = slice_timeseries(self.__chart, start, end)

        feature_slice = slice_timeseries(self.__feature_matrix, start, end)

        return Features(chart_slice, feature_slice)

    def split(self, chunk_size):

        split_chart = split_timeseries(self.__chart, chunk_size)

        split_features = split_timeseries(self.__feature_matrix, chunk_size)

        return tuple([Features(chart_slice, features_slice) for chart_slice, features_slice in zip(split_chart, split_features)])

    def copy(self, no_of_copies=1):

        copied_chart = self.__chart.copy()

        copied_features = self.__feature_matrix.copy()

        return Features(copied_chart, copied_features) if no_of_copies == 1 \
            else tuple(Features(copied_chart, copied_features) for i in range(no_of_copies))

    # Properties

    chart = property(__get_chart)
    X = property(__get_X)
    y = property(__get_y)
    dist = property(__get_dist)
    minmax = property(__get_min_max)


class Indicator():

    def __init__(self, identifier, func, **params):

        self.__func = indicator_funcs[func] \
            if type(func) == str \
            else func
        self.__identifier = identifier
        self.__params = params

        self.__scaling = None
        self.__alphas = {}

    def __add_to_dict(self, dictionary, value, *keys):

        for key in keys:
            dictionary[key] = value

    def rescale(self, scaling):

        self.__scaling = scaling

        return self

    def smooth_chart(self, alpha, *chart_columns):

        self.__add_to_dict(self.__alphas, alpha, *chart_columns)

        return self

    def compute(self, chart):
        df = chart

        # Apply upscaling and smoothing
        if self.__scaling is not None:
            df = df.resample(self.__scaling) \
                .agg({'open': 'first',
                      'close': 'last',
                      'high': 'max',
                      'low': 'min',
                      'volume': 'sum'})

        for column in self.__alphas:
            chart[[column]] = chart[[column]].apply(exs, args=(self.__alphas[column],))

        # Compute indicator
        result = self.__func(df, **self.__params)

        identifier = self.__identifier
        return {identifier: result}