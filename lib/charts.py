
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif

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
    """
    Class that represents an OHLCV-price chart, also known as a '(Japanese) candlestick chart.

    Properties:
        data: The underlying pandas dataframe, indexed using pandas timestamps and containing 5 columns: open, high, low, close and volume
    """

    def __init__(self, data_source):
        """
        Constructor for Chart-class.

        :param data_source: A dataframe, or path to a .csv-file as a string. Dataframe must be indexed with pandas timestamps, while the .csv-file ought to have a column 'timestamp' containing Unix-timestamps.
        """

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
        """
        Pre-processing method for dealing with missing values in a chart.

        :param freq: Frequency of the candlesticks, as a pandas frequency string.
        :param fill_method: Method for filling up missing values. 'ffill' for forward-fill, or 'interpolate' for interpolation
        :param kwargs: Other keyword arguments for the underlying pandas filling methods
        :return: The Chart-instance, with missing values filled in
        """
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
        """
        Method for scaling up the size of the candlesticks contained in the Chart-instance.

        :param scale: The new scale of the candlesticks, as a pandas frequency string
        :return: The Chart-instance, with rescaled candlesticks
        """

        chart = self.__data

        chart = chart.resample(scale) \
            .agg({'open': 'first',
                  'close': 'last',
                  'high': 'max',
                  'low': 'min',
                  'volume': 'sum'})

        self.__data = chart

        return self

    # Generatvie functions

    def randomize(self):
        """
        Creates a new Chart-instance containing a random walk, using this instance as a basis

        :return: A new Chart-instance, with a randomized price history
        """

        df = self.__data
        pctchg_df = df.pct_change().fillna(0).sample(frac=1, replace=True)

        new_columns = {}
        len_df = len(df)
        seed = df.iloc[0]

        for column in df.columns:
            if column == 'volume':
                new_columns['volume'] = df['volume'] #[df.iloc[i]['volume'] for i in range(0, len_df)]
            else:
                new_column = [seed[column]]
                for i in range(1, len_df):
                    prev_val = new_column[i-1]
                    new_val = prev_val + (prev_val/100)*pctchg_df.iloc[i][column]
                    new_column += [new_val]
                new_columns[column] = new_column

        return Chart(pd.DataFrame(new_columns, index=df.index).fillna(method='bfill'))



    # Accessing functions

    def slice(self, start, end=None):
        """
        Slices up a new Chart-instance from this instance, from one timestamp to the other. Timestamps are passed as strings, formatted

        YYYY-MM-DD hh:mm:ss

        However, only the year-portion is mandatory. If the hours, day or month are ommitted, it will default to 00:00:00 at the first day of the first month of the given year.

        :param start: A string specifying the starting timestamp of the slice
        :param end: Optional. A string specifying the final timestamp of the slice. If None, slice will go up until last timestamp of original instance
        :return: A Chart-instance, containing only the specified slice of the original chart.
        """
        chart_slice = slice_timeseries(self.__data, start, end)

        return Chart(chart_slice)

    def split(self, chunk_size):
        """
        Splits up the Chart-instance into multiple instances, each covering a certain time span of the original.

        :param chunk_size: The size of the individual chunks. If a string, can be 'weeks', 'months', 'quarters' and 'years'. If a float between 0 and 1, indicates the percentage size of the first and second chunks
        :return: A tuple of new Chart-instances
        """
        chart_splits = split_timeseries(self.__data, chunk_size)

        return tuple([Chart(split) for split in chart_splits])

    def copy(self, no_of_copies=1):
        """
        Create one or more copies of this Chart-instance.

        :param no_of_copies: Optional, the number of copies to be created. Default 1
        :return: A tuple containing the copied instances, even if it is only one.
        """
        return Chart(self.__data.copy()) if no_of_copies == 1 \
            else tuple(Chart(self.__data.copy()) for i in range(no_of_copies))

    # I/O

    def save_as(self, filename):
        """
        Saves the innder dataframe of the Chart-instance to a file

        :param filename: The path to the target file
        """
        self.__data.to_csv(filename)

    # Plotting

    def plot(self, from_date=None, to_date=None):
        """
        Plots the Chart or a segment of it, using matplotlib. Starting- and end-points are specified the same way as with Chart.slice

        :param from_date: Optional, starting timestamp as a string.
        :param to_date: Optional, ending timestamp as a string
        """

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
    """
    A class representing the feature matrix computed over a candlestick chart.

    Attributes:
        chart: The underlying Chart-instance
        X: The pandas dataframe containing the features. Indexed over the same timestamps as the underlying chart
        y: The pandas series containing the labels for each candle. Indexed over the same timestamps as the underlying chart
        dist: A tuple of two pandas series, indexed over the column (feature) names of X. The first contains the mean for each feature, the other one the variances.
        minmax: A tuple of two pandas series, indexed over the column (feature) names of X. The first contains the minimum of each feature, the other one the maximums.
    """

    def __init__(self, data, feature_matrix=None):
        """
        Constructor for the Features-class.

        :param data: A Chart-instance
        :param feature_matrix: Optional. A Features-instance. If given, has to have the same index as the Chart-instance
        """

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
        """
        Creates the labels for the underlying chart using a function. The function must take a pandas dataframe as first input and return a pandas series,
        indexed over the same timestamps as the dataframe, as output. The dataframe has the structure as specified in the Chart-class.

        :param with_func: The function for creating the labels, either a callable or a string. If string, it must refer to one of the built-in labeling functions.
        :param args: Further arguments to the labeling function
        :param kwargs: Further keyword argument to the labeling function
        :return: A pandas series.
        """

        label_func = label_funcs[with_func] \
            if type(with_func) == str \
            else with_func

        self.__feature_matrix['labels'] = label_func(self.__chart, *args, **kwargs)

        self.__feature_matrix['labels'].fillna(method='bfill')

        return self

    def add_chart_columns(self, *columns, smooth_with_alpha=None):
        """
        Add one of the OHLCV-columns of the underlying chart as a feature.

        :param columns: One or more column names as strings
        :param smooth_with_alpha: Optional. Float between 0 and 1, used as alpha for an exponential smoothing for the chosen columns
        :return: The Features-instance with new features
        """

        list_of_columns = list(columns)
        chart = self.__chart[list_of_columns]

        if smooth_with_alpha is not None:
            chart = chart.ewm(alpha=smooth_with_alpha, axis=1).mean()

        self.__feature_matrix[list_of_columns] = chart

        return self

    def add_indicators(self, *indicator, smooth_chart_columns=None, alpha=1):
        """
        Adds technical indicators as features to the feature matrix.

        :param indicator: One or more Indicator-instances.
        :param smooth_chart_columns: Optional. A list of strings, containing the names of the OHLCV-columns that will be smoothed before passing them to the indicators
        :param alpha: Optional. If smooth_chart_columns is given, specifies the alpha of the exponential smoothing. Default 1
        :return: The Features-instance with new features
        """

        chart = self.__chart.copy()

        if smooth_chart_columns is not None:
            chart[smooth_chart_columns] = chart[smooth_chart_columns].ewm(alpha=alpha, axis=1).mean()

        # TODO: Optimize!

        for feature in indicator:
            self.__feature_matrix = self.__feature_matrix.assign(**feature(chart))

        self.__feature_matrix.fillna(method='bfill', inplace=True)

        return self

    def add_predictions(self, *models):
        """
        Adds the label predictions of predictive models as features.

        :param models: One or more objects implementing the model-protocol.
        :return: The Features-instance with new features
        """

        X = self.__feature_matrix

        prediction_tuples = [(model.name, model.predict(X)) for model in models]
        prediction_dict = dict(prediction_tuples)

        self.__feature_matrix = self.__feature_matrix.assign(**prediction_dict)

        return self

    # Feature transformation

    def standardize(self, with_params_of=None):
        """
        Standardizes the features in the feature matrix to each have a mean of 0 and variance of 1.

        :param with_params_of: Optional. A Features-instance whose dist-property will be used for the computation
        :return: The Features-instance with standardized features
        """

        X = self.__get_X()
        features = X.columns

        if with_params_of is None:

            self.__means, self.__vars = X.mean(), X.std()
        else:
            self.__means, self.__vars = with_params_of.dist

        self.__feature_matrix[features] = ((X - self.__means)/self.__vars).fillna(0)

        return self

    def normalize(self, with_params_of):
        """
        Normalizes the values for each feature in the feature matrix between 0 and 1.

        :param with_params_of: Optional. A Features-instance whose minmax-property will be used for the computation
        :return: The Features-instance with normalized features
        """
        X = self.__get_X()
        features = X.columns

        if with_params_of is None:
            self.__mins, self.__maxs = X.min(), X.max()
        else:
            self.__mins, self.__maxs = with_params_of.minmax

        self.__feature_matrix[features] = ((X - self.__mins)/(self.__maxs - self.__mins)).fillna(0)

        return self

    def rolling_window(self, offset, periods, *features):
        """
        Creates rolling windows of the existing features and adds them to the matrix. The new features' names are their old ones,
        appended with '__t-n', where 'n' is the number of shifts in relation to the original chart.

        :param offset: Initial offset as positive integer
        :param periods: The number of periods for which rolling windows are created.
        :param features: One or more strings, containing the feature names from which rolling windows are to be created
        :return: The Features-instance with new features
        """

        to_be_shifted = self.__get_X()[list(features)] if features != () else self.__get_X()

        shifted = [to_be_shifted.shift(i, axis=0).fillna(0).rename(mapper=lambda s: s + "__" + "t-" + str(i), axis='columns')
                   for i in range(offset+1, offset+periods+1)]

        self.__feature_matrix = pd.concat([self.__feature_matrix, *shifted], axis=1)

        return self

    # Feature selection

    def prune_features(self, n, score_func=f_classif): # TODO Use built_in string!
        """
        Implements univariate feature selection, pruning all but the n most relevant features.

        :param n: The number of relevant features to be selected
        :param score_func: Optional. The scoring function for testing relevance of features. Default is f_classif-function from Scikit-Learn
        :return: The Features-instance with only n features now
        """

        X = self.__get_X()
        y = self.__get_y()

        mask = []

        if type(n) == int:
            k = n
            best_k_selector = SelectKBest(score_func, k).fit(X, y)
            mask = best_k_selector.get_support()
        elif type(n) == str:
            best_k_selector = SelectKBest(score_func, 'all').fit(X, y)
            feature_scores = best_k_selector.scores_
            baseline_score = feature_scores[X.columns.get_loc(n)]
            mask = [True if feature_scores[i] > baseline_score else False for i in range(len(feature_scores))]
        else:
            raise TypeError('prune_features: expected n to be int or str but was ' + str(type(n)))

        X = X[X.columns[mask]]

        self.__feature_matrix = pd.concat([X, y], axis=1)

        return self

    # Accessing functions

    def slice(self, start, end=None):
        """
        Works analogously to the slice-method of the Chart-class.

        :param start: see above
        :param end: see above
        :return: A Features-instance
        """

        chart_slice = slice_timeseries(self.__chart, start, end)

        feature_slice = slice_timeseries(self.__feature_matrix, start, end)

        return Features(chart_slice, feature_slice)

    def split(self, chunk_size):
        """
        Works analogously to the split-method of the Chart-class.

        :param chunk_size: see above
        :return: A tuple of Features-instances
        """

        split_chart = split_timeseries(self.__chart, chunk_size)

        split_features = split_timeseries(self.__feature_matrix, chunk_size)

        return tuple([Features(chart_slice, features_slice) for chart_slice, features_slice in zip(split_chart, split_features)])

    def copy(self, no_of_copies=1):
        """
        Works analogously to the copy-mtehod of the Chart-class.

        :param no_of_copies: see above
        :return: A tuple of Features-instances
        """

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
    """
    A class of callable objects that wrap around technical indicator-functions, in order to modify them and give context
    """

    def __init__(self, identifier, func, **params):
        """
        Constructor method.

        :param identifier: The string identifier for the Indicator. This will be the name under which it will be added inside a Features-instance
        :param func: The technical indicator. Must be a string referencing one of the built-in indicators or a callable. An indicator must fulfill the same protocoll as a labeling function for the Features-instance
        :param params: Additional keyword arguments for the indicator-function
        """

        func = indicator_funcs[func] \
            if type(func) == str \
            else func
        self.__identifier = identifier
        self.__func = lambda chart: func(chart, **params)

    def __call__(self, chart):
        result = self.__func(chart)

        identifier = self.__identifier
        return {identifier: result}

    def rescale(self, scaling):
        """
        Set rescaling of the chart before computing the indicator.

        :param scaling: The new candlesize the chart is to be rescaled to. A pandas frequency string
        :return: The Indicator-instance
        """

        old_func = self.__func

        def new_func(chart):
            scaled_chart = chart.resample(scaling) \
                        .agg({'open': 'first',
                              'close': 'last',
                              'high': 'max',
                              'low': 'min',
                              'volume': 'sum'})

            return old_func(scaled_chart)

        self.__func = new_func

        return self

    def smooth_chart(self, alpha, *chart_columns):
        """
        Set smoothing of the chart before computing the indicator.

        :param alpha: The alpha for the exponential smoothing. Float between 0 and 1
        :param chart_columns: The names of the chart columns to be smoothed, as strings
        :return: The Indicator-instance
        """

        old_func = self.__func

        def new_func(chart):
            chart[list(chart_columns)] = chart[list(chart_columns)].ewm(alpha=alpha, axis=1).mean()

            return old_func(chart)

        self.__func = new_func

        return self
