
import pandas as pd

from lib.charts import Chart, Features, Indicator
from lib.models import Strategy, Classification, Stochastic
from lib.evaluation import Evaluator
from lib.simulation import Simulator

from itertools import product


Chart = Chart
Indicator = Indicator
Features = Features
Strategy = Strategy
Classification = Classification
Stochastic = Stochastic
Evaluator = Evaluator

class Metaparameters():
    """
    Utility class for creating a grid of meta parameters for an experiment and iterate over them.

    Attributes:
        zipped: An iterable of tuples, created using Python's zip built-in
        cartesian: An iterable of tuples, created using itertools 'product'-class
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor method.

        :param args: 0 or more iterables
        :param kwargs: 0 or more iterables, passed using keywords (recommended).
        """

        self.__iterables = list(args) + [kwargs[k] for k in kwargs]

    def __zip(self):
        return zip(*self.__iterables)

    def __product(self):
        return list(product(*self.__iterables))

    zipped = property(__zip)
    cartesian = property(__product)



pd.set_option('max_rows', 5000)
pd.set_option('max_columns', 20)
pd.options.display.float_format = '{:20,.2f}'.format

