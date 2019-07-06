
import pandas as pd

from lib.charts import Chart, Features, Indicator
from lib.models import Strategy, Classifier, StochasticModel
from lib.evaluation import Evaluator
from lib.simulation import Simulator

from itertools import product


Chart = Chart
Indicator = Indicator
Features = Features
Strategy = Strategy
Classifier = Classifier
StochasticModel = StochasticModel
Evaluator = Evaluator
Simulator = Simulator

class Metaparameters():

    def __init__(self, *args, **kwargs):

        self.__iterables = args + (kwargs[k] for k in kwargs)

    def __zip(self):
        return zip(*self.__iterables)

    def __product(self):
        return product(self.__iterables)

    zipped = property(__zip)
    cartesian = property(__product)



pd.set_option('max_rows', 5000)
pd.set_option('max_columns', 20)
pd.options.display.float_format = '{:20,.2f}'.format

