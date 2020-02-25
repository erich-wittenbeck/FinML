
import pandas as pd

from numpy.random import normal

def random_standard_variable(df, mean=0, sdev=1):

    index = df.index
    samples = normal(mean, sdev, len(index))

    return pd.Series(samples, index)
