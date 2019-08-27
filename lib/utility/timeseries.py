
import pandas as pd
import numpy as np

from datetime import datetime
from lib.utility.datetime import map_str_to_datetime

def slice_timeseries(df, start, end=None):
    if (end is not None) and type(start) != type(end):
        raise TypeError("end: end needs to be of the same type as start, but was " + str(type(end)) + ' instead of ' + str(type(start)))

    if type(start) == int:
        start_index = start
        end_index = end if end is not None else -1
    elif type(start) == float:
        start_index = int(start*len(df))
        end_index = int(end*len(df)) if end is not None else -1
    elif type(start) == datetime:
        start_index = df.index.get_loc(start)
        end_index = df.index.get_loc(end) if end is not None else -1
    elif type(start) == str:
        start_index = df.index.get_loc(map_str_to_datetime(start))
        end_index = df.index.get_loc(map_str_to_datetime(end)) if end is not None else -1
    else:
        raise TypeError('start end: start and end must be int, float, str or date time, but were ' + str(type(start)) + ' and ' + str(type(end)) + ' instead!')

    return df[start_index : end_index]


def split_timeseries(df, chunk_size):

    if type(chunk_size) == str:
        freq_str = \
            'D' if chunk_size == 'day' else \
                'W' if chunk_size == 'week' else \
                    'M' if chunk_size == 'month' else \
                        'Q' if chunk_size == 'quarter' else \
                            'Y' if chunk_size == 'year' else \
                                None
        return [chunk for _, chunk in df.groupby(pd.Grouper(freq=freq_str))]
    elif type(chunk_size) == int:
        return np.array_split(df, chunk_size)
    else:
        raise TypeError()