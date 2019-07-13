
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


def split_timeseries(df, *indices):
    if not all([type(idx) == type(indices[0]) for idx in indices]):
        raise TypeError('indices: All indices must be of same type!')

    if type(indices[0]) == int:
        return np.split(df, list(indices))
    elif type(indices[0]) == float:
        return np.split(df, [int(index*len(df)) for index in indices])
    elif type(indices[0]) == datetime:
        return np.split(df, [df.index.get_loc(index) for index in indices])
    elif type(indices[0]) == str:
        return np.split(df, [df.index.get_loc(map_str_to_datetime(index)) for index in indices])
    else:
        raise TypeError('indices: indices must be int, float, datetime or str, but were ' + str(type(indices[0])) + ' instead!')