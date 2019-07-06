
from datetime import datetime
from lib.utility.datetime import map_str_to_datetime

def slice_timeseries(df, start, end=None):
    df_slice = df

    indexer = start
    if type(start) == int:
        if end == None:
            df_slice = df_slice.iloc[indexer, :].to_frame().transpose()
        else:
            indexer = slice(indexer, end)
            df_slice = df_slice.iloc[indexer, :]
    elif type(start) == str:
        indexer = map_str_to_datetime(indexer)
        if end == None:
            df_slice = df_slice.loc[indexer, :].to_frame().transpose()
        else:
            indexer = slice(indexer, map_str_to_datetime(end))
            df_slice = df_slice.loc[indexer, :]
    elif type(start) == datetime:
        if end == None:
            df_slice = df_slice.loc[indexer, :].to_frame().transpose()
        else:
            indexer = slice(indexer, end)
            df_slice = df_slice.loc[indexer, :]

    return df_slice


def split_timeseries(df, *indices):
    if not all([type(idx) == type(indices[0]) for idx in indices]):
        raise TypeError('indices: All indices must be of same type!')

    start_ends = []

    timeline_start = df.index[0].to_pydatetime()
    timeline_end = df.index[-1].to_pydatetime()

    if type(indices[0]) == int:
        start_ends = [0] + list(indices) + [-1]
    if type(indices[0]) == str:
        start_ends = [timeline_start] + \
                     [map_str_to_datetime(idx) for idx in indices] + \
                     [timeline_end]
    if type(indices[0]) == datetime:
        start_ends = [timeline_start] + \
                     list(indices) + \
                     [timeline_end]

    start_ends = list(zip(start_ends, start_ends[1:]))

    return tuple([slice_timeseries(df, start, end) for start, end in start_ends if start != end])