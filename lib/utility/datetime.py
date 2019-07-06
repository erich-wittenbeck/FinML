
import re

from datetime import datetime

def map_str_to_datetime(date_string):

    regex_year = '^\d{4}$'
    regex_month = regex_year[:-1] + '-\d{2}$'
    regex_day = regex_month[:-1] + '-\d{2}$'
    regey_daytime = regex_day[:-1] + ' \d{2}:\d{2}:\d{2}$'

    if re.match(regex_year, date_string):
        return datetime.strptime(date_string, '%Y')
    elif re.match(regex_month, date_string):
        return datetime.strptime(date_string, '%Y-%m')
    elif re.match(regex_day, date_string):
        return datetime.strptime(date_string, '%Y-%m-%d')
    elif re.match(regey_daytime, date_string):
        return datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    else:
        raise ValueError('date_string is not formatted correctly!')