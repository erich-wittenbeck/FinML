# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 11:38:35 2018

@author: User
"""

import csv
import os
import configparser

from datetime import datetime

# Writing

def write_to_csvfile(rows, filename, mode='w', newline='', delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL):
    csvfile = open(filename, mode, newline=newline)
    filewriter = csv.writer(csvfile, delimiter=delimiter, quotechar=quotechar, quoting=quoting)

    for row in rows:
        filewriter.writerow(row)

    csvfile.close()
    return

def write_to_txtfile(content, filename, mode='w'):
    txtfile = open(filename, mode)
    txtfile.write(content)
    txtfile.close()
    return

# Reading

def read_from_csvfile(filename, mode='r', newline='', delimiter=' ', quotechar='|', transpose=False, start=0, end=None,
                      conv_func= lambda row : [row[0], datetime.fromtimestamp(int(row[1]))] + [float(e) for e in row[2:]]):
    rows = []

    csvfile = open(filename, mode, newline=newline)
    filereader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)

    for row in filereader:
        if conv_func == None:
            rows.append(row)
        else:
            rows.append(conv_func(row))

    csvfile.close()
    rows = rows[start:] if end == None else rows[start:end]

    if transpose:
        return zip(*rows)
    else:
        return rows


def load_config(*keys, filename='config.ini', section=None):
    config = configparser.ConfigParser()
    CONF_DIR = os.path.dirname(os.path.abspath(__file__)) + '\\.config\\'
    config.read(CONF_DIR+filename)

    if section == None:
        return config
    elif len(keys) == 0:
        return config[section]
    else:
        return tuple([config[section][key] for key in keys])
