import time
import datetime

import numpy as np


"""
:dates: np.ndarray containing dates in the given format
:format: str format string like "%d-%m-%Y"
"""
def date_to_stamp(dates, format="%d-%m-%Y"):
    X = []
    for date in dates:
        X.append(time.mktime(datetime.datetime.strptime(date, format).timetuple()))
    return np.array(X)


"""
:stamps: np.ndarray containing unix timestamps
:format: str format string like "%d-%m-%Y"
"""
def stamp_to_date(stamps, format="%d-%m-%Y"):
    X = []
    for stamp in stamps:
        X.append(datetime.datetime.utcfromtimestamp(int(stamp)).strftime(format))
    return np.array(X)
