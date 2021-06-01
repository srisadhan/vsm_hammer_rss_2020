import yaml
import deepdish as dd
from pathlib import Path
from contextlib import contextmanager
import pickle
from dateutil import parser
import sys
import numpy as np 

class skip(object):
    """A decorator to skip function execution.

    Parameters
    ----------
    f : function
        Any function whose execution need to be skipped.

    Attributes
    ----------
    f

    """

    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        print('skipping : ' + self.f.__name__)


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """

    @contextmanager
    def check_active():
        deactivated = ['skip']
        if flag in deactivated:
            print('Skipping the block: ' + f)
            raise SkipWith()
        else:
            print('Running the block: ' + f)
            yield

    try:
        yield check_active
    except SkipWith:
        pass


def save_dataset(path, dataset, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        hdf5 dataset.
    save : Bool

    """
    if save:
        dd.io.save(path, dataset)

    return None


def save_dataframe(path, dataframe, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save

    save : Bool

    """
    if save:
        with open(path, 'wb') as f:
            pickle.dump(dataframe, f, pickle.HIGHEST_PROTOCOL)

    return None


def read_dataframe(path):
    """Save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataframe : dict
        dictionary of pandas dataframe to save


    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def save_model_log(info, save_path):
    with open(save_path + '/' + info['model_name'] + '.pkl', 'wb') as f:
        pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

    return None

# parse the millisecs by matching the length of the total digits to 6 
def parse_millisec(time_split):
    return time_split.astype(float) / 1e6
            
        
def parse_time(time_series):    
    # extract the microseconds and append it using a '.'
    time_split = time_series.str.rsplit(pat=':', n=1, expand=True) 
    milli_secs = parse_millisec(time_split[1])
   
    time_str   = time_split[0] 

    time_obj   = []
    for count, val in enumerate(time_str):
        time_secs = (parser.parse(val) - parser.parse(time_str[0])).total_seconds() + (milli_secs[count] - milli_secs[0])
        time_obj.append(time_secs)
    
    return np.array(time_obj)
             