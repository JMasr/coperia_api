import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# Useful method
def save_obj(pickle_name: str, obj: object):
    """
    Save an object in a pickle file
    :param pickle_name: path of the pickle file
    :param obj: object to save
    """
    os.makedirs(os.path.dirname(pickle_name), exist_ok=True)
    with open(pickle_name, 'wb') as handle:
        pickle.dump(obj, handle, 0)


def load_obj(path_2_pkl: str) -> object:
    """
    Load an object from a pickle file
    :param path_2_pkl: path of the pickle file
    """
    with open(path_2_pkl, 'rb') as pkl_file:
        return pickle.load(pkl_file)
