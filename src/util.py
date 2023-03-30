import json
import os
import pickle


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


def load_config_from_json(path: str):
    """
    Load a json file as a dictionary. Useful to load the configuration of the experiments
    :param path: path to the json file
    :return: dictionary with the configuration
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config_as_json(config: dict, path: str):
    """
    Save a dictionary as a json file. Useful to save the configuration of the experiments
    :param config: dictionary with the configuration
    :param path: path to save the json file
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
