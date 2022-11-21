import os
import pickle

import constants


def save_pickle(obj, rel_path):
    abs_path = os.path.join(constants.BASE_PATH, rel_path)
    with open(abs_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(rel_path):
    abs_path = os.path.join(constants.BASE_PATH, rel_path)
    with open(abs_path, 'rb') as handle:
        return pickle.load(handle)
