import os
import pickle

import src.data
from src.api import CoperiaApi
from src.data import CoperiaDataset


def download_coperia_dataset(sample_rate: int = 48000, path_to_save: str = None):
    code_cough = '84435-7'
    code_vowel_a = '84728-5'

    api = CoperiaApi(os.getcwd())
    observations = (api.get_observations_by_code(code_vowel_a))
    observations.extend(api.get_observations_by_code(code_cough))

    audios = []
    for obs in observations:
        audio = src.data.Audio(obs, sample_rate, path_to_save)
        audios.append(audio)

    pickle_name = os.path.join(path_to_save, f'coperia_audio_{sample_rate}')
    with open(pickle_name, 'wb') as handle:
        pickle.dump(audios, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return audios


if __name__ == "__main__":
    standar_fs = 16000
    original_fs = 48000

    coperia_audios_16kHz = download_coperia_dataset(standar_fs, 'dataset/v0_16kHz')
    coperia_audios_48kHz = download_coperia_dataset(original_fs, 'dataset/v0_48kHz')
    print('Done!')
