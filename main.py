import os.path
import shutil
import subprocess

import pandas as pd

from src.data import Audio, MyPatient, CoperiaMetadata
from src.util import *


def download_coperia_patients_by_observation(observations: list, path: str = 'data'):
    path = os.path.join(path, f'patients.pkl')
    if not os.path.exists(path):
        patients_ = {}
        for observation in observations:
            patient_id = observation.subject.reference.split('/')[-1]
            if patient_id not in patients_.keys():
                patient = MyPatient(observation)
                patients_[patient_id] = patient
        save_obj(path, patients_)
        return patients_
    else:
        return load_obj(path)


def process_coperia_audio(patients: dict, audio_observations: list = None, sample_rate: int = 48000,
                          path_save: str = None):
    if audio_observations is None:
        return []
    else:
        audios = []
        for obs in audio_observations:
            number_of_audios = len(obs.contained)
            for i in range(number_of_audios):
                audio = Audio(observation=obs, patients=patients, contained_slot=i, r_fs=sample_rate,
                              save_path=path_save)
                audios.append(audio)
        return audios


def download_and_save_coperia_data(path_save: str = 'dataset', version: str = 'V1'):
    # Audio codes
    code_cough = '84435-7'
    code_vowel_a = '84728-5'
    # Download the Obervation and Patients
    path_save = f'{path_save}_{version}'
    os.makedirs(path_save, exist_ok=True)
    audios_obs = download_coperia_dataset_by_code([code_cough, code_vowel_a], path_save)
    patients_data = download_coperia_patients_by_observation(path=path_save, observations=audios_obs)
    return audios_obs, patients_data


def make_coperia_audios(audios_obs, patients_data, list_fs: list = [48000], path_save: str = 'dataset',
                        version: str = 'V1'):
    # Proces and save the audio data
    coperia_audios = []
    for sample_rate in list_fs:
        data_path = os.path.join(path_save, f'coperia_audios_{sample_rate}.pkl')
        if not os.path.exists(data_path):
            coperia_audio = process_coperia_audio(patients=patients_data,
                                                  sample_rate=sample_rate,
                                                  audio_observations=audios_obs,
                                                  path_save=os.path.join(path_save, f'{version}_{sample_rate}'))

            pickle_name = os.path.join(path_save, f'coperia_audios_{sample_rate}.pkl')
            save_obj(pickle_name, coperia_audio)
            coperia_audios.append(coperia_audio)
        else:
            coperia_audio = load_obj(data_path)
            coperia_audios.extend(coperia_audio)

    return coperia_audios


def make_spectrogram(raw_audio_path: str, spectrogram_path: str):
    os.makedirs(spectrogram_path, exist_ok=True)

    for audio in os.listdir(raw_audio_path):
        audio_path = f"{raw_audio_path}/{audio}"
        subprocess.call(f'src/make_spectrogram.sh {audio_path}', shell=True)

        png_path = audio_path.replace('.wav', '.png')
        shutil.move(png_path, spectrogram_path)


def struct_spectrogram(metadata_: pd.DataFrame, spectrogram_path: str):
    df = metadata_[['patient_id', 'patient_type', 'audio_id', 'audio_type']].copy()
    spect_names = os.listdir(spectrogram_path)

    for patient_type in df.patient_type.unique():
        os.makedirs(f'{spectrogram_path}/{patient_type}', exist_ok=True)
        for audio_task in df.audio_type.unique():
            os.makedirs(f'{spectrogram_path}/{patient_type}/{audio_task}', exist_ok=True)

    for spect_name in spect_names:
        spect_path = os.path.join(spectrogram_path, spect_name)
        spect_name = spect_name.split('.')[0]

        spect_task = 'a' if df[df.eq(spect_name).any(1)].audio_type.eq('/a/').sum() else 'cough'
        spect_population = 'covid-control' if df[df.eq(spect_name).any(1)].patient_type.eq(
            'covid-control').sum() else 'covid-persistente'

        shutil.copy(spect_path, f'{spectrogram_path}/{spect_population}/{spect_task}/{spect_name}.png')


def plot_all_data(dfs: list, paths: list):
    for df, path in zip(dfs, paths):

        if not os.path.exists(path):
            os.makedirs(path)

        gender_distribution(df, path)
        gender_distribution(df, path, '/cough/')
        duration_distribution(df, path)
        duration_distribution(df, path, '/cough/')
        patients_age_distribution(df, path)
        patients_audio_distribution(df, path)
        if df['patient_type'].unique().size > 1:
            patients_type_distribution(df, path)


def check_4_new_data(path_data: str, codes: list = None):
    if codes is None:
        codes = ['84435-7', '84728-5']

    for code in codes:
        obs_path = os.path.join(path_data, f'audio_obs_{code}.pkl')
        if os.path.exists(obs_path):
            observations = load_obj(obs_path)
        else:
            observations = []

        obs_in_disk = len(observations)
        obs_in_cloud = CoperiaApi(os.getcwd()).get_observations_total(code)

        if obs_in_disk != obs_in_cloud:
            return True

    return False


def main_pipeline(root_path: str, data_version: int, sample_rates: list, codes: list):
    path_to_save = f'{root_path}_V{data_version}/'
    path_to_metadata = os.path.join(path_to_save, 'coperia_metadata.csv')

    print('Checking for new data...')
    if check_4_new_data(path_to_save, codes):
        print('Downloading new data...')
        data_version += 1
        path_to_save = f'{root_path}_V{data_version}/'
        path_to_metadata = os.path.join(path_to_save, 'coperia_metadata.csv')

        audios_obs, patients_data = download_and_save_coperia_data(path_save=root_path, version=f'V{data_version}')
        coperia = make_coperia_audios(audios_obs, patients_data, sample_rates, path_to_save, f'V{data_version}')

        coperia_metadata = CoperiaMetadata(coperia[0]).metadata
        coperia_metadata.to_csv(path_to_metadata, decimal=',')
    else:
        print('No new data found...')
        print('Loading data from disk...')
        coperia_metadata = pd.read_csv(path_to_metadata, decimal=',')

    print('Making metadata...')
    coperia_metadata_control = coperia_metadata[coperia_metadata['patient_type'] == 'covid-control']
    coperia_metadata_persistente = coperia_metadata[coperia_metadata['patient_type'] == 'covid-persistente']
    print('Making plots...')
    plot_all_data([coperia_metadata, coperia_metadata_control, coperia_metadata_persistente],
                  [os.path.join(path_to_save, 'figures_all'),
                   os.path.join(path_to_save, 'figures_control'),
                   os.path.join(path_to_save, 'figures_persistente')])
    print('Making spectrograms...')
    audios_48kHz_path = os.path.join(path_to_save, f'V{data_version}_{sample_rates[0]}')
    specto_48kHz_path = os.path.join(path_to_save, f'V{data_version}_{sample_rates[0]}_spectrogram')
    make_spectrogram(audios_48kHz_path, specto_48kHz_path)
    struct_spectrogram(coperia_metadata, specto_48kHz_path)


if __name__ == "__main__":
    main_pipeline('dataset', 3, [48000], ['84435-7', '84728-5'])
