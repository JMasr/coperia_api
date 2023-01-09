import os.path
import shutil
import subprocess

import pandas as pd

from src.data import Audio, Patient
from src.util import *


def download_coperia_patients_by_observation(observations: list, path: str = 'data'):
    path = os.path.join(path, f'patients.pkl')
    if not os.path.exists(path):
        patients_ = {}
        for observation in observations:
            patient_id = observation.subject.reference.split('/')[-1]
            if patient_id not in patients_.keys():
                patient = Patient(observation)
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


def download_and_save_coperia_audios(path_save: str = 'dataset', version: str = 'V1', list_fs: list = [48000]):
    # Audio codes
    code_cough = '84435-7'
    code_vowel_a = '84728-5'
    # Download the Obervation and Patients
    path_save = f'{path_save}_{version}'
    os.makedirs(path_save, exist_ok=True)
    audios_obs = download_coperia_dataset_by_code([code_cough, code_vowel_a], path_save)
    patients_data = download_coperia_patients_by_observation(path=path_save, observations=audios_obs)
    # Proces and save the audio data
    coperia_audios = []
    for sample_rate in list_fs:

        path_to_dataset = os.path.join(path_save, f'{version}_{sample_rate}')
        if not os.path.exists(path_to_dataset):
            coperia_audio = process_coperia_audio(patients=patients_data, audio_observations=audios_obs,
                                                  sample_rate=sample_rate,
                                                  path_save=os.path.join(path_save, f'{version}_{sample_rate}'))

            coperia_audios.append(coperia_audio)
            pickle_name = os.path.join(path_save, f'coperia_audios_{sample_rate}.pkl')
            save_obj(pickle_name, coperia_audios)
        else:
            coperia_audio = load_obj(os.path.join(path_save, f'coperia_audios_{sample_rate}.pkl'))
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
        gender_distribution(df, path)
        gender_distribution(df, path, '/cough/')
        duration_distribution(df, path)
        duration_distribution(df, path, '/cough/')
        patients_age_distribution(df, path)
        patients_audio_distribution(df, path)
        if df['patient_type'].unique().size > 1:
            patients_type_distribution(df, path)


if __name__ == "__main__":
    root_path = 'dataset'
    data_version = 'V3'
    sample_rates = [48000, 16000]
    coperia = download_and_save_coperia_audios(path_save=root_path, version=data_version, list_fs=[48000, 16000])

    coperia_48kHz = coperia[0]
    coperia_16kHz = coperia[1]

    path_to_save = f'{root_path}_{data_version}/'
    coperia_metadata = pd.read_csv(os.path.join(path_to_save, 'coperia_metadata.csv'), decimal=',')
    coperia_metadata_control = coperia_metadata[coperia_metadata['patient_type'] == 'covid-control']
    coperia_metadata_persistente = coperia_metadata[coperia_metadata['patient_type'] == 'covid-persistente']

    plot_all_data([coperia_metadata, coperia_metadata_control, coperia_metadata_persistente],
                  [os.path.join(path_to_save, 'figures_all'),
                  os.path.join(path_to_save, 'figures_control'),
                  os.path.join(path_to_save, 'figures_persistente')])

    audios_48kHz_path = os.path.join(path_to_save, f'{data_version}_{sample_rates[0]}')
    specto_48kHz_path = os.path.join(path_to_save, f'{data_version}_{sample_rates[0]}_spectrogram')
    make_spectrogram(audios_48kHz_path, specto_48kHz_path)
    struct_spectrogram(coperia_metadata, specto_48kHz_path)
