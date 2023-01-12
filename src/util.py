import os
import pickle
import shutil
import subprocess
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.api import CoperiaApi
from src.data import Audio, MyPatient


# Useful method
def save_obj(pickle_name, obj):
    with open(pickle_name, 'wb') as handle:
        pickle.dump(obj, handle, 0)


def load_obj(path_2_pkl: str):
    with open(path_2_pkl, 'rb') as pkl_file:
        return pickle.load(pkl_file)


def download_coperia_dataset_by_code(codes: list = None, path: str = 'data'):
    if codes is None:
        return []
    else:
        api = CoperiaApi(os.getcwd())

        dataset, total_samples = [], 0
        for code in codes:
            path_audios = os.path.join(path, f'audio_obs_{code}.pkl')
            if not os.path.exists(path_audios):
                data = api.get_observations_by_code(code)
                dataset.extend(data)

                total_samples += api.get_observations_total(code)
                save_obj(path_audios, data)
            else:
                data = load_obj(path_audios)
                dataset.extend(data)

        print(f"+=== {total_samples} observations downloaded. ===+")
        return dataset


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


def gender_distribution(metadata, path_store_figure: str = 'dataset/', audio_type: str = '/a/'):
    # Filtering the data
    df = metadata.copy()
    gender_labels = df['gender'].unique()[::-1]
    gender_cnt = []
    for i in range(len(gender_labels)):
        gender_cnt.append(len(df[(df['gender'] == gender_labels[i]) & (df['audio_type'] == audio_type)]))
        # & (df['audio_type']==audio_type)]))
    # Create a Figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Set colors
    clr_1 = 'tab:blue'
    clr_2 = 'tab:red'
    # Plot data
    ax.bar(2, gender_cnt[0], align='center', alpha=1, ecolor='black', capsize=5, hatch=r"\\", color=clr_1, width=.6)
    ax.bar(4, gender_cnt[1], align='center', alpha=1, ecolor='black', capsize=5, hatch="//", color=clr_2, width=.6)
    # Adding a label with the total above each bar
    for i, v in enumerate(gender_cnt):
        ax.text(2 * (i + 1) - .1, v + 3, str(v), color='black', fontweight='bold', fontsize=14)
    # Captions
    plt.title(f'COPERIA2022: Gender distribution in task {audio_type}.')
    plt.xticks([2, 4], ['MALE', 'FEMALE'], rotation=0)
    plt.ylabel('COUNT', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Extra details
    ax.set_xlim(1, 5)
    ax.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Save the figure
    if path_store_figure:
        path_store_figure = os.path.join(path_store_figure,
                                         f"COPERIA2022_metadata_gender{audio_type.replace('/', '-')[:-1]}.jpg")
        ax.figure.savefig(path_store_figure, bbox_inches='tight')
    # Plotting
    plt.show()


def duration_distribution(metadata, path_store_figure: str = 'dataset/', audio_type: str = '/a/'):
    # Filtering the data
    df = metadata.copy()
    duration = df[(df['audio_type'] == audio_type)]['duration'].astype(int)
    counter_d = Counter(duration)
    data = list(counter_d.values())
    name = sorted(list(counter_d.keys()))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(counter_d)), data, tick_label=name, align='center', alpha=1, ecolor='black', capsize=5,
           color='tab:blue', width=.6)
    for i, v in enumerate(data):
        ax.text(i - .1, v + .2, str(v), color='black', fontweight='bold')

    plt.title(f'COPERIA2022: Duration distribution in task {audio_type}.')
    plt.xlabel('Duration (s)', fontsize=14)
    plt.ylabel('COUNT', fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(0, max(data) + 2)
    ax.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if path_store_figure:
        path_store_figure = os.path.join(path_store_figure,
                                         f"COPERIA2022_metadata_duration{audio_type.replace('/', '-')[:-1]}.jpg")
    ax.figure.savefig(path_store_figure, bbox_inches='tight')
    plt.show()


def patients_age_distribution(metadata, path_store_figure: str = 'dataset/'):
    """

    :param metadata:
    :param path_store_figure:
    :return:
    """
    # Filtering the data
    df = metadata.copy()
    df = df[['age', 'patient_id', 'gender']].drop_duplicates()
    # Find labels
    age_labels = df['age'].unique()
    age_cnt_male = []
    age_cnt_female = []
    for i in range(len(age_labels)):
        if age_labels[i] == 'X':
            age_labels[i] = 0
        age_cnt_male.append(len(df[(df['age'] == age_labels[i]) & (df['gender'] == 'male')]))
        age_cnt_female.append(len(df[(df['age'] == age_labels[i]) & (df['gender'] == 'female')]))
    # Counting
    age_cnt_male = df[(df['gender'] == 'male')]['age'].values
    age_cnt_female = df[(df['gender'] == 'female')]['age'].values
    # Clustering
    age_grouped_male = []
    age_grouped_female = []
    age_labels = ['0-18', '18-30', '30-40', '40-50', '50-60', '60-70', '70-80']
    for i in age_labels:
        age_grouped_male.append(len(age_cnt_male[(age_cnt_male > (int(i.split('-')[0]) - 1)) &
                                                 (age_cnt_male < int(i.split('-')[1]))]))
        age_grouped_female.append(len(age_cnt_female[(age_cnt_female > (int(i.split('-')[0]) - 1)) &
                                                     (age_cnt_female < int(i.split('-')[1]))]))
    # Create a Figure
    fig, ax = plt.subplots(figsize=(7, 6))
    # Set colors
    clr_1 = 'tab:blue'
    clr_2 = 'tab:red'
    # Plot data
    ax.bar(np.arange(0, len(age_labels)), age_grouped_male, align='center', alpha=1, hatch="\\\\", ecolor='black',
           capsize=5, color=clr_1, width=.3, label=f'MALE = {sum(age_grouped_male)}')
    ax.bar(np.arange(0, len(age_labels)) + .3, age_grouped_female, align='center', alpha=1, hatch="\\\\",
           ecolor='black', capsize=5, color=clr_2, width=.3, label=f'FEMALE = {sum(age_grouped_female)}')
    # Adding a label with the total above each bar
    for i, v in enumerate(age_grouped_male):
        if v != 0:
            ax.text(i - .1, v + .2, str(v), color='black', fontweight='bold')

    for i, v in enumerate(age_grouped_female):
        if v != 0:
            ax.text(i + .2, v + .2, str(v), color='black', fontweight='bold')
    # Captionns
    ax.legend(frameon=False, loc='upper right', fontsize=14)
    plt.title("COPERIA2022: Patient's age & gender distribution", fontsize=14)
    plt.ylabel('COUNT', fontsize=14)
    plt.xlabel('AGE GROUP', fontsize=14)
    plt.xticks(np.arange(0, len(age_labels)), age_labels, rotation=0, fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_ylim(0, max(max(age_grouped_male), max(age_grouped_female)) + 1)
    ax.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if path_store_figure:
        path_store_figure = os.path.join(path_store_figure,
                                         f"COPERIA2022_metadata_age-gender_distrubution.jpg")
    ax.figure.savefig(path_store_figure, bbox_inches='tight')
    plt.show()


def patients_type_distribution(metadata, path_store_figure: str = 'dataset/'):
    """

    :param metadata:
    :param path_store_figure:
    :return:
    """
    # Filtering the data
    df = metadata.copy()
    labels = df['patient_type'].unique()
    df = df[['patient_id', 'patient_type']].drop_duplicates().groupby('patient_type').patient_type.size()

    # Create a Figure
    fig, ax = plt.subplots(figsize=(8, 6))
    # Set colors
    clr_1 = 'tab:blue'
    clr_2 = 'tab:red'
    # Plot data
    ax.bar(2, df[labels[0]], align='center', alpha=1, ecolor='black', capsize=5, hatch=r"\\", color=clr_1, width=.6)
    ax.bar(4, df[labels[1]], align='center', alpha=1, ecolor='black', capsize=5, hatch="//", color=clr_2, width=.6)
    # Adding a label with the total above each bar
    for i, v in enumerate(labels):
        ax.text(2 * (i + 1) - .1, df[v] + 3, str(df[v]), color='black', fontweight='bold', fontsize=14)
    # Captions
    plt.title(f'COPERIA2022: amount of patients in CONTROL and TEST group.')
    plt.xticks([2, 4], labels, rotation=0)
    plt.ylabel('COUNT', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Extra details
    ax.set_ylim(0, df.max() + 10)
    ax.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Save the figure
    if path_store_figure:
        path_store_figure = os.path.join(path_store_figure,
                                         f"COPERIA2022_metadata_patient_type.jpg")
        ax.figure.savefig(path_store_figure, bbox_inches='tight')
    # Plotting
    plt.show()


def patients_audio_distribution(metadata, path_store_figure: str = 'dataset/'):
    """

    :param metadata:
    :param path_store_figure:
    :return:
    """
    # Filtering the data
    df = metadata.copy()

    patients_male = {}
    patients_female = {}
    for index, row in df.iterrows():
        patient_id = row['patient_id']
        patient_gender = row['gender']
        if patient_id in patients_female.keys():
            patients_female[patient_id] += row['duration']
        elif patient_gender == 'female':
            patients_female[patient_id] = row['duration']
        elif patient_id in patients_male.keys():
            patients_male[patient_id] += row['duration']
        else:
            patients_male[patient_id] = row['duration']
    # Create a Figure
    fig, ax = plt.subplots(figsize=(12, 6))
    # Set colors
    clr_1 = 'tab:blue'
    clr_2 = 'tab:red'
    # Plot data
    male_values = sorted(patients_male.values())
    female_values = sorted(patients_female.values())

    male_ind = len(patients_male.values())
    female_ind = len(patients_female.values())
    ax.bar(range(male_ind), male_values, align='center', alpha=1, ecolor='black',
           capsize=5, color=clr_1, width=.6, hatch="\\\\", label=f'MALE = {male_ind}')
    ax.bar(range(male_ind, female_ind + male_ind), female_values, align='center', alpha=1, ecolor='black',
           capsize=5, color=clr_2, width=.6, hatch="//", label=f'FEMALE = {female_ind}')
    # Captionns
    ax.legend(frameon=False, loc='upper right', fontsize=14)
    plt.title("COPERIA2022: Patient's data duration distribution", fontsize=14)
    plt.ylabel('DURATION (s)', fontsize=14)
    plt.xlabel('Patients', fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_ylim(0, max(max(patients_male.values()), max(patients_female.values())) + 50)
    ax.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if path_store_figure:
        path_store_figure = os.path.join(path_store_figure,
                                         f"COPERIA2022_metadata_patient_duration.jpg")
    ax.figure.savefig(path_store_figure, bbox_inches='tight')
    plt.show()


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


def make_coperia_audios(audios_obs, patients_data, list_fs=None, path_save: str = 'dataset',
                        version: str = 'V1'):
    # Proces and save the audio data
    if list_fs is None:
        list_fs = [48000]

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
            coperia_audios.append(coperia_audio)

    return coperia_audios


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
