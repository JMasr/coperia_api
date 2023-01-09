import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from spafe.features.bfcc import bfcc
from spafe.features.cqcc import cqcc
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc
from spafe.features.lpc import lpcc
from spafe.features.mfcc import mfcc, imfcc
from spafe.features.msrcc import msrcc
from spafe.features.ngcc import ngcc
from spafe.features.pncc import pncc
from spafe.features.psrcc import psrcc
from spafe.features.rplp import plp, rplp

from src.api import CoperiaApi
from src.config import Config


class FeatureExtractor:
    """ Class for feature extraction
    args: input arguments dictionary
    Mandatory arguments: resampling_rate, feature_type, window_size, hop_length
    For MFCC: f_max, n_mels, n_mfcc
    For MelSpec/logMelSpec: f_max, n_mels
    Optional arguments: compute_deltas, compute_delta_deltas
    """

    def __init__(self, feature_type: str = None):
        self.conf = Config('.env.feats')
        self.feature_transformers = {'mfcc': mfcc,
                                     'imfcc': imfcc,
                                     'bfcc': bfcc,
                                     'cqcc': cqcc,
                                     'gfcc': gfcc,
                                     'lfcc': lfcc,
                                     'lpc': lpc,
                                     'lpcc': lpcc,
                                     'msrcc': msrcc,
                                     'ngcc': ngcc,
                                     'pncc': pncc,
                                     'psrcc': psrcc,
                                     'plp': plp,
                                     'rplp': rplp}

        if feature_type is None:
            self.feat_type = self.conf.get_key('feature_type')

    def do_feature_extraction(self, s: torch.Tensor, fs: int):
        """ Feature preparation
        Steps:
        1. Apply feature extraction to waveform
        2. Convert amplitude to dB if required
        3. Append delta and delta-delta features
        """
        if self.feat_type.lower() in self.feature_transformers:
            # Spafe feature selected
            F = self.feature_transformers[self.feat_type](s, fs,
                                                          num_ceps=int(self.config.get('num_ceps')),
                                                          low_freq=int(self.config.get('low_freq')),
                                                          high_freq=int(fs / 2),
                                                          normalize=self.config.get('normalize'),
                                                          pre_emph=self.config.get('pre_emph'),
                                                          pre_emph_coeff=float(self.config.get('pre_emph_coeff')),
                                                          win_len=float(self.config.get('win_len')),
                                                          win_hop=float(self.config.get('win_hop')),
                                                          win_type=self.config.get('win_type'),
                                                          nfilts=int(self.config.get('nfilts')),
                                                          nfft=int(self.config.get('nfft')),
                                                          lifter=float(self.config.get('lifter')),
                                                          use_energy=self.config.get('use_energy') == 'True')
            F = np.nan_to_num(F)
            F = torch.from_numpy(F).T

            if self.conf('compute_deltas') == 'True':
                FD = torchaudio.functional.compute_deltas(F)
                F = torch.cat((F, FD), dim=0)

            if self.conf('compute_delta_deltas') == 'True':
                FDD = torchaudio.functional.compute_deltas(FD)
                F = torch.cat((F, FDD), dim=0)

            return F.T

        else:
            raise ValueError('Feature type not implemented')


# Useful method
def save_obj(pickle_name, obj):
    with open(pickle_name, 'wb') as handle:
        pickle.dump(obj, handle, 0)


def load_obj(path_2_pkl: str):
    with open(path_2_pkl, 'rb') as pkl_file:
        return pickle.load(pkl_file)


def download_coperia_dataset_by_code(codes: list = [], path: str = 'data'):
    api = CoperiaApi(os.getcwd())

    dataset, total_samples = [], 0
    for code in codes:
        path_audios = os.path.join(path, f'audio_obs_{code}.pkl')
        if not os.path.exists(path_audios):
            dataset.extend(api.get_observations_by_code(code))
            total_samples += api.get_observations_total(code)
            save_obj(path_audios, dataset)
        else:
            data = load_obj(path_audios)
            dataset.extend(data)
    print(f"+=== {total_samples} samples downloaded. ===+")
    return dataset


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
        age_grouped_male.append(len(age_cnt_male[(age_cnt_male > (int(i.split('-')[0]) - 1)) & \
                                                 (age_cnt_male < int(i.split('-')[1]))]))
        age_grouped_female.append(len(age_cnt_female[(age_cnt_female > (int(i.split('-')[0]) - 1)) & \
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
