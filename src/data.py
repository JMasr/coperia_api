import base64
import os
import pickle
import subprocess
import tempfile
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchaudio
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient

from src.api import CoperiaApi
from src.util import FeatureExtractor

# Set the Seaborn style
sns.set_style("white")
sns.set_style("ticks")


class Audio:
    def __init__(self, observation: Observation, patients: dict, contained_slot: int = 0, r_fs: int = 16000, save_path: str = None):
        # Audio section
        self.audio_id = observation.contained[contained_slot].id
        self.duration = float(observation.contained[contained_slot].duration)
        self.type_code = observation.code.coding[0].code
        self.audio_moment = 'after' if observation.meta.tag[0].code == 'after' else 'before'

        self.data_base64 = observation.contained[contained_slot].content.data
        self.wave_form, self.sample_rate = self._load_audio(r_fs, save_path)

        patient_id = observation.subject.reference.split('/')[-1]
        self.patient = patients[patient_id]
        self.observation = observation

    def __str__(self):
        return f'ID: {self.id}\n' \
               f'Type: {self.type} \n' \
               f'Duration: {self.duration}\n' \
               f'Extension: {self.extension}'

    def __len__(self):
        return self.duration

    @staticmethod
    def _compute_SAD(sig, fs, threshold=0.0001, sad_start_end_sil_length=100, sad_margin_length=50):
        """ Compute threshold based sound activity """

        if sig.shape[0] > 1:
            sig = sig.mean(dim=0).unsqueeze(0)
        sig = sig / torch.max(torch.abs(sig))
        sig = sig / torch.max(torch.abs(sig))

        # Leading/Trailing margin
        sad_start_end_sil_length = int(sad_start_end_sil_length * 1e-3 * fs)
        # Margin around active samples
        sad_margin_length = int(sad_margin_length * 1e-3 * fs)

        sample_activity = np.zeros(sig.shape)
        sample_activity[np.power(sig, 2) > threshold] = 1
        sad = np.zeros(sig.shape)
        for i in range(sample_activity.shape[1]):
            if sample_activity[0, i] == 1: sad[0, i - sad_margin_length:i + sad_margin_length] = 1
        sad[0, 0:sad_start_end_sil_length] = 0
        sad[0, -sad_start_end_sil_length:] = 0
        return sad

    @staticmethod
    def resample_audio(audio, sample_rate, resample_rate):
        return torchaudio.functional.resample(audio, orig_freq=sample_rate,
                                              new_freq=resample_rate), resample_rate

    def _load_audio(self, resample_f: int, save_path: str):
        decode64_data = base64.b64decode(self.data_base64)

        wav_file = tempfile.NamedTemporaryFile(suffix='.wav')
        wav_file.write(decode64_data)

        if save_path is None:
            save_path = tempfile.TemporaryFile(mode='wb')
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = f'{os.path.join(os.getcwd(), save_path, self.audio_id)}.wav'

        subprocess.run(f'ffmpeg -y -i {wav_file.name} -acodec pcm_s16le -ar {resample_f} -ac 1 {save_path}', shell=True)

        s, fs = torchaudio.load(save_path)
        sad = self._compute_SAD(s, fs)
        s = s[np.where(sad == 1)]
        return s, fs

    def save(self, pickle_path: str = None):
        if pickle_path is None:
            os.makedirs('dataset/default', exist_ok=True)
            pickle_path = os.path.join('dataset/default', 'default_coperia_audio.pkl')

        with open(pickle_path, 'wb') as handle:
            pickle.dump(self, handle, 0)

    def extract_metadata(self) -> pd.DataFrame:
        """
        Extract the relevant metadata (duration, age, gender, audio type,and patient type) from a list of Audios
        :return: a pd.DataFrame with the metadata
        """
        metadata = {'patient_id': self.patient.id,
                    'patient_type': self.patient.patient_type,
                    'age': self.patient.age,
                    'gender': self.patient.gender,
                    'audio_id': self.audio_id,
                    'covid': self.patient.covid,
                    'long_covid': self.patient.long_covid,
                    'audio_type': '/cough/' if self.type_code == '84435-7' else '/a/',
                    'audio_moment': self.audio_moment,
                    'audio_code': self.type_code,
                    'sample_rate': self.sample_rate,
                    'duration': self.duration,
                    }
        return pd.DataFrame([metadata])


class MyPatient:
    def __init__(self, observation: Observation):

        self.id: str = None
        self.age: int = None
        self.gender: str = None
        self.patient_type: str = None

        self.covid: bool = None
        self.long_covid: bool = None

        self._set_info_from_observation(observation)

    @staticmethod
    def get_id(patient: Patient) -> str:
        """
        Return the patient's id.
        :return: patient's id.
        """
        return patient.identifier[0].value

    @staticmethod
    def get_age(patient: Patient) -> int:
        """
        Return the patient's age.
        :return: patient's age.
        """
        if isinstance(patient.birthDate, str):
            born = int(patient.birthDate)
            return datetime.utcnow().year - born
        else:
            born = datetime(patient.birthDate.year, patient.birthDate.month, patient.birthDate.day)
            return (datetime.utcnow() - born).days // 365

    @staticmethod
    def get_gender(patient: Patient) -> str:
        """
        Return the patient's gender.
        :return: patient's gender.
        """
        return patient.gender

    @staticmethod
    def get_long_covid(observation: Observation):
        """
        Return the value of long COVID
        :param observation: Observation of sample
        :return:
        """
        return observation.meta.tag[-1].code != 'covid-control'

    @staticmethod
    def get_patient_type(observation: Observation):
        """
        Return the Patient type (covid-control or covid-persistente)
        :param observation: Observation of sample
        :return:
        """
        return observation.meta.tag[-1].code

    def put_assign_covid(self, diagnosis: bool):

        self.covid = diagnosis

    def put_long_covid(self, diagnosis: bool):
        self.long_covid = diagnosis

    def _set_info_from_observation(self, observation: Observation):
        """
        Extract all the information from an Observation to populate the Patient data
        :param observation: Observation of sample
        :return:
        """
        patient_id = observation.subject.reference.split('/')[-1]
        corilga_api = CoperiaApi()
        patient = corilga_api.get_patient(patient_id)

        self.id = patient_id
        self.age = self.get_age(patient)
        self.gender = self.get_gender(patient)
        self.long_covid = self.get_long_covid(observation)
        self.patient_type = self.get_patient_type(observation)


class CoperiaMetadata:
    def __init__(self, data):
        if isinstance(data, list):
            self.metadata: pd.DataFrame = self._populate_dataset(data)
        elif isinstance(data, pd.DataFrame):
            self.metadata = data

    @staticmethod
    def _populate_dataset(audios: list) -> pd.DataFrame:
        """
        Populate a metadata dataset from a list of Audios
        :param audios: A list with Audio objects
        :return: a pd.Dataframe with all the metadata.
        """
        metadata = pd.DataFrame()
        for inx, audio in enumerate(audios):
            audio_metadata = audio.extract_metadata()
            metadata = pd.concat([metadata, audio_metadata], ignore_index=True)
        return metadata

    def plot_all(self, path_save_img: str):
        self._gender_distribution(self.metadata, path_save_img)
        self._gender_distribution(self.metadata, path_save_img, '/cough/')
        self._duration_distribution(self.metadata, path_save_img)
        self._duration_distribution(self.metadata, path_save_img, '/cough/')
        self._patients_age_distribution(self.metadata, path_save_img)
        self._patients_type_distribution(self.metadata, path_save_img)
        self._patients_audio_distribution(self.metadata, path_save_img)

    @staticmethod
    def _gender_distribution(metadata, path_store_figure: str = 'dataset/', audio_type: str = '/a/'):
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

    @staticmethod
    def _duration_distribution(metadata, path_store_figure: str = 'dataset/', audio_type: str = '/a/'):
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

    @staticmethod
    def _patients_age_distribution(metadata, path_store_figure: str = 'dataset/'):
        """

        :param path_store_figure:
        :param audio_type:
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

    @staticmethod
    def _patients_type_distribution(metadata, path_store_figure: str = 'dataset/'):
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

    @staticmethod
    def _patients_audio_distribution(metadata, path_store_figure: str = 'dataset/'):
        """

        :param path_store_figure:
        :param audio_type:
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
        ax.bar(range(male_ind), male_values,  align='center', alpha=1, ecolor='black',
               capsize=5, color=clr_1, width=.6, hatch="\\\\", label=f'MALE = {male_ind}')
        ax.bar(range(male_ind, female_ind+male_ind), female_values, align='center', alpha=1, ecolor='black',
               capsize=5, color=clr_2, width=.6, hatch="//", label=f'FEMALE = {female_ind}')
        # Captionns
        ax.legend(frameon=False, loc='upper right', fontsize=14)
        plt.title("COPERIA2022: Patient's data duration distribution", fontsize=14)
        plt.ylabel('DURATION (s)', fontsize=14)
        plt.xlabel('Patients', fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_ylim(0, max(max(patients_male.values()), max(patients_female.values())) + 10)
        ax.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if path_store_figure:
            path_store_figure = os.path.join(path_store_figure,
                                             f"COPERIA2022_metadata_patient_duration.jpg")
        ax.figure.savefig(path_store_figure, bbox_inches='tight')
        plt.show()

    def to_csv(self, path):
        """

        :param path:
        :return:
        """
        self.metadata.to_csv(path, decimal=',')


class CoperiaDataset(torch.utils.data.Dataset):
    def __init__(self, observation: list, covid_or_long_covid: bool = True, feat_type: str = None):
        self.audios: list = self.get_audios(observation)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.egs: list = []
        self.generate_examples(covid_or_long_covid, feat_type)

    @staticmethod
    def get_audios(observations):
        audios = []
        for obs in observations:
            audio = Audio(obs)
            audios.append(audio)
        return audios

    def generate_examples(self, select_type_label: bool = True, feat_type='mfcc'):
        for audio in self.audios:
            signal = audio.resample_wave_form
            sample_rate = audio.resample_rate
            label = audio.patient.covid if select_type_label else audio.patient.long_covid

            feat_extractor = FeatureExtractor(feat_type)
            F = feat_extractor.do_feature_extraction(signal, sample_rate)

            self.egs.append((F.to(self.device), torch.FloatTensor([label]).to(self.device)))

    def __len__(self):
        return len(self.egs)
