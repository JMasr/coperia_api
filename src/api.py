import argparse
import os
import json
import base64
import pickle
import shutil
import tempfile
import requests
import warnings
import subprocess
from tqdm import tqdm
from collections import Counter
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from keycloak import KeycloakOpenID

from src.config import Config
from src.data import FeatureExtractor
from src.util import load_obj, save_obj


class Audio:
    def __init__(self, observation: Observation, patients: dict, contained_slot: int = 0,
                 r_fs: int = 44100, save_path: str = None):
        # Audio section
        self.audio_id: str = observation.contained[contained_slot].id
        self.duration = float(observation.contained[contained_slot].duration)
        self.type_code = observation.code.coding[0].code
        self.audio_moment: str = 'after' if observation.meta.tag[0].code == 'after' else 'before'

        self.data_base64 = observation.contained[contained_slot].content.data
        self._load_audio(r_fs, save_path)

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
    def get_long_covid(patient: Patient) -> bool:
        """
        Return the value of long COVID
        :param patient: Patient of the sample
        :return:
        """
        for tag in patient.meta.tag:
            if tag.code == "covid-control":
                return False
        return True

    @staticmethod
    def get_patient_type(patient: Patient) -> str:
        """
        Return the Patient type (covid-control or covid-persistente)
        :param patient: Patient of the sample
        :return:
        """
        code_list = [tag.code for tag in patient.meta.tag]
        if 'covid-control' in code_list and 'dicoperia' in code_list:
            return 'covid-control'
        elif 'covid-persistente' in code_list and 'dicoperia' in code_list:
            return 'covid-persistente'
        elif 'covid-control' in code_list and 'dicoperia' not in code_list:
            return 'unk-control'
        elif 'covid-persistente' in code_list and 'dicoperia' not in code_list:
            return 'coperia-rehab'
        else:
            return 'UNK'

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
        self.long_covid = self.get_long_covid(patient)
        self.patient_type = self.get_patient_type(patient)


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


# An Coperia API class
class CoperiaApi:

    def __init__(self, env_path: str = ''):
        self.fhir_config = Config(os.path.join(env_path, '.env.fhir'))
        self.keycloak_config = Config(os.path.join(env_path, '.env.keycloak'))
        self.url_server = f"{self.fhir_config.get('URL_FHIR')}/api/v4"

    @staticmethod
    def connection_is_success(response: requests.request):

        if response.status_code == 200:
            return True
        else:
            return False

    def basic_request(self, r_type: str, url: str, header=None, payload: dict = None,
                      files: dict = None) -> dict:
        """
        Constructs and sends a :class:`Request <Request>` with the parameters r_type, url, header, payload and file
        :param r_type: Verb of the API request (GET, PUT, POST, DELETE)
        :param url: Endpoint of the API request
        :param header: HTTP headers of the API request as a python dictionary
        :param payload: Body of the API request as a python dictionary
        :param files: Multipart encoding upload of the API request
        :return: A JSONDecoder with the data requested

        Usage::
            >> h = {'Authorization': 'Bearer ' + self.access_token, 'Accept': "multipart/form-data"}
            >> p = {'id': 'asdfasxcvasdf'}
            >> document = {'file': open(document_path, 'rb')}
            >> response = basic_request('PUT', "https://url_to_api_request", headers=h, data=p, files=document)
        """

        header = {} if header is None else header
        payload = {} if payload is None else payload
        files = {} if files is None else files

        r = requests.request(r_type, url, headers=header, data=payload, files=files)

        if self.connection_is_success(r):
            response_in_json = json.loads(r.text)
            return response_in_json
        else:
            raise ConnectionError

    def get_access_token(self) -> str:
        # credential for your keycloak instance
        username = self.keycloak_config.get('USER')
        password = self.keycloak_config.get('PASSWORD')
        server_url = self.keycloak_config.get('URL_KEYCLOAK')
        client_id = self.keycloak_config.get('CLIENT_ID')

        # Configure client
        keycloak_openid_ = KeycloakOpenID(server_url=server_url,
                                          client_id=client_id,
                                          realm_name="coperia")
        # Get Access Token With Code
        token = keycloak_openid_.token(username, password)
        return token.get('access_token')

    def get_patient_by_identifier(self, identifier: str = 'COPERIA-REHAB-00002'):
        """
        Get a patient using its identifier as a query
        :param identifier: identifier of the patient, usually it is a human-readable id
        :return: a fhir.resources.patient.Patient object
        """

        access_token = self.get_access_token()
        header = {'Authorization': 'Bearer ' + access_token}

        r_url = f"{self.url_server}/Patient?identifier={identifier}"
        response_json = self.basic_request('GET', r_url, header)
        return Patient.parse_obj(response_json)

    def get_patient(self, patient_id: str = '1803c4d57a5-23938f19-2e4d-445b-a4cc-cb6e78387e87'):
        """
        Get a patient using its id as a query
        :param patient_id: id of the patient
        :return: a fhir.resources.patient.Patient object
        """
        access_token = self.get_access_token()
        header = {'Authorization': 'Bearer ' + access_token}

        r_url = f"{self.url_server}/Patient/{patient_id}"
        response_json = self.basic_request('GET', r_url, header)
        return Patient.parse_obj(response_json)

    def get_observations_total(self, observation_code: str = "84728-5") -> int:
        """
        Get the amount of observation by code
        :param observation_code: Code of observation
        :return: the number of observation by code
        """
        access_token = self.get_access_token()
        header = {'Authorization': f'Bearer {access_token}'}

        r_url = f'https://api.coperia.es/fhir-server/api/v4/Observation?code={observation_code}'

        response: dict = self.basic_request('GET', r_url, header)
        return response.get('total')

    def get_observations_by_code(self, observation_code: str = "84728-5") -> list:
        """
        Get all the observations with the same code, using it as the query
        :param observation_code: code used as the search query
        :return: a list with the observation or an empty list if the code does not exist
        """
        access_token = self.get_access_token()
        header = {'Authorization': f'Bearer {access_token}'}

        observations_total = self.get_observations_total(observation_code)
        if observations_total > 0:
            r_url = f'https://api.coperia.es/fhir-server/api/v4/Observation?code={observation_code}' \
                    f'&_count={observations_total}'
            response: dict = self.basic_request('GET', r_url, header)
            raw_observations: list = response.get('entry')
            if raw_observations is None:
                raise ValueError('The Bundle of Observation does not have entries')
            return [Observation.parse_obj(raw.get('resource')) for raw in raw_observations]
        else:
            warnings.warn(ResourceWarning(f'No Observation with the code {observation_code}'))
            return []


def make_inference_files(root_path: str, output_path: str, metadata: pd.DataFrame):
    """
    Giving a pd.DataFrame with the audio dataset metadata, make a scp file for each group of patients
    :param root_path: root path of the data directory
    :param output_path: path where the scp files will be saved
    :param metadata: a list with all the audio samples as an Audio class
    """
    print("Making scp files...")
    os.makedirs(output_path, exist_ok=True)
    # Filtering data
    patient_control = metadata[metadata['patient_type'] == 'covid-control']
    patient_control_auidio_type_a = patient_control[patient_control['audio_type'] == '/a/']
    patient_control_auidio_type_cough = patient_control[patient_control['audio_type'] == '/cough/']

    patient_persistente = metadata[metadata['patient_type'] == 'covid-persistente']
    patient_persistente_auidio_type_a = patient_persistente[patient_persistente['audio_type'] == '/a/']
    patient_persistente_auidio_type_cough = patient_persistente[patient_persistente['audio_type'] == '/cough/']

    # Making scp files
    with open(os.path.join(output_path, 'scp_all'), 'w') as f:
        for row in patient_control.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')
        for row in patient_persistente.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')

    with open(os.path.join(output_path, 'reference_all'), 'w') as f:
        for row in patient_control.itertuples():
            f.write(f'{row.audio_id}\tn\n')
        for row in patient_persistente.itertuples():
            f.write(f'{row.audio_id}\tp\n')

    with open(os.path.join(output_path, 'scp_all_a'), 'w') as f:
        for row in patient_control_auidio_type_a.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')
        for row in patient_persistente_auidio_type_a.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')

    with open(os.path.join(output_path, 'reference_all_a'), 'w') as f:
        for row in patient_control_auidio_type_a.itertuples():
            f.write(f'{row.audio_id}\tn\n')
        for row in patient_persistente_auidio_type_a.itertuples():
            f.write(f'{row.audio_id}\tp\n')

    with open(os.path.join(output_path, 'scp_all_cough'), 'w') as f:
        for row in patient_control_auidio_type_cough.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')
        for row in patient_persistente_auidio_type_cough.itertuples():
            f.write(f'{row.audio_id}\t{os.path.join(root_path, row.audio_id)}.wav\n')

    with open(os.path.join(output_path, 'reference_all_cough'), 'w') as f:
        for row in patient_control_auidio_type_cough.itertuples():
            f.write(f'{row.audio_id}\tn\n')
        for row in patient_persistente_auidio_type_cough.itertuples():
            f.write(f'{row.audio_id}\tp\n')


def make_audios_spectrogram(root_path: str, metadata: pd.DataFrame):
    """
    Make a spectrogram of each audio in the dataset
    :param root_path: root path of the data directory
    :param metadata: a list with all the audio samples as an Audio class
    """

    def make_spectrogram(raw_audio_path: str, spectrogram_path: str):
        """
        Make a spectrogram from a raw audio file
        :param raw_audio_path: path to the raw audio file
        :param spectrogram_path: path to store the spectrogram
        """
        os.makedirs(spectrogram_path, exist_ok=True)
        spectrogram_done = os.listdir(spectrogram_path)

        for audio in tqdm(os.listdir(raw_audio_path)):
            audio_path = f"{raw_audio_path}/{audio}"
            png_path = audio_path.replace('.wav', '.png')

            png = audio.replace('.wav', '.png')
            if png not in spectrogram_done:
                subprocess.call(f'src/make_spectrogram.sh {audio_path}', shell=True)
                shutil.move(png_path, spectrogram_path)

    def struct_spectrogram(metadata_: pd.DataFrame, spectrogram_path: str):
        """
        Structure the spectrogram in a folder for each patient
        :param metadata_: dataframe with the metadata
        :param spectrogram_path: path to the spectrogram
        """
        df = metadata_[['patient_id', 'patient_type', 'audio_id', 'audio_type']].copy()
        spect_names = os.listdir(spectrogram_path)
        spect_names = [i for i in spect_names if i.endswith('.png')]

        for patient_type in df.patient_type.unique():
            os.makedirs(f'{spectrogram_path}/{patient_type}', exist_ok=True)
            for audio_task in df.audio_type.unique():
                os.makedirs(f'{spectrogram_path}/{patient_type}/{audio_task}', exist_ok=True)

        for spect_name in tqdm(spect_names):
            spect_path = os.path.join(spectrogram_path, spect_name)
            spect_name = spect_name.split('.')[0]

            spect_task = 'a' if df[df.eq(spect_name).any(1)].audio_type.eq('/a/').sum() else 'cough'
            spect_population = 'covid-persistente' if df[df.eq(spect_name).any(1)].patient_type.eq(
                'covid-persistente').sum() else 'covid-control'

            new_location = f'{spectrogram_path}/{spect_population}/{spect_task}/{spect_name}.png'
            if not os.path.exists(new_location):
                print(f"Moving {spect_name} to {new_location}")
                shutil.copy(spect_path, new_location)

    path_audio = os.path.join(root_path, f'wav_48000kHz')
    path_spectrogram = os.path.join(root_path, f'wav_48000kHz_spectrogram')
    print('Making spectrogram...')
    make_spectrogram(path_audio, path_spectrogram)
    print('Ordering spectrogram...')
    struct_spectrogram(metadata, path_spectrogram)


def make_metadata_plots(root_path: str, metadata: pd.DataFrame):
    """
    Plot and save a set of png files with information about the dataset
    :param root_path: root path of the data directory
    :param metadata: a list with all the audio samples as an Audio class
    """

    def plot_all_data(dfs: list, paths: list):
        """
        Plot all the data
        :param dfs: list of dataframes
        :param paths: list of paths to store the figures
        """
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

    def gender_distribution(metadata_, path_store_figure: str = 'dataset/', audio_type: str = '/a/'):
        """
        Plot the gender distribution of the dataset
        :param metadata_: dataframe with the metadata
        :param path_store_figure: path to store the figure
        :param audio_type: type of audio to filter the data
        """
        # Filtering the data
        df = metadata_.copy()
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

    def duration_distribution(metadata_, path_store_figure: str = 'dataset/', audio_type: str = '/a/'):
        """
        Plot the duration distribution of the dataset
        :param metadata_: dataframe with the metadata
        :param path_store_figure: path to store the figure
        :param audio_type: type of audio to filter the data
        """
        # Filtering the data
        df = metadata_.copy()
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

    def patients_age_distribution(metadata_, path_store_figure: str = 'dataset/'):
        """
        Plot the age distribution of the dataset
        :param metadata_: dataframe with the metadata
        :param path_store_figure: path to store the figure
        """
        # Filtering the data
        df = metadata_.copy()
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

    def patients_type_distribution(metadata_, path_store_figure: str = 'dataset/'):
        """
        Plot the type distribution of the dataset
        :param metadata_: dataframe with the metadata
        :param path_store_figure: path to store the figure
        """
        # Filtering the data
        df = metadata_.copy()
        labels = df['patient_type'].unique()
        df = df[['patient_id', 'patient_type']].drop_duplicates().groupby('patient_type').patient_type.size()

        # Create a Figure
        fig, ax = plt.subplots(figsize=(8, 6))
        # Set colors
        clr = ['tab:blue', 'tab:red', 'tab:green']
        deco = [r"\\", "//", r"\\"]
        # Plot data
        for ind in range(len(labels)):
            ax.bar(ind + 1, df[labels[ind]], align='center', alpha=1, ecolor='black',
                   capsize=5, hatch=deco[ind], color=clr[ind], width=.6)
        # Adding a label with the total above each bar
        for i, v in enumerate(labels):
            ax.text((i + 1) - .1, df[v] + 3, str(df[v]), color='black', fontweight='bold', fontsize=14)
        # Captions
        plt.title(f'COPERIA2022: amount of patients in CONTROL and TEST group.')
        plt.xticks(range(1, len(labels) + 1), labels, rotation=0)
        plt.ylabel('COUNT', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=14)
        # Extra details
        ax.set_xlim(0, len(labels) + 2)
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

    def patients_audio_distribution(metadata_, path_store_figure: str = 'dataset/'):
        """
        Plot the audio distribution of the dataset
        :param metadata_: dataframe with the metadata
        :param path_store_figure: path to store the figure
        """
        # Filtering the data
        df = metadata_.copy()

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

    print('Making metadata...')
    coperia_metadata_control = metadata[metadata['patient_type'] == 'covid-control']
    coperia_metadata_persistente = metadata[metadata['patient_type'] == 'covid-persistente']
    print('Making plots...')
    plot_all_data([metadata, coperia_metadata_control, coperia_metadata_persistente],
                  [os.path.join(root_path, 'figures_all'),
                   os.path.join(root_path, 'figures_control'),
                   os.path.join(root_path, 'figures_persistente')])


def make_dicoperia_metadata(save_path: str, metadata: pd.DataFrame, filters_: dict = None, remove_samples: dict = None):
    """
    Make a metadata file for the COPERIA dataset filtering some columns
    :param save_path: path to save the metadata file
    :param metadata: a list with all the audio samples in COPERIA as an Audio class
    :param filters_: a dictionary with the columns and values to keep
    :param remove_samples: a dictionary with the columns and values to remove
    :return: a pandas dataframe with the metadata of the DICOPERIA dataset
    """
    print('=== Filtering the metadata... ===')
    df = metadata.copy()

    if filters_ is None:
        filters_ = {'patient_type': ['covid-control', 'covid-persistente']}

    if remove_samples is None:
        remove_samples = {'audio_id': ['c15e54fc-5290-4652-a3f7-ff3b779bd980', '244b61cc-4fd7-4073-b0d8-7bacd42f6202'],
                          'patient_id': ['coperia-rehab']}

    for key, values in remove_samples.items():
        df = df[~df[key].isin(values)]

    for key, values in filters_.items():
        df = df[df[key].isin(values)]

    # df.replace(['covid-control', 'covid-persistente'], [0, 1], inplace=True)
    df.to_csv(os.path.join(save_path, 'dicoperia_metadata.csv'), index=False, decimal=',')
    print('Metadata saved in: {}'.format(save_path))
    print('=== Filtering DONE!! ===\n')
    return df


def make_audios_metadata(root_path: str, audios_dataset: list) -> pd.DataFrame:
    """
    Make a csv file with all the audio dataset metadata
    :param root_path: root path of the data directory
    :param audios_dataset: a list with all the audio samples as an Audio class
    :return: a pandas.DataFrame with the audio dataset metadata
    """
    audios_metadata = CoperiaMetadata(audios_dataset).metadata
    audios_metadata.to_csv(os.path.join(root_path, 'coperia_metadata.csv'), decimal=',', index=False)
    return audios_metadata


def make_audios_dataset(root_path: str, observations: list, patients: dict) -> list:
    """
    Make the audio samples and a dataset with instance of the class Audio
    :param root_path: root path of the data directory
    :param observations: list with the observation
    :param patients: a dictionary with all the patients {patient_id: MyPatient}
    :return: a list with all the audio samples as an Audio class
    """
    # Proces and save the audio data
    sample_rate = 48000
    coperia_audio = process_coperia_audio(patients=patients,
                                          sample_rate=sample_rate,
                                          audio_observations=observations,
                                          path_save=os.path.join(root_path, f'wav_{sample_rate}kHz'))

    data_path = os.path.join(root_path, f'coperia_audios_{sample_rate}.pkl')
    save_obj(data_path, coperia_audio)
    return coperia_audio


def make_coperia_audios(audios_obs, patients_data, list_fs=None, path_save: str = 'dataset',
                        version: str = 'V1') -> list:
    """
    Make the COPERIA dataset from the raw audios
    :param audios_obs: list of audios to use
    :param patients_data: dataframe with the metadata
    :param list_fs: list of sampling frequencies
    :param path_save: path to store the dataset
    :param version: version of the dataset
    """
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


def process_coperia_audio(patients: dict, audio_observations: list = None, sample_rate: int = 48000,
                          path_save: str = None) -> list:
    """
    Process the COPERIA dataset
    :param patients: dictionary with the patient information
    :param audio_observations: list of audio observations
    :param sample_rate: sampling frequency of the audio
    :param path_save: path to store the dataset
    """
    if audio_observations is None:
        return []
    else:
        audios = []
        print(f"Processing {len(audio_observations)} audios")
        for obs in tqdm(audio_observations):
            number_of_audios = len(obs.contained)
            for i in range(number_of_audios):
                audio = Audio(observation=obs, patients=patients, contained_slot=i, r_fs=sample_rate,
                              save_path=path_save)
                audios.append(audio)
        return audios


def download_coperia_patients(root_path: str, observations: list) -> dict:
    """
    Download and store in a dictionary the patient's metadata given a list of observation
    :param observations: list with the observation
    :param root_path: root path of the data directory
    :return: a dictionary where the key is the patient's id and the value is an instance of the class MyPatient
    """
    path = os.path.join(root_path, f'patients.pkl')

    patients_dict = {}
    print('Downloading patients...')
    for observation in tqdm(observations):
        patient_id = observation.subject.reference.split('/')[-1]
        if patient_id not in patients_dict.keys():
            patient = MyPatient(observation)
            patients_dict[patient_id] = patient
    save_obj(path, patients_dict)
    return patients_dict


def download_coperia_observations(root_path: str, codes: list = None) -> list:
    """
    Download the observation of Coperia by the voice codes, save it as pickle files, and return it as a list
    :param root_path: root path of the data directory
    :param codes: list with the observation codes
    :return: a list with two elements, each with the observation of one Coperia voice code
    """
    if codes is None:
        codes = ['84435-7', '84728-5']

    dataset = []
    api = CoperiaApi(os.getcwd())

    for code in codes:
        data = api.get_observations_by_code(code)
        dataset.extend(data)

        print(f"+=== Downloaded {len(data)} observations with code {code}. ===+")
        path_audios = os.path.join(root_path, f'audio_obs_{code}.pkl')
        save_obj(path_audios, data)
    return dataset


def download_and_save_coperia_data(path_save: str = 'dataset', version: str = 'V1'):
    """
    Download and save the COPERIA dataset
    :param path_save: path to store the dataset
    :param version: version of the dataset
    """
    # Audio codes
    code_cough = '84435-7'
    code_vowel_a = '84728-5'
    # Download the Obervation and Patients
    path_save = f'{path_save}_{version}'
    os.makedirs(path_save, exist_ok=True)
    audios_obs = download_coperia_dataset_by_code([code_cough, code_vowel_a], path_save)
    patients_data = download_coperia_patients_by_observation(path=path_save, observations=audios_obs)
    return audios_obs, patients_data


def download_coperia_dataset_by_code(codes: list = None, path: str = 'data') -> list:
    """
    Download the COPERIA dataset from the FHIR server
    :param codes: list of codes to filter the observations
    :param path: path to store the dataset
    """
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


def download_coperia_patients_by_observation(observations: list, path: str = 'data') -> dict:
    """
    Download the COPERIA dataset from the observation codes
    :param observations: list of observations to download
    :param path: path to store the dataset
    """
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


def check_4_new_data(path_data: str, codes: list = None):
    """
    Check if there is new data in the Coperia server
    :param path_data: path of the data directory
    :param codes: list with the observation codes
    """
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


def update_data(root_path: str = 'dataset_V4') -> bool:
    """
    Check for new data in the Coperia Cloud and update the local files of the dataset
    :param root_path: root path of the data directory
    """

    if check_4_new_data(root_path):
        print("There are new data.")
        observations = download_coperia_observations(root_path)
        patients = download_coperia_patients(root_path, observations)
        audio_dataset = make_audios_dataset(root_path, observations, patients)
        audio_metadata = make_audios_metadata(root_path, audio_dataset)
        dicoperia_metadata = make_dicoperia_metadata(root_path, audio_metadata)
        make_metadata_plots(root_path, dicoperia_metadata)
        make_audios_spectrogram(root_path, dicoperia_metadata)
        make_inference_files(os.path.join(root_path, 'wav_48000kHz'), 'dataset/inference_files', dicoperia_metadata)
        print("Dataset update!")
        return True
    else:
        print("There isn't new data.")
        return False


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()
    # Set a directory to save the data
    parser.add_argument('--data_path', '-o', default='dataset', type=str)
    args = parser.parse_args()
    # Check for new data
    update_data(args.data_path)
