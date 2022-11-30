import base64
import os
import subprocess
import tempfile
from datetime import datetime

import numpy as np
import soundfile
import torch
import torchaudio

from fhir.resources.observation import Observation
from fhir.resources.patient import Patient

from src.util import FeatureExtractor
from src.api import CoperiaApi


class Audio:
    def __init__(self, observation: Observation, r_fs: int = 16000, save_path: str = None):
        # Audio section
        self.audio_id = observation.id
        self.duration = float(observation.contained[0].duration)
        self.type_code = observation.code.coding[0].code

        self.data_base64 = observation.contained[0].content.data
        self.wave_form, self.sample_rate = self._load_audio(r_fs, save_path)

        self.patient = Patient(observation)

    def __str__(self):
        return f'ID: {self.id}\n' \
               f'Type: {self.type} \n' \
               f'Duration: {self.duration}\n' \
               f'Extension: {self.extension}'

    def __len__(self):
        return self.duration

    @staticmethod
    def compute_SAD(sig, fs, threshold=0.0001, sad_start_end_sil_length=100, sad_margin_length=50):
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

    def _load_audio(self, resample_f: int, save_path: str):
        decode64_data = base64.b64decode(self.data_base64)

        wav_file = tempfile.NamedTemporaryFile(suffix='.wav')
        wav_file.write(decode64_data)

        if save_path is None:
            save_path = tempfile.TemporaryFile(mode='wb')
        else:
            os.makedirs(save_path, exist_ok=True)
            save_path = f'{os.path.join(os.getcwd(),save_path, self.audio_id)}.wav'

        subprocess.run(f'ffmpeg -y -i {wav_file.name} -acodec pcm_s16le -ar {resample_f} -ac 1 {save_path}', shell=True)

        s, fs = torchaudio.load(save_path)
        sad = self.compute_SAD(s, fs)
        s = s[np.where(sad == 1)]
        return s, fs

    @staticmethod
    def _resample_audio(audio, sample_rate, resample_rate):
        return torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=resample_rate), resample_rate


class Patient:
    def __init__(self, observation: Observation):

        self.id: str = None
        self.age: int = None
        self.gender: str = None

        self.covid: bool = None
        self.long_covid: bool = None

        self._get_info_from_observation(observation)

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

        patient_fhir = raw_patient if isinstance(raw_patient, Patient) else Patient.parse_obj(raw_patient)

        self.id = get_id(patient_fhir)
        self.age = get_age(patient_fhir)
        self.gender = get_gender(patient_fhir)

    def put_assign_covid(self, diagnosis: bool):
        self.covid = diagnosis

    def put_long_covid(self, diagnosis: bool):
        self.long_covid = diagnosis

    def _get_info_from_observation(self, observation: Observation):
        patient_id = observation.subject.reference.split('/')[-1]
        corilga_api = CoperiaApi()
        patient = corilga_api.get_patient(patient_id)

        self.id = patient_id
        self.age = self.get_age(patient)
        self.gender = self.get_gender(patient)


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
