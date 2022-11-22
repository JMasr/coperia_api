import base64
import tempfile
from datetime import datetime

import torch
import torchaudio
from spafe.features.bfcc import bfcc

from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from spafe.features.cqcc import cqcc

from src.config import config


class Response:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class CoperiaAudio:
    def __init__(self, observation: Observation, resample_rate: int = 16000):
        self.id = observation.id
        self.status = observation.status
        self.extension = observation.contained[0].content.contentType.split('/')[-1]
        self.duration = float(observation.contained[0].duration)

        self.type = observation.code.text.lower().replace(' ', '_')
        self.type_code = observation.code.coding[0].code

        self.data_base64 = observation.contained[0].content.data

        self.wave_form, self.sample_rate = self._load_audio()
        self.resample_rate = resample_rate
        self.resample_wave_form = self._resample_audio()

    def __str__(self):
        return f'ID: {self.id}\n' \
               f'Type: {self.type} \n' \
               f'Duration: {self.duration}\n' \
               f'Extension: {self.extension}'

    def __len__(self):
        return self.duration

    def _load_audio(self):
        decode64_data = base64.b64decode(self.data_base64)

        with tempfile.NamedTemporaryFile(suffix='.wav') as wav_file:
            wav_file.write(decode64_data)
            return torchaudio.load(wav_file.name)

    def _resample_audio(self):
        return torchaudio.functional.resample(self.wave_form, self.resample_rate, self.resample_rate)


class Feats:
    def __init__(self, audio: CoperiaAudio):
        sig, fs = audio.resample_wave_form, audio.resample_rate
        self.mfcc = self.make_mfcc(sig, fs)
        self.bfcc = self.make_bfcc(sig, fs)
        self.cqcc = self.make_cqcc(sig, fs)

    @staticmethod
    def make_mfcc(wave_form: torch.Tensor = None, sample_rate: int = None):
        """
        Make MFCC from a waveform
        :param wave_form:  Pytorch Tensor with the waveform
        :param sample_rate: int with the sample rate of the waveform
        :return: MFCC
        """
        arg = {"n_fft": int(config.get_key('N_FFT')),
               "hop_length": int(config.get_key('HOP_LENGTH')),
               "n_mels": int(config.get_key('N_MELS')),
               }

        transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=int(config.get_key('N_MFCC')),
                                               log_mels=bool(config.get_key('LOG_MELS')), melkwargs=arg)
        return transform(wave_form)

    @staticmethod
    def make_bfcc(wave_form: torch.Tensor = None, sample_rate: int = None):
        bfccs = bfcc(wave_form, fs=sample_rate,
                     pre_emph=1,
                     pre_emph_coeff=0.97,
                     win_len=0.030,
                     win_hop=0.015,
                     win_type="hamming",
                     nfilts=128,
                     nfft=2048,
                     low_freq=0,
                     high_freq=sample_rate/2,
                     normalize="mvn")

        return torch.from_numpy(bfccs)

    @staticmethod
    def make_cqcc(wave_form: torch.Tensor = None, sample_rate: int = None):
        cqccs = cqcc(wave_form, fs=sample_rate,
                     pre_emph=1,
                     pre_emph_coeff=0.97,
                     win_len=0.030,
                     win_hop=0.015,
                     win_type="hamming",
                     nfft=2048,
                     low_freq=0,
                     high_freq=sample_rate/2,
                     normalize="mvn")
        return torch.from_numpy(cqccs)

    # TODO: Adding more feats


class UVigoPatient:
    def __init__(self, json_patient=None, json_obs: list = None):
        self.patient = Patient.parse_obj(json_patient)
        self.observations = [Observation.parse_obj(obs) for obs in json_obs]

        self.patient_audios: list = self._load_audios()
        self.audios_feats: list = self._extract_feats()

        self.covid = None
        self.long_covid = None

    def get_id(self) -> str:
        """
        Return the patient's id.
        :return: patient's id.
        """
        return self.patient.id

    def get_age(self) -> int:
        """
        Return the patient's age.
        :return: patient's age.
        """
        born = datetime(self.patient.birthDate.year, self.patient.birthDate.month, self.patient.birthDate.day)
        return (datetime.utcnow() - born).days // 365

    def get_gender(self) -> str:
        """
        Return the patient's gender.
        :return: patient's gender.
        """
        return self.patient.gender

    def _load_audios(self) -> list:
        audios = []
        for obs in self.observations:
            audio = CoperiaAudio(obs, int(config.get_key('RESAMPLE_RATE')))
            audios.append(audio)
        return audios

    def _extract_feats(self) -> torch.Tensor:
        feats = []
        for audio in self.patient_audios:
            feat = Feats(audio)
            feats.append(feat)
        return feats
