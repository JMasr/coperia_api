import io
import pickle
import torchaudio
import soundfile as sf
from fhir.resources.organization import Organization
from fhir.resources.observation import Observation
from fhir.resources.address import Address
from fhir.resources.patient import Patient


class Response:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Feats:
    def __init__(self):
        pass


class Audio:
    def __init__(self, json_contained):
        self.id = json_contained.id
        self.extention = json_contained.content.contentType.split('/')[-1]
        self.duration = float(json_contained.duration)

        self.waveform, self.sample_rate = None, None
        #torchaudio.load('data/test.wav')
        #sf.read(file=io.BytesIO(json_contained.content.data))

    def __len__(self):
        return self.duration


class UVigoPatient:
    def __init__(self, json_patient=None, json_obs=None):
        self.patient = Patient.parse_obj(json_patient)
        self.observation = Observation.parse_obj(json_obs)
        self.patient_audios: list = self.extract_audios(self.observation.contained)
        self.feats: list = None

    @staticmethod
    def extract_audios(contained) -> list:
        audios = []
        for content in contained:
            audio = Audio(content)
            audios.append(audio)
        return audios

    def load_feats(self, feats_path):
        self.feats = pickle.load(feats_path)
