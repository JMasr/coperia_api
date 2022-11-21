from src.data import UVigoPatient
from src.util import Util


if __name__ == "__main__":
    json_patient = {"resourceType": "Patient",
                    "id": "p001",
                    "active": True,
                    "name": [{"text": "Adam Smith"}],
                    "birthDate": "1985-06-12"
                    }
    json_observation = Util.read_json('data/observation_audio.json')

    uvigo_pat = UVigoPatient(json_patient, json_observation)
    print('Done!')
