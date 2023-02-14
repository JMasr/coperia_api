import json
import os
import warnings

import requests
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from keycloak import KeycloakOpenID

from src.config import Config


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
            r_url = f'https://api.coperia.es/fhir-server/api/v4/Observation?code={observation_code}&_count={observations_total}'
            response: dict = self.basic_request('GET', r_url, header)
            raw_observations: list = response.get('entry')
            if raw_observations is None:
                raise ValueError('The Bundle of Observation does not have entries')
            return [Observation.parse_obj(raw.get('resource')) for raw in raw_observations]
        else:
            warnings.warn(ResourceWarning(f'No Observation with the code {observation_code}'))
            return []
