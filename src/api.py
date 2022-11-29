import os
import json
import warnings
import requests

from src.config import Config

from fhir.resources.patient import Patient
from fhir.resources.observation import Observation


# An CORILGA API class
class CorilgaApi:

    def __init__(self, env_path: str = ''):
        self.fhir_config = Config(os.path.join(env_path, '.env.fhir'))
        self.keycloak_config = Config(os.path.join(env_path, '.env.keycloak'))
        self.url_server = f"{self.fhir_config.get_key('URL_FHIR')}/api/v4"

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
        """
        Getting token to call keycloak add user api
        :return: the access token request
        """
        accessTokenUrl = self.keycloak_config.get_key('URL_KEYCLOAK_TOKEN')

        # credential for your keycloak instance
        username = self.keycloak_config.get_key('USER')
        password = self.keycloak_config.get_key('PASSWORD')
        payload = f'client_id=uvigo-app&username={username}&password={password}&grant_type=password'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        response = requests.request("POST", accessTokenUrl, headers=headers, data=payload)
        if self.connection_is_success(response):
            return json.loads(response.text)['access_token']
        else:
            raise ConnectionError(f'Error -> {json.loads(response.text)["error"]} |'
                                  f' Description -> {json.loads(response.text)["error_description"]}')

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

    def get_observations_by_code(self, observation_code: str = "84728-5", update_data: str = 'gt2022-08-31') -> list:
        """
        Get all the observations with the same code, using it as the query
        :param observation_code: code used as the search query
        :param update_data: data to filter the version of data
        :return: a list with the observation or an empty list if the code does not exist
        """
        access_token = self.get_access_token()
        header = {'Authorization': 'Bearer ' + access_token}
        r_url = f'{self.url_server}/Observation?code=http://loinc.org|{observation_code}&_lastUpdated={update_data}'

        response: dict = self.basic_request('GET', r_url, header)

        if response.get('total') > 0:
            raw_observations: list = response.get('entry')
            if raw_observations is None:
                raise ValueError('The Bundle of Observation does not have entries')
            return [Observation.parse_obj(raw.get('resource')) for raw in raw_observations]
        else:
            warnings.warn(ResourceWarning(f'No Observation with the code {observation_code}'))
            return []
