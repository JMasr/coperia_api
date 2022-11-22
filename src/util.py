import json
import requests
from src.data import Response


class Util:

    @staticmethod
    def write_json(response: dict, path_to_write: str = "data/json_test.json"):
        try:
            with open(f"{path_to_write}", "w") as outfile:
                json_str = json.dumps(response, indent=len(response))
                outfile.write(json_str)
                return True
        except IOError:
            raise "Error writing JSON"

    @staticmethod
    def read_json(path_to_json: str) -> dict:
        try:
            with open(path_to_json, 'r') as file:
                json_data = json.load(file)
        except IOError:
            raise "Error read JSON"

        return json_data

    @staticmethod
    def response_to_object(response: dict) -> Response:
        return Response(**response)

    @staticmethod
    def connection_is_success(response: requests.request):

        if response.status_code == 200:
            return True
        else:
            raise ConnectionError(f"Status code: {response.status_code}.\n"
                                  f"{response.text}")

    def basic_request(self, r_type: str, url: str, header=None, payload: dict = None,
                      files: dict = None) -> json.JSONDecoder:
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
