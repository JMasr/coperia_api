import io
import os


class Config:

    def __init__(self, path: str = '.env.default'):
        # Variables loaded from .env
        self.env: dict = {}
        self.load_env(path)

    # Load environment variables from .env
    def load_env(self, path: str) -> bool:
        """
        Load a configuration file
        :param path: Path to the configuration file
        :return: True if the file is valid and False if it isn't
        """

        if os.path.exists(path):
            with io.open(path) as stream:
                env_variables = stream.readlines()

            for variable in env_variables:
                parts = variable.split('=')
                self.env[parts[0].upper()] = parts[1].strip()
            return True

        else:
            return False

    def get(self, variable: str) -> str:
        return self.env[variable.upper()]


config = Config()
