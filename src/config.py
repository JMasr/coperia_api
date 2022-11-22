import io


class Config:

    def __init__(self, path: str = '.env.default'):
        # Variables loaded from .env
        self.env: dict = {}
        self.load_env(path)

    # Load environment variables from .env
    def load_env(self, path: str) -> bool:

        try:
            with io.open(path) as stream:
                env_variables = stream.readlines()
        except IOError:
            print(f"Error opening {path}")

        for variable in env_variables:
            parts = variable.split('=')
            self.env[parts[0]] = parts[1].strip()

    def get_key(self, variable) -> str:
        return self.env[variable]


config = Config()
