from src.config import Config


# An CORILGA API class
class CorilgaApi:

    def __init__(self, env_path):
        config: Config = Config(env_path)
        self.key: str = config.get_key('CORILGA_API_KEY')
        self.url: str = config.get_key('CORILGA_API_URL')

