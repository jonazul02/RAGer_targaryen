import os
import yaml
from yaml.loader import SafeLoader

# import json


class ConfigurationLoader:

    @staticmethod
    def get_config():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "../configs/cfg_file.yml")
        with open(config_path, "r", encoding="utf-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=SafeLoader)
        return cfg
