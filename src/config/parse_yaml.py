import os
import yaml
from types import SimpleNamespace

def parse_yaml(config_file=None):
    if config_file is not None:
        assert(os.path.isfile(config_file))
        with open(config_file, 'r') as fo:
            parsed_yaml = yaml.load(fo, Loader=yaml.FullLoader)
    else:
        print('config file not present')
        parsed_yaml = None
            
    return parsed_yaml


def get_config(config_file=None):
    parsed_yaml = parse_yaml(config_file)
    for key in parsed_yaml:
        parsed_yaml[key] = SimpleNamespace(**parsed_yaml[key] )
    cfg = SimpleNamespace(**parsed_yaml)
    return cfg