import yaml
from pathlib import Path


def parse_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        conf = yaml.safe_load(f)
    return conf


# Alias so both names work
load_config = parse_config
