import yaml


def load_params(params_path:str)->dict:
    with open(params_path, 'r') as file:
        params=yaml.safe_load(file)
    return params