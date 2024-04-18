import os
import yaml
def load_config_file(EXP_NAME, path_prefix=''):
    '''
    Load YAML config file corresponding to an experiment name
    '''
    config_path = os.path.join(path_prefix, 'configs/'+EXP_NAME+".yml")
    with open(config_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
