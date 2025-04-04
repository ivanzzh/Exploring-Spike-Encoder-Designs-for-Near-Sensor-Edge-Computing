import yaml
from filelock import FileLock

# read configuration file
def load_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# save configuration file
def save_config(config, file_path):
    with open(file_path, 'w') as f:
        yaml.dump(config, f)
    # print('save file to ' + file_path)

# configuration for reservoir experiments
Res_config = {
    '0': {
        'name': 'FingerMovements',
        'input_size': 28,
        'r_state': 120,
        'layer1': 100,
        'layer2': 100,
        'layer3': 2,
        'k_value': 0.9,
        'window_length': 50,
        'batch_size': 4,
        'point_num': 8,
        'no_bias': False,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 39
    },
    '1': {
        'name': 'BasicMotions',
        'input_size': 6,
        'r_state': 30,
        'layer1': 100,
        'layer2': 100,
        'layer3': 4,
        'k_value': 0.7,
        'window_length': 100,
        'batch_size': 4,
        'point_num': 8,
        'no_bias': False,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 10
    },
    '2': {
        'name': 'Epilepsy',
        'input_size': 3,
        'r_state': 15,
        'layer1': 100,
        'layer2': 100,
        'layer3': 4,
        'k_value': 0.9,
        'window_length': 207,
        'batch_size': 4,
        'point_num': 8,
        'no_bias': False,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 17
    },
    '3': {
        'name': 'JapaneseVowels',
        'input_size': 12,
        'r_state': 50,
        'layer1': 100,
        'layer2': 100,
        'layer3': 9,
        'k_value': 0.7,
        'window_length': 29,
        'batch_size': 4,
        'point_num': 8,
        'no_bias': False,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 22
    },
    '4': {
        'name': 'RacketSports',
        'input_size': 6,
        'r_state': 30,
        'layer1': 100,
        'layer2': 100,
        'layer3': 4,
        'k_value': 0.9,
        'window_length': 30,
        'batch_size': 4,
        'point_num': 8,
        'no_bias': False,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 18
    },
    '5': {
        'name': 'SelfRegulationSCP1',
        'input_size': 6,
        'r_state': 24,
        'layer1': 100,
        'layer2': 100,
        'layer3': 4,
        'k_value': 0.9,
        'window_length': 896,
        'batch_size': 4,
        'point_num': 20,
        'no_bias': False,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 20
    },
    '6': {
        'name': 'EMG_Action',
        'input_size': 8,
        'r_state': 40,
        'layer1': 200,
        'layer2': 200,
        'layer3': 10,
        'k_value': 0.7,
        'window_length': 1000,
        'batch_size': 4,
        'point_num': 30,
        'no_bias': False,
        'data_split': False,
        'all_data_path': '/home/zzhan281/dataset/EMG_Action/dealed_data/total_data_1000length_500step_Normalclasses.npz',
        'split_ratio': 0.8,
        'print_period': 14
    }
}

# configuration for population coding experiments
config = {
    '0': {
        'name': 'EMG_gesture',
        'input_size': 8,
        'layer1': 100,
        'layer2': 100,
        'layer3': 7,
        'window_length': 100,
        'batch_size': 40,
        'point_num': 8,
        'data_split': False,
        'all_data_path': '/data/zhenhang/EMG_gesture/dealed_data/all_data_100len_100step_filtered.npz',
        'split_ratio': 0.8,
        'print_period': 98
    },
    '1': {
        'name': 'FingerMovements',
        'input_size': 28,
        'layer1': 100,
        'layer2': 100,
        'layer3': 2,
        'window_length': 50,
        'batch_size': 4,
        'point_num': 2,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 39
    },
    '2': {
        'name': 'BasicMotions',
        'input_size': 6,
        'layer1': 100,
        'layer2': 100,
        'layer3': 4,
        'window_length': 100,
        'batch_size': 4,
        'point_num': 4,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 10
    },
    '3': {
        'name': 'Epilepsy',
        'input_size': 3,
        'layer1': 100,
        'layer2': 100,
        'layer3': 4,
        'window_length': 207,
        'batch_size': 4,
        'point_num': 8,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 17
    },
    '4': {
        'name': 'JapaneseVowels',
        'input_size': 12,
        'layer1': 100,
        'layer2': 100,
        'layer3': 9,
        'window_length': 29,
        'batch_size': 4,
        'point_num': 8,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 34
    },
    '5': {
        'name': 'RacketSports',
        'input_size': 6,
        'layer1': 100,
        'layer2': 100,
        'layer3': 4,
        'window_length': 30,
        'batch_size': 4,
        'point_num': 8,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 19
    },
    '6': {
        'name': 'SelfRegulationSCP1',
        'input_size': 6,
        'layer1': 100,
        'layer2': 100,
        'layer3': 4,
        'window_length': 896,
        'batch_size': 4,
        'point_num': 20,
        'data_split': True,
        'all_data_path': None,
        'split_ratio': 0.8,
        'print_period': 20
    },
    '7': {
        'name': 'EMG_Action',
        'input_size': 8,
        'layer1': 100,
        'layer2': 100,
        'layer3': 10,
        'window_length': 1000,
        'batch_size': 40,
        'point_num': 8,
        'data_split': False,
        # 'all_data_path': None,
        'all_data_path': '/home/zzhan281/dataset/EMG_Action/dealed_data/total_data_1000length_500step_Normalclasses.npz',
        'split_ratio': 0.85,
        'print_period': 16
    }
}

# autosave as importing
save_config(config, 'config.yaml')
save_config(Res_config, 'res_config.yaml')
