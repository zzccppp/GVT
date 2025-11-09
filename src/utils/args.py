import argparse
from copy import deepcopy
import json
from typing import Dict, Any

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def set_by_path(d: Dict[str, Any], path: str, value: Any, sep: str = '.') -> None:
    keys = path.split(sep)
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def get_value_type(value: Any) -> type:
    if isinstance(value, bool):
        return lambda x: bool(eval(x))
    return type(value)

def parse_args(config: Dict[str, Any]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Configuration arguments')
    
    flat_config = flatten_dict(config)
    
    for key, value in flat_config.items():
        value_type = get_value_type(value)
        parser.add_argument(
            f'--{key}',
            type=value_type,
            default=None,
            help=f'Default: {value}, Type: {value_type.__name__}'
        )
    
    parser.add_argument('--config_file', type=str, help='Path to JSON config file')
    
    args = parser.parse_args()
    return args

def update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = deepcopy(config)
    args_dict = vars(args)
    
    if args.config_file:
        with open(args.config_file, 'r') as f:
            file_config = json.load(f)
            for key, value in flatten_dict(file_config).items():
                set_by_path(config, key, value)
    
    for key, value in args_dict.items():
        if value is not None and key != 'config_file':
            set_by_path(config, key, value)
    
    return config

def print_config_diff(original_config: Dict[str, Any], updated_config: Dict[str, Any]) -> None:
    flat_original = flatten_dict(original_config)
    flat_updated = flatten_dict(updated_config)
    
    print("Configuration changes:")
    for key in flat_original:
        if key in flat_updated and flat_original[key] != flat_updated[key]:
            print(f"{key}: {flat_original[key]} -> {flat_updated[key]}")
