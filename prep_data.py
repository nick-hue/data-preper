import argparse
from pathlib import Path
import yaml
from dataclasses import dataclass, fields, _MISSING_TYPE
from typing import Literal, get_type_hints
import logging

@dataclass
class Preper:
    train_method: Literal["nerfacto", "splatfacto"] = "nerfacto"
    sfm_tool: Literal["colmap", "glomap"] = "colmap"
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"

    def __post_init__(self) -> None:
        '''
        makes sure fields that were given from the config file are correctly passed
        '''
        for field in fields(self):
            field_value = getattr(self, field.name)
            allowed_values = field.type.__args__
            if field_value not in allowed_values:
                raise ValueError(f"Invalid value <{field_value} for field [{field.name}]. Allowed values are: {allowed_values}.")

            if not isinstance(field.default, _MISSING_TYPE) and getattr(self, field.name) is None:
                raise ValueError(f"No value was passed for field : {field.name}")


def read_config_file(config_file: Path) -> Preper:
    '''
    reads the fields from the config file and creates a preper 
    '''
    with open(config_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    return Preper(train_method=data['train_method'], sfm_tool=data['sfm_tool'], matching_method=data['matching_method'])

def run_sfm(config_file: Path, output_dir: Path, vocab_tree_path: Path) -> None:
    '''
    runs the Structure-from-Motion command with the speficied configurations
    '''

    preper: Preper = read_config_file(config_file=config_file)

    if preper.matching_method == "vocab_tree" and vocab_tree_path is None:
        raise FileNotFoundError("If [matching_method] is <vocab_tree>, then a [vocab_tree_path] is needed.")
    
    print(preper)
    print(f"{output_dir=}")
    print(f"{vocab_tree_path=}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input data for nerfstudio training via config file.")
    parser.add_argument('--config_file', required=True, help="Path to the config file.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory.")
    parser.add_argument('--vocab_tree_path', required=False, help="Path to the vocab tree, only needed when <matching_method> is <vocab_tree>.")
    
    args = parser.parse_args()

    run_sfm(args.config_file, args.output_dir, args.vocab_tree_path)



