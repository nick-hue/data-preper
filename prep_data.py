import argparse
from pathlib import Path
import yaml
from dataclasses import dataclass, fields, _MISSING_TYPE
from typing import Literal, Optional, get_type_hints
from rich.console import Console
import subprocess
import sys

CONSOLE = Console(width=170)

@dataclass
class Preper:
    train_method: Literal["nerfacto", "splatfacto"] = "nerfacto"
    sfm_tool: Literal["colmap", "glomap"] = "colmap"
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    database_path: Path = Path("")
    image_dir: Path = Path("")
    camera_model: Literal["OPENCV", "OPENCV_FISHEYE", "EQUIRECTANGULAR", "PINHOLE", "SIMPLE_PINHOLE"] = "OPENCV"
    use_gpu: Literal[0,1] = 1

    def __post_init__(self) -> None:
        '''
        makes sure fields that were given from the config file are correctly passed
        '''
        type_hints = get_type_hints(self.__class__)

        for field in fields(self):
            if hasattr(type_hints[field.name], '__args__'):
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
    
    print(data)
    return Preper(train_method=data['train_method'],\
                sfm_tool=data['sfm_tool'], \
                matching_method=data['matching_method'],
                database_path=data['database_path'],
                image_dir=data['image_dir'],
                camera_model=data['camera_model'],
                use_gpu=data['use_gpu'])

def run_sfm(config_file: Path, output_dir: Path, vocab_tree_path: Path) -> None:
    '''
    runs the Structure-from-Motion command with the speficied configurations
    '''

    preper: Preper = read_config_file(config_file=config_file)

    if preper.matching_method == "vocab_tree" and vocab_tree_path is None:
        raise FileNotFoundError("If [matching_method] is <vocab_tree>, then a [vocab_tree_path] is needed.")
    
    # print(preper)
    # print(f"{output_dir=}")
    # print(f"{vocab_tree_path=}")

    colmap_cmd = 'colmap'
    verbose = True

    # Feature extraction command 
    feature_extractor_cmd = [
        f"{colmap_cmd} feature_extractor",
        f"--database_path {preper.database_path}",
        f"--image_path {preper.image_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {preper.camera_model}",
        f"--SiftExtraction.use_gpu {preper.use_gpu}",
    ]
    
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    print(f"{feature_extractor_cmd=}")

    with CONSOLE.status("Running feature extraction...", spinner="moon"):
        output = subprocess.run(f"{feature_extractor_cmd}", capture_output=True, shell=True, check=False)

    # print(f"{output=}")
    if output.stderr and verbose:
        CONSOLE.log(f"{output.stderr.decode('utf-8')}")
    
    CONSOLE.log("[bold green]:tada: Done extracting COLMAP features.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input data for nerfstudio training via config file.")
    parser.add_argument('--config_file', required=True, help="Path to the config file.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory.")
    parser.add_argument('--vocab_tree_path', required=False, help="Path to the vocab tree, only needed when <matching_method> is <vocab_tree>.")
    
    args = parser.parse_args()

    run_sfm(args.config_file, args.output_dir, args.vocab_tree_path)



