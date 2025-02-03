import argparse
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class Preper:
    train_method: str 
    sfm_tool: str
    matcher_type: str 



def read_config_file(config_file: Path):
    with open(config_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    
    return Preper(*data.values())


def run_sfm(config_file: Path, output_dir: Path):
    # print(f"{config_file=}")

    preper: Preper = read_config_file(config_file=config_file)
    print(preper)
    print(f"{output_dir=}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP with specified image path and matcher type.")
    parser.add_argument('--config_file', required=True, help="Path to the config file.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory.")
    
    args = parser.parse_args()

    run_sfm(args.config_file, args.output_dir)



