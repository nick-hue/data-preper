import argparse
from pathlib import Path
import yaml
from dataclasses import dataclass, fields, _MISSING_TYPE
from typing import Literal, Optional, get_type_hints
from rich.console import Console
import subprocess
import sys
from contextlib import nullcontext
import time

CONSOLE = Console(width=120)

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
    
    # print(data)
    return Preper(train_method=data['train_method'],\
                sfm_tool=data['sfm_tool'], \
                matching_method=data['matching_method'],
                database_path=data['database_path'],
                image_dir=data['image_dir'],
                camera_model=data['camera_model'],
                use_gpu=data['use_gpu'])

def run_command(cmd: str, verbose=False) -> Optional[str]:
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        CONSOLE.rule("[bold red] :skull: :skull: :skull: ERROR :skull: :skull: :skull: ", style="red")
        CONSOLE.print(f"[bold red]Error running command: {cmd}")
        CONSOLE.rule(style="red")
        CONSOLE.print(out.stderr.decode("utf-8"))
        sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out

def status(msg: str, spinner: str = "bouncingBall", verbose: bool = False):
    """A context manager that does nothing is verbose is True. Otherwise it hides logs under a message.

    Args:
        msg: The message to log.
        spinner: The spinner to use.
        verbose: If True, print all logs, else hide them.
    """
    if verbose:
        return nullcontext()
    return CONSOLE.status(msg, spinner=spinner)

def prompt_user_command(command_name: str):
    choice = input(f"Do you want to run {command_name}? [y]/n\n")
    if choice.lower() == "n":
        CONSOLE.log("[bold red]:x: Exiting...")   
        sys.exit(0)
    

def run_sfm(config_file: Path, output_dir: str, vocab_tree_path: str, prompt: bool, verbose: bool = False) -> None:
    '''
    runs the Structure-from-Motion command with the speficied configurations
    '''

    preper: Preper = read_config_file(config_file=config_file)

    # checking if valid vocab_tree arguments passed 
    if preper.matching_method == "vocab_tree":
        if not vocab_tree_path.endswith(".fbow"):
            raise FileNotFoundError(f"Supplied file [{vocab_tree_path}] does not end with '.fbow', a valid vocab tree path is needed.")
        elif vocab_tree_path is None:
            raise FileNotFoundError("If [matching_method] is <vocab_tree>, then a [vocab_tree_path] is needed.")

    colmap_cmd = 'colmap'

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
    if verbose:
        print(f"{feature_extractor_cmd=}")
    
    if prompt:
        prompt_user_command("feature extraction")

    CONSOLE.log(f"[bold green]Running feature extraction.")   
    with status("Running...", spinner="moon", verbose=verbose):
        run_command(cmd=feature_extractor_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done extracting COLMAP features.")   

    # Feature matching command 
    feature_matcher_cmd = [f"colmap {preper.matching_method}_matcher",
        f"--database_path {preper.database_path}",
        f"--SiftMatching.use_gpu {preper.use_gpu}"
    ]
    if preper.matching_method == "vocab_tree":
        feature_matcher_cmd.append(f'--VocabTreeMatching.vocab_tree_path "{vocab_tree_path}"')
    
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    if verbose:
        print(f"{feature_matcher_cmd=}")

    if prompt:
        prompt_user_command("feature matching")

    CONSOLE.log(f"[bold green]Running {preper.matching_method} matcher feature matching.")   
    with status("Running...", spinner="moon", verbose=verbose):
        run_command(cmd=feature_matcher_cmd, verbose=verbose)        
    CONSOLE.log("[bold green]:tada: Done matching COLMAP features.")   

    # Mapping
    sparse_dir = Path(output_dir) / preper.sfm_tool / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    mapper_cmd = [
        f"{preper.sfm_tool} mapper",
        f"--database_path {preper.database_path}",
        f"--image_path {preper.image_dir}",
        f"--output_path {sparse_dir}",
    ]

    if preper.sfm_tool == 'colmap':
        #if colmap_version >= Version("3.7"):
        mapper_cmd.append("--Mapper.ba_global_function_tolerance=1e-6")

    mapper_cmd = " ".join(mapper_cmd)
    if verbose:
        print(f"{mapper_cmd=}")

    if prompt:
        prompt_user_command("mapper")

    CONSOLE.log(f"[bold green]Running {preper.sfm_tool} mapper.")   
    with status("Running...", spinner="moon", verbose=verbose):
        run_command(cmd=mapper_cmd, verbose=verbose)    
    CONSOLE.log("[bold green]:tada: Done COLMAP mapping.")   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input data for nerfstudio training via config file.")
    parser.add_argument('--config_file', required=True, help="Path to the config file.")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory.")
    parser.add_argument('--vocab_tree_path', required=False, help="Path to the vocab tree, only needed when <matching_method> is <vocab_tree>.")
    parser.add_argument('-p', '--prompt', action='store_true', help="Flag to prompt each time before running a command.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Flag to print command extra information about commands.")
    
    # TODO: turn colmaped data into nerfstudio data
    # TODO: make nerfacto feature
    # TODO: log command information

    args = parser.parse_args()

    run_sfm(args.config_file, args.output_dir, args.vocab_tree_path, args.prompt, args.verbose)

