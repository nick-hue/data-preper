import argparse
from pathlib import Path
from rich.console import Console
from utils.config_loader import Preper, read_config_file
from utils.log_utils import prompt_user_command, status, run_command
import logging

def run_sfm(config_file: Path,
            output_dir: Path,
            vocab_tree_path: Path,
            prompt: bool,
            verbose: bool = False
            ) -> None:
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
        prompt_user_command(command_name="feature extraction", console=CONSOLE)

    info_msg = f"Running feature extraction."
    logger.info(f"Command >> {feature_extractor_cmd}")
    logger.info(info_msg)
    CONSOLE.log("[bold green]"+info_msg)
    with status("Running...", spinner="moon", verbose=verbose, console=CONSOLE):
        run_command(cmd=feature_extractor_cmd, verbose=verbose, console=CONSOLE)
    info_msg = "Done extracting COLMAP features."
    logger.info(info_msg) 
    CONSOLE.log("[bold green]:tada:"+info_msg)
    
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
        prompt_user_command(command_name="feature matching", console=CONSOLE)

    with status("Running...", spinner="moon", verbose=verbose, console=CONSOLE):
        run_command(cmd=feature_extractor_cmd, verbose=verbose, console=CONSOLE)
    
    info_msg = f"Running {preper.matching_method} matcher feature matching."
    logger.info(f"Command >> {feature_matcher_cmd}")
    logger.info(info_msg)
    CONSOLE.log("[bold green]"+info_msg)    
    with status("Running...", spinner="moon", verbose=verbose, console=CONSOLE):
        run_command(cmd=feature_matcher_cmd, verbose=verbose, console=CONSOLE)        
    info_msg = "Done matching COLMAP features."
    logger.info(info_msg) 
    CONSOLE.log("[bold green]:tada:"+info_msg)
    
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
        prompt_user_command(command_name="mapper", console=CONSOLE)

    info_msg = f"Running {preper.sfm_tool} mapper."   
    logger.info(f"Command >> {mapper_cmd}")
    logger.info(info_msg)
    CONSOLE.log("[bold green]"+info_msg)    
    with status("Running...", spinner="moon", verbose=verbose, console=CONSOLE):
        run_command(cmd=mapper_cmd, verbose=verbose, console=CONSOLE)    
    info_msg = "Done COLMAP mapping."
    logger.info(info_msg) 
    CONSOLE.log("[bold green]:tada:"+info_msg)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare input data for nerfstudio training via config file.")
    parser.add_argument('--config_file', required=True, type=Path, help="Path to the config file.")
    parser.add_argument('--output_dir', required=True, type=Path, help="Path to the output directory.")
    parser.add_argument('--vocab_tree_path', required=False, type=Path, help="Path to the vocab tree, only needed when <matching_method> is <vocab_tree>.")
    parser.add_argument('-p', '--prompt', action='store_true', help="Flag to prompt each time before running a command.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Flag to print command extra information about commands.")
    parser.add_argument('-l', '--log', action='store_true', help="Flag to log command outputs and information.")
    parser.add_argument('--log_file', required=False, type=Path, help="Logging file path, if [log] flag is set. (default: command_logs.log)")
    
    # TODO: verbose console loggins with loggers ?? 
    # TODO: error logging
    # TODO: turn colmaped data into nerfstudio data
    # TODO: make nerfacto feature

    CONSOLE = Console(width=120)

    args = parser.parse_args()    
    logger = logging.getLogger(__name__)

    if args.log:
        print("OK")
        logging.basicConfig(filename=args.log_file if args.log_file else "command_logs.log", 
                            format='%(asctime)s : %(levelname)s : %(message)s',
                            filemode='w',
                            level=logging.INFO)
    else:
        print("NOT OK")
    # 
   


    
    run_sfm(args.config_file, args.output_dir, args.vocab_tree_path, args.prompt, args.verbose)
    # sfm to nerfacto
    # train model

