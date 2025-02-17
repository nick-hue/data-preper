from pathlib import Path
from dataclasses import dataclass, fields, _MISSING_TYPE
from typing import Literal, get_type_hints
import yaml 


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