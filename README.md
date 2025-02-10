# Data preperation script

## Script Information
This script is made to automate the processing of the input data for training a Radiance Field (RF) model with the *nerfstudio* library. In this script via a config file the user supplies information about the type of processing that they want to apply to the input data. 

The script first executes the Structure-From-Motion (SfM) process which estimates the camera poses of the frames supplied as input data.
Then this processed data is converted via the *nerfstudio* library and namely the `ns-process-data` command into train ready data to train our model of choice.

## Installation
The following command downloads the needed library that are used to run the script.


```
pip install -r requirements.txt
```

## Script Execution

### 1. Config file
First you will need to make a config file with the following attributes:
* **train_method**: model training method (`nerfacto`, `splatfacto`)
* **sfm_tool**: tool to be used for the SfM (`colmap`, `glomap`)
* **matching_method**: feature matching method (`exhaustive`, `sequential`,`vocab_tree`)
* **database_path**: path for the `database.db` file for the SfM
* **image_dir**: directory of the input data
* **camera_model**: Camera Model
* **use_gpu**: flag whether to use the GPU for the SfM 

### 2. Script Arguments
After having your config file ready the script expects the following arguments:
* **config_file** : Path to the config file. (required)
* **output_dir** : Path to the output directory. (required)
* **vocab_tree_path** : Path to the vocab tree (required, only for `matching_method` being `vocab_tree`)

### 3. Running the Script
Now you should be ready for executing the script, which looks like this:
```
python prep_data.py --config_file <your_conf_file> --output_dir <output_directory>
```