# AIR-Project

## Environment Setup
To install the environment, run the following

`git clone https://github.com/axelbr/racecar_gym.git`
`cd racecar_gym`
`pip install -e .`

make sure you clone the repositroy into the root folder of the code

## CNN - YOLO (Windows)
### Setup
1. Install python 3.10 (required for compatibility with racecar_gym dependencies)
2.  Create venv in the root directory using python 3.10 `py -3.10 -m venv venv`
3. Activate venv
4. Install ultralytics `pip install ultralytics`
5. Downgrade numpy due to compatibility issues with racecar_gym: `pip install numpy==1.22.3`

Optional (for yolo training)
5. Remove current torch version `pip uninstall torch torchvision -y`
6. Reinstall torch with CUDA (NVIDIA GPU) support `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 `

### Execution
Refer to [single_test.py](./detection_model/single_test.py) for basic usage of trained model
