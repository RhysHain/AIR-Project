# AIR-Project

## Environment Setup
To install the environment, run the following

`git clone https://github.com/axelbr/racecar_gym.git`
`cd racecar_gym`
`pip install -e .`

make sure you clone the repositroy into the root folder of the code

## CNN - YOLO 
### Setup
1. Create venv in the root directory `python -m venv venv`
2. Activate venv
3. Install ultralytics `pip install ultralytics`
4. Remove current torch version `pip uninstall torch torchvision -y`
5. Reinstall torch with CUDA (NVIDIA GPU) support `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 `

### Execution
Refer to [single_test.py](./detection_model/single_test.py) for basic usage of trained model
