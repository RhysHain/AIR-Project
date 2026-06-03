# AIR-Project

## Group Information
**Group Number:** 5

**Rhys Hain** - *Simulation, Documentation and Organisation*
- Set up and tested simulation environment
- Created and managed GitHub repository, as well as code set up
- Managed portfolio, video and presentation development

**Huu Thanh Jack Nguyen** - *YOLO training & Environment Design*
- Designed custom sign models (STOP, SLOW, GO) in Blender
- Created labeled dataset from custom models
- Trained YOLOv8 detection model
- Added sign models into simulation environment

**Mateusz Kopaczynski** - *PPO Algorithm*
- Developed the observation function
- Determined parameters and calculations involved in the reward function
- Tested and evalutated PPO for car movement

**Alexander Seretis** - *System Integration*
- Expanded observation function to include sign classification via custom detection array
- Added sign classification visual to test rendering
- Assisted adding signs to environment
- Trained driving agent with sign detection

## Project Overview
This project implements a car (agent) in a simulated race track (Racecar Gym) that combines computer vision and reinforcement learning for autonomous driving around the track

### Key Features
- **Sign Detection**: YOLOv8 model trained on a labelled dataset to detect custom signs (STOP, SLOW, GO) in real time
- **Autonomous Navigation**: PPO agent learns to stay on the track boundaries (walls) and avoid collisions

### System Components
1. **YOLO Detection**
  - Custom trained YOLOv8 on ~600 labelled sign images capturing 'STOP, 'SLOW', 'GO' signs from multiple angles and distances
  - Real time sign detection from vehicle camera feed
  - **Decision Making - Signs**: Agent will adjust behaviour based on detected signs:
      - **STOP sign**: Make a complete stop
      - **SLOW sign**: Reduce speed
      - **GO sign**: Increase speed or maintain high speed

## System Setup
### Environment Setup
To install the environment, run the following

`git clone https://github.com/axelbr/racecar_gym.git`
`cd racecar_gym`
`pip install -e .`

make sure you clone the repositroy into the root folder of the code

### CNN - YOLO (Windows) Setup
1. Install python 3.10 (required for compatibility with racecar_gym dependencies)
2. Create venv in the root directory using python 3.10 `py -3.10 -m venv venv`
3. Activate venv `venv\Scripts\activate`
4. Install ultralytics `pip install ultralytics`
5. Downgrade numpy due to compatibility issues with racecar_gym: `pip install numpy==1.22.3`

Optional (for yolo training)

6. Remove current torch version `pip uninstall torch torchvision -y`
7. Reinstall torch with CUDA (NVIDIA GPU) support `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 `

### Execution
Refer to [single_test.py](./detection_model/single_test.py) for basic usage of trained model

Trained model location: : `detection_model/runs/detect/signs_v1-7/weights/best.pt`

### PPO
Use [test.py](./test.py) to run the trained PPO driving model with rendering and debug output.

The current PPO setup also uses a modified steering setting in Racecar Gym:

`racecar_gym/models/vehicles/racecar/racecar.yml`

In that file, the steering actuator should have:

```yml
steering_multiplier: 1.0
```

This increases the usable steering range for tight hairpin turns. If the car is not turning sharply enough, check that this value has not been reset to `0.5`.

### Detection in PPO Setup
To view detection incoporation, swap to `detection` branch and pull changes.
Then:
1. Repeat YOLO setup
2. Install ultralytics outside of venv `pip install ultralytics`
3. Downgrade numpy: `pip install numpy==1.23.5`
