from time import sleep
import gymnasium
from racecar_gym.envs import gym_api
import pybullet as p

env = gymnasium.make(
    id='MultiAgentRaceEnv-v0',
    scenario='track.yml',
    render_mode='human'
)

print(env.observation_space)
print(env.action_space)

obs = env.reset(options=dict(mode='grid'))

# What needs to be added into other exeutables for signs to spawn in environment
# Define signs
signs_config = [
    {
        'name': 'stop_sign_1',
        'obj': 'env_signs/stop.obj',
        'texture': 'racecar_gym/models/signs/stop.png',
        'position': [0, -0.3, 0.75],           # x, y, z coordinates
        'rotation': [0, 1.57, 1.57],           # roll (red), pitch (green), yaw (blue)
        'scale': [0.5, 0.5, 0.5],              # x, y, z scale (0.5 = half size)
        'collision_size': [0.25, 0.05, 0.3]    # width/2, depth/2, height/2
    },
    
    {
        'name': 'go_sign_1',
        'obj': 'env_signs/go.obj',
        'texture': 'env_signs/go.png',
        'position': [2, -0.3, 0.75],
        'rotation': [0, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'slow_sign_1',
        'obj': 'env_signs/slow.obj',
        'texture': 'env_signs/slow.png',
        'position': [9, -0.3, 0.75],
        'rotation': [0, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'go_sign_2',
        'obj': 'env_signs/go.obj',
        'texture': 'env_signs/go.png',
        'position': [10.5, -2, 0.75],
        'rotation': [1.57, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'slow_sign_2',
        'obj': 'env_signs/slow.obj',
        'texture': 'env_signs/slow.png',
        'position': [16, -18, 0.75],
        'rotation': [1.1, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'go_sign_3',
        'obj': 'env_signs/go.obj',
        'texture': 'env_signs/go.png',
        'position': [14.25, -19.5, 0.75],
        'rotation': [-2.8, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'slow_sign_3',
        'obj': 'env_signs/slow.obj',
        'texture': 'env_signs/slow.png',
        'position': [1.3, -14.5, 0.75],
        'rotation': [-2.8, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'go_sign_4',
        'obj': 'env_signs/go.obj',
        'texture': 'env_signs/go.png',
        'position': [1.3, -12.5, 0.75],
        'rotation': [5.84, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'slow_sign_4',
        'obj': 'env_signs/slow.obj',
        'texture': 'env_signs//slow.png',
        'position': [8.7, -13.4, 0.75],
        'rotation': [6.5, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'go_sign_5',
        'obj': 'env_signs/go.obj',
        'texture': 'env_signs/go.png',
        'position': [6, -8, 0.75],
        'rotation': [2.7, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'slow_sign_5',
        'obj': 'env_signs/slow.obj',
        'texture': 'env_signs/slow.png',
        'position': [-6, -6.2, 0.75],
        'rotation': [3.14, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'go_sign_6',
        'obj': 'env_signs/go.obj',
        'texture': 'env_signs/go.png',
        'position': [-7.2, -4, 0.75],
        'rotation': [4.71, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'slow_sign_6',
        'obj': 'env_signs/slow.obj',
        'texture': 'env_signs/slow.png',
        'position': [-7.2, -1.2, 0.75],
        'rotation': [4.71, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },

    {
        'name': 'go_sign_7',
        'obj': 'env_signs/go.obj',
        'texture': 'env_signs/go.png',
        'position': [-5, -0.6, 0.75],
        'rotation': [0, 1.57, 1.57],
        'scale': [0.5, 0.5, 0.5],
        'collision_size': [0.25, 0.05, 0.3]
    },
]

# Load Signs
print(f"Loading {len(signs_config)} signs...")

for sign_cfg in signs_config:
    try:
        # Load texture
        texture_id = p.loadTexture(sign_cfg['texture'])
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=sign_cfg['obj'],
            meshScale=sign_cfg['scale'],       # Now using scale from config
            rgbaColor=[1, 1, 1, 1]
        )
        
        # Create collision shape
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=sign_cfg['collision_size']
        )
        
        # Create the sign
        sign_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=sign_cfg['position'],
            baseOrientation=p.getQuaternionFromEuler(sign_cfg['rotation'])
        )
        
        # Apply texture
        p.changeVisualShape(sign_id, -1, textureUniqueId=texture_id)
        
        # Debuggin Purposes
        print(f"Loaded {sign_cfg['name']} at {sign_cfg['position']}")
        
    except Exception as e:
        print(f"Failed to load {sign_cfg['name']}: {e}")


print("Press Ctrl+C to exit")

# End of sign spawning

try:
    while True:
        action = env.action_space.sample()
        obs, rewards, dones, truncated, states = env.step(action)
        env.render()
        sleep(0.01)
except KeyboardInterrupt:
    print("\nExiting...")
    
env.close()