import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

class CustomRacecarWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # 1. Define custom observation space (e.g., concatenated state of selected sensors)
        # Assuming we combine LIDAR (1080,) and pose/velocity features into a smaller vector.
        # Here, 1080 (Lidar) + 6 (Pose) + 6 (Velocity) = 1092
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(1092,), 
            dtype=np.float64
        )
        
        # Define the PPO Action space (e.g., continuous throttle and steering)
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Process the observation and custom reward
        processed_obs = self._process_observation(obs)
        custom_reward = self._compute_reward(obs, reward, terminated, truncated, info)
        
        return processed_obs, custom_reward, terminated, truncated, info

    def _process_observation(self, obs):
        """
        Extract and flatten specific features from the dictionary observation.
        """
        lidar = obs['lidar']
        pose = obs['pose']
        velocity = obs['velocity']
        
        # Flatten and concatenate arrays into a single 1D vector
        processed_obs = np.concatenate([lidar, pose, velocity])
        
        return processed_obs

    def _compute_reward(self, obs, base_reward, terminated, truncated, info):
        """
        Define your custom reward function.
        """
        # Example: Reward based on forward velocity, with a large penalty for crashing (if detectable)
        velocity = obs['velocity'][:3] # x, y, z velocities
        forward_speed = np.linalg.norm(velocity)
        
        custom_reward = forward_speed * 1.0 # Reward for going fast
        
        if terminated and info.get('crashed', False): # Adjust info key based on your specific YAML
            custom_reward -= 100.0 # Penalty for crashes
            
        return custom_reward

if __name__ == "__main__":
    # Initialize the wrapped environment
    env = gym.make(id='SingleAgentAustria-v0', render_mode=None, scenario= 'track.yml')
    env = CustomRacecarWrapper(env)

    # Initialize PPO model with a Multi-Layer Perceptron (MlpPolicy)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/")

    # Train the model
    model.learn(total_timesteps=100_000)

    # Save the trained model
    model.save("ppo_racecar_model")