from time import sleep
import gymnasium as gym
import racecar_gym
from stable_baselines3 import PPO

# 1. Import your custom wrapper (Ensure this matches your training wrapper exactly)
from train import CustomRacecarWrapper 

# 2. Re-create the environment with rendering enabled
env = gym.make('SingleAgentAustria-v0', render_mode='human')

# 3. Apply the exact same wrapper so the observation shape matches what PPO expects
env = CustomRacecarWrapper(env)

# 4. Load the trained PPO model
model_path = "ppo_racecar_model"
print(f"Loading trained model from {model_path}...")
model = PPO.load(model_path)

# 5. Run the evaluation loop
obs, info = env.reset()
done = False
cumulative_reward = 0

print("Starting simulation. Press Ctrl+C in the terminal to exit.")

while not done:
    # Use the model to predict the action based on current observations
    # deterministic=True ensures the model takes its best calculated actions
    action, _states = model.predict(obs, deterministic=True)
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward
    
    # Check if the episode is finished
    done = terminated or truncated
    
    # Control the simulation speed (PyBullet runs fast by default)
    sleep(0.01) 

print(f"Episode finished! Cumulative Reward: {cumulative_reward:.2f}")
env.close()