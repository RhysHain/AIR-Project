import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from train import CustomRacecarWrapper

BASE_MODEL_PATH = "./models/ppo_racecar_model_detection2further"
SAVE_MODEL_PATH = "./models/ppo_racecar_model_detection3"

# Load existing environment
env = gym.make(
    id="SingleAgentAustria-v0",
    render_mode=None,
    scenario="track.yml"
)

env = CustomRacecarWrapper(env)
env = Monitor(env)

# Load existing model
print(f"Loading existing model from {BASE_MODEL_PATH}...")
model = PPO.load(
    BASE_MODEL_PATH,
    env=env,
    device="cpu",
    custom_objects={"ent_coef": 0.015}
)
model.learning_rate = 5e-4
model.lr_schedule = lambda _: 5e-4

print("Continuing training...")
print("Current timesteps trained: (check tensorboard for exact count)")

# Continue training with the original exploration level.
model.learn(
    total_timesteps=1000000,
    tb_log_name="PPO_continue"
)

print("Saving model...")
model.save(SAVE_MODEL_PATH)

env.close()
print("Done!")
