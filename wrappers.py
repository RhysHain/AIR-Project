import gymnasium as gym
import numpy as np


class RacecarPPOWrapper(gym.Wrapper):
    """
    Wrapper for racecar_gym so Stable-Baselines3 PPO can train on:
    - flattened LiDAR
    - velocity
    - yaw rate
    - previous action

    Action:
    np.array([speed_or_motor, steering])
    """

    def __init__(self, env, agent_id="A"):
        super().__init__(env)
        self.agent_id = agent_id

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_progress = 0.0

        # We will infer lidar size after first reset, but Racecar Gym default is 1080.
        lidar_size = 1080

        # lidar + velocity(6) + previous_action(2)
        obs_size = lidar_size + 6 + 2

        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        # PPO-friendly action: [speed/motor, steering]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_progress = self._get_progress(info)

        return self._process_obs(obs), info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        env_action = self._format_action(action)

        obs, base_reward, terminated, truncated, info = self.env.step(env_action)

        processed_obs = self._process_obs(obs)
        reward = self._compute_reward(obs, action, terminated, truncated, info)

        self.prev_action = action.copy()

        return processed_obs, reward, terminated, truncated, info

    def _unwrap_agent_obs(self, obs):
        """
        Handles both:
        obs = {'lidar': ..., 'velocity': ...}
        and:
        obs = {'A': {'lidar': ..., 'velocity': ...}}
        """
        if isinstance(obs, dict) and self.agent_id in obs:
            return obs[self.agent_id]
        return obs

    def _unwrap_agent_info(self, info):
        """
        Handles both single-agent and nested multi-agent info/state.
        """
        if isinstance(info, dict) and self.agent_id in info:
            return info[self.agent_id]
        return info

    def _process_obs(self, obs):
        obs = self._unwrap_agent_obs(obs)

        lidar = np.asarray(obs["lidar"], dtype=np.float32).flatten()

        # Replace inf/nan and clip distances.
        lidar = np.nan_to_num(lidar, nan=0.0, posinf=10.0, neginf=0.0)
        lidar = np.clip(lidar, 0.0, 10.0)

        # Normalise lidar roughly to 0..1
        lidar = lidar / 10.0

        velocity = np.asarray(obs["velocity"], dtype=np.float32).flatten()
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.clip(velocity, -10.0, 10.0)

        processed = np.concatenate([
            lidar,
            velocity,
            self.prev_action
        ]).astype(np.float32)

        return processed

    def _format_action(self, action):
        """
        Racecar Gym action may be Dict-based depending on scenario.
        This maps PPO's simple 2-value action into Racecar Gym format.
        """

        raw_space = self.env.action_space

        # If action space is already a simple Box, pass it through.
        if isinstance(raw_space, gym.spaces.Box):
            return action

        # If action space is Dict with motor/speed + steering.
        if isinstance(raw_space, gym.spaces.Dict):
            if "motor" in raw_space.spaces:
                drive_key = "motor"
            elif "speed" in raw_space.spaces:
                drive_key = "speed"
            else:
                drive_key = list(raw_space.spaces.keys())[0]

            return {
                drive_key: np.array([action[0]], dtype=np.float32),
                "steering": np.array([action[1]], dtype=np.float32)
            }

        return action

    def _get_progress(self, info):
        info = self._unwrap_agent_info(info)

        if isinstance(info, dict):
            return float(info.get("progress", self.prev_progress))

        return self.prev_progress

    def _compute_reward(self, obs, action, terminated, truncated, info):
        info_agent = self._unwrap_agent_info(info)

        progress = self._get_progress(info)
        progress_delta = progress - self.prev_progress

        # Handle lap wrap-around: progress goes from near 1 back to 0.
        if progress_delta < -0.5:
            progress_delta += 1.0

        self.prev_progress = progress

        reward = 0.0

        # Main objective: move forward around the lap.
        reward += progress_delta * 1000.0

        # Small alive reward so it does not learn to stop.
        reward += 0.02

        # Encourage forward command, but not too strongly.
        throttle = float(action[0])
        steering = float(action[1])

        if throttle > 0:
            reward += 0.05 * throttle
        else:
            reward -= 0.1

        # Penalise aggressive steering.
        reward -= 0.03 * abs(steering)

        # Penalise sudden action changes.
        reward -= 0.02 * float(np.linalg.norm(action - self.prev_action))

        if isinstance(info_agent, dict):
            if info_agent.get("wall_collision", False):
                reward -= 5.0

            if info_agent.get("wrong_way", False):
                reward -= 2.0

        if terminated:
            reward -= 3.0

        return float(reward)