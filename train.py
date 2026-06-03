import gymnasium as gym
import numpy as np
import pybullet
import cv2

# This import registers the racecar_gym environments
from racecar_gym.envs import gym_api

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from ultralytics import YOLO

OBS_LIDAR_RANGE = 10.0
SENSOR_LIDAR_RANGE = 15.0
SPEED_ACTUATOR_MAX_MPS = 3.5
MAX_TARGET_SPEED_MPS = 2.6
STEERING_GAIN = 1.0
LIDAR_ANGLE_START = -2.36 + (np.pi / 2.0)
LIDAR_ANGLE = 4.71
LIDAR_FORWARD = np.pi / 2.0


class CustomRacecarWrapper(gym.Wrapper):
    def __init__(self, env, yolo_model_path="detection_model/runs/detect/signs_v1-7/weights/best.pt"):
        super().__init__(env)

        # Load YOLO model for sign detection
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"YOLO model loaded from {yolo_model_path}")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}. Running without detection.")
            self.yolo_model = None

        # Expanded observation space:
        # LiDAR (1080) + Pose (6) + Velocity (6) + Turn features (2) + Detection (8) = 1102
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1102,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_progress = 0.0
        self.last_progress = 0.0
        self.previous_steering = 0.0
        self.crash_counter = 0

        self.last_turn_urgency = 0.0
        self.last_asymmetry = 0.0
        self.last_front_dist = SENSOR_LIDAR_RANGE
        self.last_min_dist = SENSOR_LIDAR_RANGE
        self.last_left_side_dist = SENSOR_LIDAR_RANGE
        self.last_right_side_dist = SENSOR_LIDAR_RANGE
        self.previous_nearest_side_dist = SENSOR_LIDAR_RANGE
        self.previous_front_dist = SENSOR_LIDAR_RANGE
        self.last_front_closing = 0.0
        self.last_speed_limit_mps = MAX_TARGET_SPEED_MPS
        self.last_target_speed_mps = 0.0
        self.last_line_target_steering = 0.0
        self.last_line_speed_limit_mps = MAX_TARGET_SPEED_MPS
        self.last_line_phase = "none"
        self.last_applied_action = np.zeros(2, dtype=np.float32)
        self.last_detection = np.zeros(8, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_progress = self._get_progress(info, default=0.0)
        self.last_progress = self.previous_progress
        self.previous_steering = 0.0
        self.crash_counter = 0

        # Extract camera image and run YOLO detection for initial observation
        camera_image = None
        if isinstance(obs, dict):
            for key in ["rgb_camera_front", "rgb_camera", "camera_image", "image"]:
                if key in obs:
                    camera_image = obs[key]
                    break
        
        if camera_image is not None:
            self.last_detection = self._get_detection_features(camera_image)
        else:
            self.last_detection = np.zeros(8, dtype=np.float32)

        processed_obs = self._process_observation(obs)
        self.last_turn_urgency = float(processed_obs[-2])
        self.last_asymmetry = float(processed_obs[-1])
        self._update_clearance_state(obs)
        self.previous_nearest_side_dist = min(self.last_left_side_dist, self.last_right_side_dist)
        self.previous_front_dist = self.last_front_dist
        self.last_front_closing = 0.0
        self.last_speed_limit_mps = MAX_TARGET_SPEED_MPS
        self.last_target_speed_mps = 0.0
        self.last_line_target_steering = 0.0
        self.last_line_speed_limit_mps = MAX_TARGET_SPEED_MPS
        self.last_line_phase = "none"
        self.last_applied_action = np.zeros(2, dtype=np.float32)

        return processed_obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        raw_speed = float(action[0])
        steering = float(np.clip(action[1] * STEERING_GAIN, -1.0, 1.0))

        throttle = float(np.clip((raw_speed + 1.0) / 2.0, 0.0, 1.0))
        target_speed_mps = MAX_TARGET_SPEED_MPS * throttle

        speed_limit_mps = self._speed_limit_from_lidar(steering)
        target_speed_mps = min(target_speed_mps, speed_limit_mps)

        target_speed_mps = float(np.clip(target_speed_mps, 0.0, SPEED_ACTUATOR_MAX_MPS))
        speed_command = self._speed_mps_to_command(target_speed_mps)

        env_action = {
            # Do not drive the motor actuator at the same time. The speed
            # actuator is the actual throttle/brake path here.
            "motor": np.array([0.0], dtype=np.float32),
            "speed": np.array([speed_command], dtype=np.float32),
            "steering": np.array([steering], dtype=np.float32),
        }

        obs, base_reward, terminated, truncated, info = self.env.step(env_action)

        # Extract camera image and run YOLO detection
        camera_image = None
        if isinstance(obs, dict):
            # Try common camera observation keys
            for key in ["hd_camera", "rgb_camera_front", "rgb_camera", "camera_image", "image"]:
                if key in obs:
                    camera_image = obs[key]
                    break
        
        if camera_image is not None:
            self.last_detection = self._get_detection_features(camera_image)

        action_used = np.array([speed_command, steering], dtype=np.float32)
        crashed = self._detect_crash(obs, info)
        if crashed:
            terminated = True

        reward = self._compute_reward(obs, action_used, terminated, info, crashed)
        self.last_progress = self._get_progress(info)

        processed_obs = self._process_observation(obs)
        self.last_turn_urgency = float(processed_obs[-2])
        self.last_asymmetry = float(processed_obs[-1])
        self.previous_front_dist = self.last_front_dist
        self._update_clearance_state(obs)
        self.last_front_closing = max(0.0, self.previous_front_dist - self.last_front_dist)
        self.previous_nearest_side_dist = min(self.last_left_side_dist, self.last_right_side_dist)
        self.last_speed_limit_mps = speed_limit_mps
        self.last_target_speed_mps = target_speed_mps
        self.last_applied_action = action_used.copy()
        self.previous_action = action_used.copy()

        return processed_obs, reward, terminated, truncated, info

    def _speed_mps_to_command(self, speed_mps):
        return float(np.clip((speed_mps / SPEED_ACTUATOR_MAX_MPS) * 2.0 - 1.0, -1.0, 1.0))

    def _command_to_speed_mps(self, speed_command):
        return ((speed_command + 1.0) / 2.0) * SPEED_ACTUATOR_MAX_MPS

    def _racing_line_guide(self, progress):
        progress = progress % 1.0
        target_steering = 0.0
        speed_limit_mps = MAX_TARGET_SPEED_MPS
        phase = "none"

        # Austria's second right hairpin. The LiDAR sees the opening, but only
        # after the car is already committed, so this acts like a pace note:
        # brake and begin rotating before the wall is directly ahead.
        if 0.300 <= progress <= 0.365:
            turn_in = float(np.clip((progress - 0.300) / 0.030, 0.0, 1.0))
            exit_fade = float(np.clip((0.365 - progress) / 0.020, 0.0, 1.0))
            strength = min(turn_in, exit_fade)
            target_steering = 0.35 + 0.65 * strength

            if progress < 0.315:
                speed_limit_mps = 1.25
            elif progress < 0.335:
                speed_limit_mps = 0.85
            else:
                speed_limit_mps = 0.45

            phase = "right_hairpin_2"

        # Later Austria left hairpin. The failure log showed the car getting a
        # strong left cue around progress 0.62-0.64 but continuing almost
        # straight, so give training an explicit left-turn pace note here too.
        elif 0.600 <= progress <= 0.660:
            turn_in = float(np.clip((progress - 0.600) / 0.030, 0.0, 1.0))
            exit_fade = float(np.clip((0.660 - progress) / 0.020, 0.0, 1.0))
            strength = min(turn_in, exit_fade)
            target_steering = -(0.35 + 0.65 * strength)

            if progress < 0.615:
                speed_limit_mps = 1.10
            elif progress < 0.630:
                speed_limit_mps = 0.70
            else:
                speed_limit_mps = 0.35

            phase = "left_hairpin_3"

        return target_steering, speed_limit_mps, phase

    def _speed_limit_from_lidar(self, steering):
        line_target, line_limit, line_phase = self._racing_line_guide(self.last_progress)
        self.last_line_target_steering = line_target
        self.last_line_speed_limit_mps = line_limit
        self.last_line_phase = line_phase

        front_limit = float(np.interp(
            self.last_front_dist,
            [0.45, 0.75, 1.20, 2.00, 3.50],
            [0.20, 0.45, 0.90, 1.70, MAX_TARGET_SPEED_MPS],
        ))

        min_limit = float(np.interp(
            self.last_min_dist,
            [0.25, 0.40, 0.60, 0.90],
            [0.20, 0.55, 1.20, MAX_TARGET_SPEED_MPS],
        ))

        turn_limit = float(np.interp(
            self.last_turn_urgency,
            [0.00, 0.25, 0.55, 0.85, 1.00],
            [MAX_TARGET_SPEED_MPS, 2.00, 1.10, 0.55, 0.35],
        ))

        approach_limit = MAX_TARGET_SPEED_MPS
        if self.last_turn_urgency > 0.12 or abs(self.last_asymmetry) > 0.18:
            approach_limit = float(np.interp(
                self.last_front_dist,
                [0.75, 1.20, 2.00, 3.00, 4.50],
                [0.25, 0.35, 0.60, 1.05, 1.80],
            ))

        steering_limit = float(np.interp(
            abs(steering),
            [0.00, 0.35, 0.70, 1.00],
            [MAX_TARGET_SPEED_MPS, 2.00, 1.00, 0.50],
        ))

        closing_limit = MAX_TARGET_SPEED_MPS
        if self.last_front_closing > 0.02:
            closing_limit = float(np.interp(
                self.last_front_closing,
                [0.02, 0.08, 0.16],
                [1.40, 0.80, 0.35],
            ))

        nearest_side = min(self.last_left_side_dist, self.last_right_side_dist)
        side_limit = MAX_TARGET_SPEED_MPS
        if nearest_side < 1.20:
            side_limit = float(np.interp(
                nearest_side,
                [0.35, 0.55, 0.75, 1.20],
                [0.35, 0.90, 1.60, MAX_TARGET_SPEED_MPS],
            ))

        speed_limit = min(
            front_limit,
            min_limit,
            turn_limit,
            approach_limit,
            line_limit,
            steering_limit,
            closing_limit,
            side_limit,
        )
        return float(np.clip(speed_limit, 0.20, MAX_TARGET_SPEED_MPS))

    def _process_observation(self, obs):
        lidar = np.asarray(obs["lidar"], dtype=np.float32).flatten()
        pose = np.asarray(obs["pose"], dtype=np.float32).flatten()
        velocity = np.asarray(obs["velocity"], dtype=np.float32).flatten()

        lidar = np.nan_to_num(lidar, nan=0.0, posinf=OBS_LIDAR_RANGE, neginf=0.0)
        lidar = np.clip(lidar, 0.0, OBS_LIDAR_RANGE)
        lidar = lidar / OBS_LIDAR_RANGE

        pose = np.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

        pose = np.clip(pose, -10.0, 10.0)
        velocity = np.clip(velocity, -10.0, 10.0)

        turn_urgency, asymmetry, _ = self._track_features(lidar, normalized=True)
        turn_features = np.array([turn_urgency, asymmetry], dtype=np.float32)
        
        # Use cached detection features from previous step
        detection_features = self.last_detection.copy()

        return np.concatenate([lidar, pose, velocity, turn_features, detection_features]).astype(np.float32)

    def _get_detection_features(self, image):
        """
        Get YOLO detections and format as feature vector.
        Returns: [stop_prob, slow_prob, go_prob, confidence, bbox_center_x, bbox_center_y, bbox_width, bbox_height]
        All values normalized to [-1, 1] range.
        """
        if self.yolo_model is None or image is None:
            return np.zeros(8, dtype=np.float32)
        
        try:
            # Run YOLO inference
            results = self.yolo_model.predict(image, conf=0.4, verbose=False)

                    # Display detections
            annotated_frame = results[0].plot()

            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("YOLO Detections", annotated_frame)
            cv2.waitKey(1)
            
            # Initialize with "no detection" (GO signal by default)
            detection = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            if results and len(results[0].boxes) > 0:
                # Get largest bounding box (closest sign)
                boxes = results[0].boxes
                best_idx = int(np.argmax(boxes.conf.cpu().numpy()))  # confidence-based 
                
                # Get largest bounding box instead
                # boxes_data = boxes.xyxy.cpu().numpy()
                # areas = (boxes_data[:, 2] - boxes_data[:, 0]) * (boxes_data[:, 3] - boxes_data[:, 1])
                # best_idx = int(np.argmax(areas))
                box = boxes[best_idx]
                
                cls_id = int(box.cls[0])  # 0=STOP, 1=SLOW, 2=GO
                conf = float(box.conf[0])
                
                # Normalize bounding box to [-1, 1] range
                h, w = image.shape[0], image.shape[1]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                bbox_center_x = ((x1 + x2) / 2.0 - w / 2.0) / (w / 2.0)  # normalize to [-1, 1]
                bbox_center_y = ((y1 + y2) / 2.0 - h / 2.0) / (h / 2.0)  # normalize to [-1, 1]
                bbox_w = (x2 - x1) / w  # normalize to [0, 1]
                bbox_h = (y2 - y1) / h  # normalize to [0, 1]
                
                # One-hot encode sign type
                probs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                probs[cls_id] = conf
                
                detection = np.array([
                    probs[0],           # STOP probability (0-1)
                    probs[1],           # SLOW probability (0-1)
                    probs[2],           # GO probability (0-1)
                    conf,               # confidence (0-1)
                    np.clip(bbox_center_x, -1.0, 1.0),  # x position (-1 to 1)
                    np.clip(bbox_center_y, -1.0, 1.0),  # y position (-1 to 1)
                    bbox_w,             # width (0-1)
                    bbox_h              # height (0-1)
                ], dtype=np.float32)
        except Exception as e:
            print(f"Warning: YOLO inference failed: {e}")
            detection = np.zeros(8, dtype=np.float32)
        
        return detection

    def _detect_crash(self, obs, info):
        if isinstance(info, dict):
            if info.get("wall_collision", False) or info.get("collision", False):
                return True

        lidar = np.asarray(obs["lidar"], dtype=np.float32).flatten()
        lidar = np.nan_to_num(lidar, nan=SENSOR_LIDAR_RANGE, posinf=SENSOR_LIDAR_RANGE, neginf=0.0)
        lidar = np.clip(lidar, 0.0, SENSOR_LIDAR_RANGE)

        if float(np.min(lidar)) < 0.12:
            self.crash_counter += 1
        else:
            self.crash_counter = 0

        return self.crash_counter >= 3

    def _lidar_angles(self, n):
        return LIDAR_ANGLE_START + (LIDAR_ANGLE * np.arange(n, dtype=np.float32) / n)

    def _sector_mean(self, lidar, center, width):
        angles = self._lidar_angles(len(lidar))
        half_width = width / 2.0
        mask = (angles >= center - half_width) & (angles <= center + half_width)
        if not np.any(mask):
            return float(np.mean(lidar))
        return float(np.mean(lidar[mask]))

    def _track_features(self, lidar, normalized=False):
        if normalized:
            lidar = lidar * OBS_LIDAR_RANGE
        scale = SENSOR_LIDAR_RANGE

        front = self._sector_mean(lidar, LIDAR_FORWARD, np.deg2rad(20))
        right_front = self._sector_mean(lidar, LIDAR_FORWARD + np.deg2rad(35), np.deg2rad(25))
        left_front = self._sector_mean(lidar, LIDAR_FORWARD - np.deg2rad(35), np.deg2rad(25))
        left_side = self._sector_mean(lidar, LIDAR_FORWARD - np.deg2rad(90), np.deg2rad(30))
        right_side = self._sector_mean(lidar, LIDAR_FORWARD + np.deg2rad(90), np.deg2rad(30))

        direction_signal = np.clip((right_front - left_front) / (0.22 * scale), -1.0, 1.0)
        front_approach = max(0.0, ((0.37 * scale) - front) / (0.37 * scale))
        front_blocked = max(0.0, ((0.24 * scale) - front) / (0.24 * scale))
        close_wall = min(left_side, right_side)
        side_danger = max(0.0, ((0.05 * scale) - close_wall) / (0.05 * scale))

        # Positive asymmetry means steer right. Positive steering turns right.
        side_opening = np.clip((right_side - left_side) / (0.25 * scale), -1.0, 1.0)
        side_avoidance = np.clip((right_side - left_side) / (0.45 * scale), -1.0, 1.0)
        asymmetry = float(np.clip(
            0.8 * direction_signal
            + (0.75 * front_approach + 0.25 * front_blocked) * side_opening
            + 0.15 * side_avoidance,
            -1.0,
            1.0,
        ))
        direction_urgency = abs(direction_signal) * (0.20 + 0.80 * front_approach)
        opening_urgency = abs(side_opening) * front_approach
        turn_urgency = float(np.clip(
            direction_urgency
            + front_blocked
            + 0.35 * front_approach
            + 0.45 * opening_urgency
            + 0.3 * side_danger,
            0.0,
            1.0,
        ))

        return turn_urgency, asymmetry, {
            "front": front,
            "left_front": left_front,
            "right_front": right_front,
            "left_side": left_side,
            "right_side": right_side,
        }

    def _update_clearance_state(self, obs):
        lidar = np.asarray(obs["lidar"], dtype=np.float32).flatten()
        lidar = np.nan_to_num(lidar, nan=SENSOR_LIDAR_RANGE, posinf=SENSOR_LIDAR_RANGE, neginf=0.0)
        lidar = np.clip(lidar, 0.0, SENSOR_LIDAR_RANGE)
        _, _, sectors = self._track_features(lidar, normalized=False)
        self.last_front_dist = sectors["front"]
        self.last_left_side_dist = sectors["left_side"]
        self.last_right_side_dist = sectors["right_side"]
        self.last_min_dist = float(np.min(lidar))

    def _get_progress(self, info, default=None):
        if isinstance(info, dict):
            return float(info.get("progress", self.previous_progress if default is None else default))
        return self.previous_progress if default is None else default

    def _compute_reward(self, obs, action, terminated, info, crashed):
        speed_command = float(action[0])
        steering = float(action[1])
        target_speed_mps = self._command_to_speed_mps(speed_command)

        lidar = np.asarray(obs["lidar"], dtype=np.float32).flatten()
        lidar = np.nan_to_num(lidar, nan=0.0, posinf=SENSOR_LIDAR_RANGE, neginf=0.0)
        lidar = np.clip(lidar, 0.0, SENSOR_LIDAR_RANGE)

        turn_urgency, asymmetry, sectors = self._track_features(lidar, normalized=False)
        progress = self._get_progress(info)
        line_target, _, line_phase = self._racing_line_guide(progress)
        front_dist = sectors["front"]
        left_front_dist = sectors["left_front"]
        right_front_dist = sectors["right_front"]
        left_side_dist = sectors["left_side"]
        right_side_dist = sectors["right_side"]
        min_dist = float(np.min(lidar))
        nearest_side_dist = min(left_side_dist, right_side_dist)

        reward = 0.0
        reward += 0.05 * target_speed_mps

        if target_speed_mps < 0.1:
            reward -= 0.5

        reward += 0.02 * min(front_dist, 3.0)
        reward += 0.03 * min(nearest_side_dist, 2.0)

        side_balance_error = abs(left_front_dist - right_front_dist) / SENSOR_LIDAR_RANGE
        reward -= 0.10 * (1.0 - turn_urgency) * side_balance_error

        wall_center_error = abs(left_side_dist - right_side_dist) / (left_side_dist + right_side_dist + 1e-6)
        reward -= 0.12 * (1.0 - turn_urgency) * wall_center_error

        if min_dist < 0.25:
            reward -= 3.0
        elif min_dist < 0.5:
            reward -= 1.0

        if nearest_side_dist < 0.45:
            reward -= 2.0 * (0.45 - nearest_side_dist)
        if nearest_side_dist < 0.75:
            reward -= 1.5 * (0.75 - nearest_side_dist)
        if nearest_side_dist < 1.05:
            reward -= 0.8 * (1.05 - nearest_side_dist)

        side_clearance_delta = nearest_side_dist - self.previous_nearest_side_dist
        if nearest_side_dist < 1.20:
            reward += 0.8 * np.clip(side_clearance_delta, -0.25, 0.25)

        side_avoidance = np.clip((right_side_dist - left_side_dist) / 1.5, -1.0, 1.0)
        wall_pressure = np.clip((1.20 - nearest_side_dist) / 1.20, 0.0, 1.0)
        reward += 0.35 * wall_pressure * steering * side_avoidance

        steering_change = abs(steering - self.previous_steering)
        approach_pressure = np.clip((5.2 - front_dist) / 3.2, 0.0, 1.0)
        if approach_pressure > 0.0:
            steering_alignment = steering * asymmetry
            reward += 0.55 * approach_pressure * steering_alignment
            reward -= 0.30 * approach_pressure * max(0.0, -steering_alignment)

        if line_phase != "none":
            line_strength = abs(line_target)
            line_sign = float(np.sign(line_target))
            signed_steering = steering * line_sign
            required_steering = max(0.25, 0.85 * line_strength)

            reward += 2.0 * line_strength * signed_steering
            reward -= 2.5 * line_strength * max(0.0, required_steering - signed_steering)
            reward -= 1.2 * line_strength * max(0.0, -signed_steering)

        if turn_urgency > 0.2:
            turn_alignment = steering * asymmetry
            turn_strength = turn_urgency * abs(asymmetry)
            required_steering = float(np.interp(
                turn_strength,
                [0.25, 0.60, 1.00],
                [0.25, 0.55, 0.90],
            ))

            reward += 1.2 * turn_urgency * turn_alignment
            if turn_strength > 0.25:
                reward -= 1.8 * turn_strength * max(0.0, required_steering - turn_alignment)
                reward -= 1.0 * turn_strength * max(0.0, -turn_alignment)

            reward -= 0.005 * abs(steering)
            reward -= 0.02 * steering_change
        else:
            reward -= 0.12 * abs(steering)
            reward -= 0.15 * steering_change

        action_change = np.linalg.norm(action - self.previous_action)
        reward -= 0.03 * action_change

        if abs(steering) > 0.6 and target_speed_mps > 1.0:
            reward -= 0.25 * target_speed_mps * abs(steering)

        # Sign compliance rewards
        stop_prob = self.last_detection[0]
        slow_prob = self.last_detection[1]
        go_prob = self.last_detection[2]

        if stop_prob > 0.7:  # STOP sign detected
            if target_speed_mps < 0.1:
                reward += 20.0  # reward near-stop
            else:
                reward -= 200.0  # penalize speeding past STOP
        elif slow_prob > 0.4:  # SLOW sign detected
            if target_speed_mps < 1.5:
                reward += 0.5  # reward obeying SLOW
            else:
                reward -= 1.0  # penalize ignoring SLOW
        elif go_prob > 0.6:  # GO sign detected
            if target_speed_mps > 1.5:
                reward += 0.3  # reward maintaining speed on GO
            else:
                reward -= 0.5  # penalize going too slow on GO

        if isinstance(info, dict) and "progress" in info:
            progress_delta = progress - self.previous_progress
            if progress_delta < -0.5:
                progress_delta += 1.0

            reward += 300.0 * progress_delta
            self.previous_progress = progress

        if isinstance(info, dict) and info.get("wrong_way", False):
            reward -= 5.0

        if crashed:
            reward -= 20.0

        if terminated:
            reward -= 5.0

        self.previous_steering = steering
        return float(reward)


if __name__ == "__main__":
    env = gym.make(
        id="SingleAgentAustria-v0",
        render_mode=None,
        scenario="track.yml"
    )

    env.reset()

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
            'position': [9, -0.3, 0.5],
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
            'position': [16, -18, 0.5],
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
            'position': [1.3, -14.5, 0.5],
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
            'position': [8.7, -13.4, 0.5],
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
            'position': [-6, -6.2, 0.5],
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
            'position': [-7.2, -1.2, 0.5],
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
            texture_id = pybullet.loadTexture(sign_cfg['texture'])
            
            # Create visual shape
            visual_shape = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=sign_cfg['obj'],
                meshScale=sign_cfg['scale'],       # Now using scale from config
                rgbaColor=[1, 1, 1, 1]
            )
            
            # Create collision shape
            collision_shape = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=sign_cfg['collision_size']
            )
            
            # Create the sign
            sign_id = pybullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=sign_cfg['position'],
                baseOrientation=pybullet.getQuaternionFromEuler(sign_cfg['rotation'])
            )
            
            # Apply texture
            pybullet.changeVisualShape(sign_id, -1, textureUniqueId=texture_id)
            
            # Debuggin Purposes
            print(f"Loaded {sign_cfg['name']} at {sign_cfg['position']}")
            
        except Exception as e:
            print(f"Failed to load {sign_cfg['name']}: {e}")

    env = CustomRacecarWrapper(env)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        learning_rate=5e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        device="cpu"
    )

    model.learn(total_timesteps=300_000)
    model.save("./models/ppo_racecar_model_detection2")
    env.close()
