from time import sleep
from pathlib import Path
import numpy as np

import gymnasium as gym
import pybullet

# This import registers the racecar_gym environments
from racecar_gym.envs import gym_api

from stable_baselines3 import PPO

from train import CustomRacecarWrapper, LIDAR_ANGLE, LIDAR_ANGLE_START, LIDAR_FORWARD


STEERING_CONFIG_PATH = Path("racecar_gym/models/vehicles/racecar/racecar.yml")
STEERING_JOINT_INDICES = (0, 2)
STEERING_JOINT_NAMES = ("left_steering_hinge_joint", "right_steering_hinge_joint")


def load_steering_config():
    max_angle = 0.42
    multiplier = 0.5
    if not STEERING_CONFIG_PATH.exists():
        return max_angle, multiplier

    for line in STEERING_CONFIG_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("max_steering_angle:"):
            max_angle = float(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("steering_multiplier:"):
            multiplier = float(stripped.split(":", 1)[1].strip())
    return max_angle, multiplier


def find_steering_body_id():
    for body_id in range(pybullet.getNumBodies()):
        if pybullet.getNumJoints(body_id) <= max(STEERING_JOINT_INDICES):
            continue
        joint_names = tuple(
            pybullet.getJointInfo(body_id, joint_index)[1].decode("utf-8")
            for joint_index in STEERING_JOINT_INDICES
        )
        if joint_names == STEERING_JOINT_NAMES:
            return body_id
    return None


def get_steering_physics(body_id, command, max_angle_rad, multiplier):
    target_rad = -command * max_angle_rad * multiplier
    debug = {
        "target_rad": target_rad,
        "left_rad": None,
        "right_rad": None,
        "avg_rad": None,
        "error_rad": None,
    }
    if body_id is None:
        return debug

    left_rad = pybullet.getJointState(body_id, STEERING_JOINT_INDICES[0])[0]
    right_rad = pybullet.getJointState(body_id, STEERING_JOINT_INDICES[1])[0]
    avg_rad = (left_rad + right_rad) / 2.0
    debug.update(
        left_rad=left_rad,
        right_rad=right_rad,
        avg_rad=avg_rad,
        error_rad=target_rad - avg_rad,
    )
    return debug


env = gym.make(
    id="SingleAgentAustria-v0",
    render_mode="human",
    scenario="track.yml"
)



env = CustomRacecarWrapper(env)

model = PPO.load("./models/ppo_racecar_model_detection3", device="cpu")

obs, info = env.reset()

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
        
done = False
total_reward = 0.0
step_count = 0
previous_debug = None
last_opening_event_step = -999
steering_max_angle_rad, steering_multiplier = load_steering_config()
steering_effective_max_rad = steering_max_angle_rad * steering_multiplier
steering_body_id = find_steering_body_id()


def sector_mean(lidar, angles, center_offset_deg, width_deg):
    center = LIDAR_FORWARD + np.deg2rad(center_offset_deg)
    half_width = np.deg2rad(width_deg) / 2.0
    mask = (angles >= center - half_width) & (angles <= center + half_width)
    if not np.any(mask):
        return float(np.mean(lidar))
    return float(np.mean(lidar[mask]))


def clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def required_turn_steering(turn_urgency, asymmetry):
    turn_strength = abs(turn_urgency * asymmetry)
    if turn_strength <= 0.25:
        return 0.0
    magnitude = float(np.interp(
        turn_strength,
        [0.25, 0.60, 1.00],
        [0.25, 0.55, 0.90],
    ))
    return float(np.sign(asymmetry) * magnitude)


print("=" * 80)
print("TESTING MODEL - Debugging Output")
print("=" * 80)
print(
    "Steering config - "
    f"max_angle: {steering_max_angle_rad:.3f} rad ({np.degrees(steering_max_angle_rad):.1f} deg) | "
    f"multiplier: {steering_multiplier:.2f} | "
    f"command +1.00 target: {-steering_effective_max_rad:.3f} rad ({np.degrees(steering_effective_max_rad):.1f} deg)"
)
if steering_body_id is None:
    print("Steering physics - could not find the racecar steering joints in PyBullet.")
else:
    print(f"Steering physics - body id: {steering_body_id} | joints: {STEERING_JOINT_INDICES}")

while not done:
    step_count += 1
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    # Extract LiDAR data from observation
    # obs = [lidar (1080), pose (6), velocity (6), turn_features (2)] = 1094 total
    lidar = obs[:1080] * 10.0  # Denormalize (was divided by 10)
    turn_urgency = obs[1092]
    asymmetry = obs[1093]
    
    # Calculate wall distances
    n = 1080
    angles = LIDAR_ANGLE_START + (LIDAR_ANGLE * np.arange(n, dtype=np.float32) / n)
    left_mask = (angles >= LIDAR_FORWARD - np.deg2rad(105)) & (angles <= LIDAR_FORWARD - np.deg2rad(75))
    right_mask = (angles >= LIDAR_FORWARD + np.deg2rad(75)) & (angles <= LIDAR_FORWARD + np.deg2rad(105))
    front_mask = (angles >= LIDAR_FORWARD - np.deg2rad(10)) & (angles <= LIDAR_FORWARD + np.deg2rad(10))
    left_wall = np.mean(lidar[left_mask])
    right_wall = np.mean(lidar[right_mask])
    front = lidar[front_mask]
    front_dist = float(np.mean(front))
    min_dist = float(np.min(lidar))

    front_wide = sector_mean(lidar, angles, 0, 45)
    left_front = sector_mean(lidar, angles, -35, 25)
    right_front = sector_mean(lidar, angles, 35, 25)
    left_side = sector_mean(lidar, angles, -90, 30)
    right_side = sector_mean(lidar, angles, 90, 30)

    if previous_debug is None:
        previous_debug = {
            "front": front_dist,
            "left_side": left_side,
            "right_side": right_side,
            "left_front": left_front,
            "right_front": right_front,
        }

    front_drop = max(0.0, previous_debug["front"] - front_dist)
    left_side_jump = max(0.0, left_side - previous_debug["left_side"])
    right_side_jump = max(0.0, right_side - previous_debug["right_side"])
    left_front_jump = max(0.0, left_front - previous_debug["left_front"])
    right_front_jump = max(0.0, right_front - previous_debug["right_front"])

    left_opening = (
        0.45 * max(0.0, left_front - front_wide)
        + 0.35 * left_side_jump
        + 0.20 * left_front_jump
    )
    right_opening = (
        0.45 * max(0.0, right_front - front_wide)
        + 0.35 * right_side_jump
        + 0.20 * right_front_jump
    )
    opening_signal = float(np.clip((right_opening - left_opening) / 3.0, -1.0, 1.0))
    opening_strength = clip01(max(left_opening, right_opening) / 3.0)
    front_pressure = clip01((4.5 - front_dist) / 3.0)
    side_opening_signal = float(np.clip((right_side - left_side) / 4.0, -1.0, 1.0))
    hairpin_signal = float(np.clip(opening_signal + front_pressure * side_opening_signal, -1.0, 1.0))
    hairpin_strength = max(opening_strength, abs(side_opening_signal) * front_pressure)
    turn_required_steering = required_turn_steering(turn_urgency, asymmetry)
    
    raw_speed = float(action[0])
    steering = float(action[1])
    applied_speed = float(env.last_applied_action[0])
    applied_steering = float(env.last_applied_action[1])
    applied_speed_mps = ((applied_speed + 1.0) / 2.0) * 3.5
    steering_physics = get_steering_physics(
        steering_body_id,
        applied_steering,
        steering_max_angle_rad,
        steering_multiplier,
    )
    
    # Progress tracking
    progress = float(info.get("progress", 0.0)) if isinstance(info, dict) else 0.0
    
    # Print periodic updates
    if step_count % 50 == 0:
        print(f"\n[Step {step_count}] Progress: {progress:.3f} | Reward: {reward:+.2f}")
        print(f"  LiDAR - Front: {front_dist:.2f}m | Left: {left_wall:.2f}m | Right: {right_wall:.2f}m | Min: {min_dist:.2f}m")
        print(f"  Signals - Turn Urgency: {turn_urgency:.3f} | Asymmetry: {asymmetry:+.3f}")
        print(f"  TurnDemand - RequiredSteer: {turn_required_steering:+.2f} | Align: {steering * np.sign(turn_required_steering):+.2f}")
        print(f"  Sectors - LF: {left_front:.2f}m | RF: {right_front:.2f}m | LS: {left_side:.2f}m | RS: {right_side:.2f}m | FrontWide: {front_wide:.2f}m")
        print(f"  OpeningCue - L: {left_opening:.2f} | R: {right_opening:.2f} | Signal: {opening_signal:+.2f} | Strength: {opening_strength:.2f} | Align: {steering * opening_signal:+.2f} | FrontDrop: {front_drop:.2f}m")
        print(f"  HairpinCue - SideOpen: {side_opening_signal:+.2f} | Signal: {hairpin_signal:+.2f} | Strength: {hairpin_strength:.2f} | Align: {steering * hairpin_signal:+.2f} | FrontPressure: {front_pressure:.2f}")
        print(f"  LineGuide - Phase: {env.last_line_phase} | TargetSteer: {env.last_line_target_steering:+.2f} | LineLimit: {env.last_line_speed_limit_mps:.2f} m/s | Align: {env.last_applied_action[1] * env.last_line_target_steering:+.2f}")
        print(f"  Raw Action - Speed: {raw_speed:+.2f} | Steering: {steering:+.2f}")
        print(f"  Applied - SpeedCmd: {applied_speed:+.2f} ({applied_speed_mps:.2f} m/s) | Target: {env.last_target_speed_mps:.2f} m/s | Limit: {env.last_speed_limit_mps:.2f} m/s | Closing: {env.last_front_closing:.2f}m | Steering: {env.last_applied_action[1]:+.2f} | LeftSide: {env.last_left_side_dist:.2f}m | RightSide: {env.last_right_side_dist:.2f}m")
        if steering_physics["avg_rad"] is None:
            print(f"  SteeringPhysics - Target: {steering_physics['target_rad']:+.3f} rad ({np.degrees(steering_physics['target_rad']):+.1f} deg) | Actual: unavailable")
        else:
            print(f"  SteeringPhysics - Target: {steering_physics['target_rad']:+.3f} rad ({np.degrees(steering_physics['target_rad']):+.1f} deg) | ActualAvg: {steering_physics['avg_rad']:+.3f} rad ({np.degrees(steering_physics['avg_rad']):+.1f} deg) | L/R: {steering_physics['left_rad']:+.3f}/{steering_physics['right_rad']:+.3f} | Error: {steering_physics['error_rad']:+.3f}")

    if (
        hairpin_strength > 0.25
        and front_pressure > 0.15
        and step_count - last_opening_event_step > 25
    ):
        turn_name = "RIGHT" if hairpin_signal > 0 else "LEFT"
        print(f"\n  >>> Hairpin event: {turn_name} cue at step {step_count} | Progress: {progress:.3f} | Front: {front_dist:.2f}m | Cue: {hairpin_signal:+.2f} | Steering: {steering:+.2f} | Align: {steering * hairpin_signal:+.2f}")
        last_opening_event_step = step_count

    total_reward += reward
    done = terminated or truncated
    
    # If crash or termination, show why
    if done:
        print(f"\n{'!'*80}")
        print(f"EPISODE ENDED at step {step_count}")
        print(f"{'!'*80}")
        print(f"  Terminated: {terminated} (lap complete/collision) | Truncated: {truncated} (time limit)")
        print(f"  Final Progress: {progress:.3f}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Final LiDAR - Front: {front_dist:.2f}m | Min Distance: {min_dist:.2f}m")
        print(f"  Final Turn Signal - Urgency: {turn_urgency:.3f} | Asymmetry: {asymmetry:+.3f}")
        print(f"  Final TurnDemand - RequiredSteer: {turn_required_steering:+.2f} | Align: {steering * np.sign(turn_required_steering):+.2f}")
        print(f"  Final OpeningCue - L: {left_opening:.2f} | R: {right_opening:.2f} | Signal: {opening_signal:+.2f} | Strength: {opening_strength:.2f} | Align: {steering * opening_signal:+.2f}")
        print(f"  Final HairpinCue - SideOpen: {side_opening_signal:+.2f} | Signal: {hairpin_signal:+.2f} | Strength: {hairpin_strength:.2f} | Align: {steering * hairpin_signal:+.2f}")
        print(f"  Final LineGuide - Phase: {env.last_line_phase} | TargetSteer: {env.last_line_target_steering:+.2f} | LineLimit: {env.last_line_speed_limit_mps:.2f} m/s | Align: {env.last_applied_action[1] * env.last_line_target_steering:+.2f}")
        print(f"  Final Raw Action - Speed: {raw_speed:+.2f} | Steering: {steering:+.2f}")
        print(f"  Final Applied - SpeedCmd: {applied_speed:+.2f} ({applied_speed_mps:.2f} m/s) | Target: {env.last_target_speed_mps:.2f} m/s | Limit: {env.last_speed_limit_mps:.2f} m/s | Closing: {env.last_front_closing:.2f}m | Steering: {env.last_applied_action[1]:+.2f} | LeftSide: {env.last_left_side_dist:.2f}m | RightSide: {env.last_right_side_dist:.2f}m")
        if steering_physics["avg_rad"] is None:
            print(f"  Final SteeringPhysics - Target: {steering_physics['target_rad']:+.3f} rad ({np.degrees(steering_physics['target_rad']):+.1f} deg) | Actual: unavailable")
        else:
            print(f"  Final SteeringPhysics - Target: {steering_physics['target_rad']:+.3f} rad ({np.degrees(steering_physics['target_rad']):+.1f} deg) | ActualAvg: {steering_physics['avg_rad']:+.3f} rad ({np.degrees(steering_physics['avg_rad']):+.1f} deg) | L/R: {steering_physics['left_rad']:+.3f}/{steering_physics['right_rad']:+.3f} | Error: {steering_physics['error_rad']:+.3f}")
        if min_dist < 0.15:
            print(f"  CRASH: Hit a wall! (min_dist = {min_dist:.3f}m)")

    previous_debug = {
        "front": front_dist,
        "left_side": left_side,
        "right_side": right_side,
        "left_front": left_front,
        "right_front": right_front,
    }

    sleep(0.01)

print(f"\n{'='*80}")
print(f"Episode finished. Total steps: {step_count} | Total reward: {total_reward:.2f}")
print(f"{'='*80}\n")

env.close()
