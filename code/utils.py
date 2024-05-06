import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from minigrid.core.world_object import Goal, Key, Door
import cv2

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def step_cost(action):
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    return 1  # the cost of action


def step(env, action):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        0: env.unwrapped.actions.forward,
        1: env.unwrapped.actions.left,
        2: env.unwrapped.actions.right,
        3: env.unwrapped.actions.pickup,
        4: env.unwrapped.actions.toggle,
    }

    (obs, reward, terminated, truncated, info) = env.step(actions[action])
    return step_cost(action), terminated


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    with open(path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height, 
        "width": env.unwrapped.width, 
        "init_agent_pos": env.unwrapped.agent_pos, 
        "init_agent_dir": env.unwrapped.dir_vec,
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.unwrapped.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Door):
                info["door_pos"] = np.array([j, i])
            elif isinstance(env.unwrapped.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info


def load_random_env(env_folder):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    # at the beginning of load_random_env(env_folder) defined in utils.py, change the line of env_list to
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder) if env_file.endswith(".env")]
    env_path = random.choice(env_list)
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.unwrapped.height,
        "width": env.unwrapped.width,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info, env_path


def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq, env, path="./gif/doorkey.gif"):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    with imageio.get_writer(path, mode="I", duration=0.8) as writer:
        img = env.render()
        writer.append_data(img)
        for act in seq:
            step(env, act)              # swap these two lines, here I show the code after swapping them
            img = env.render()     # swap these two lines
            writer.append_data(img)
    print(f"GIF is written to {path}")
    return
    
def draw_video_from_seq(seq, env, env_name):
    """
    Save video with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0, 0, 0, 0] or [MF, MF, MF, MF]

    env:
        The doorkey environment

    env_name:
        The name of the environment, used for saving the video file.
    """
    # First render to establish frame size
    initial_img = env.render()
    if initial_img.shape[2] == 3:
        initial_img = cv2.cvtColor(initial_img, cv2.COLOR_RGB2BGR)

    height, width, layers = initial_img.shape
    path = f"./video/{env_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 5, (width, height))

    # Write the first image
    out.write(initial_img)

    for act in seq:
        step(env, act)
        img = env.render()
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out.write(img)

    out.release()
    print(f"Video is written to {path}")


def draw_component_img_from_seq(seq, env, env_name):
    """
    Save a single composite image containing all frames of the given action sequence laid out side by side,
    with a maximum of 8 frames per row and a white gap between frames.
    --------------------------------------------------------------------------------
    seq:
        Action sequence, e.g., [MF, TL, TR, PK, UD]

    env:
        The environment used to generate the frames.

    env_name:
        The name of the environment, used for naming the saved image file.
    """
    initial_img = env.render()
    
    initial_img_bgr = cv2.cvtColor(initial_img, cv2.COLOR_RGB2BGR)
    height, width, channels = initial_img_bgr.shape

    # Set the maximum number of frames per row
    max_frames_per_row = 8
    gap = 10  # Space between frames

    # Calculate the number of rows needed
    num_frames = len(seq) + 1
    num_rows = (num_frames + max_frames_per_row - 1) // max_frames_per_row  # Ceiling division

    # Calculate the dimensions of the composite image
    row_width = width * max_frames_per_row + gap * (max_frames_per_row - 1)
    total_width = min(row_width, width * num_frames + gap * (num_frames - 1))  # Adjust for last row if fewer frames
    total_height = height * num_rows + gap * (num_rows - 1)

    # Create a white blank image to hold all frames
    composite_image = np.full((total_height, total_width, channels), 255, dtype=np.uint8)

    # Place the initial frame
    x_offset = 0
    y_offset = 0
    composite_image[y_offset:y_offset + height, x_offset:x_offset + width] = initial_img_bgr
    x_offset += width + gap  # Move the offset for the next frame

    frame_count = 1

    for act in seq:
        step(env, act)
        frame = env.render()

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame_count % max_frames_per_row == 0:
            x_offset = 0  # Reset to start of the line
            y_offset += height + gap  # Move down a row
        # Place the frame in the composite image
        composite_image[y_offset:y_offset + height, x_offset:x_offset + width] = frame_bgr
        x_offset += width + gap  # Move the offset for the next frame
        frame_count += 1

    # Save the resulting composite image
    save_path = f"./img/{env_name}_composite.png"
    cv2.imwrite(save_path, composite_image)
    print(f"Composite image is saved to {save_path}")