from utils import *
from example import example_use_of_gym_env

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door




def partA():
    env_folder = "envs/known_envs/"
    optimal_policies = {
        "doorkey-5x5-normal.env": [TL, TL, PK, TR, UD, MF, MF, TR, MF],
        "doorkey-6x6-direct.env": [MF, MF, TR, MF, MF],
        "doorkey-6x6-normal.env": [TL, MF, PK, TL, MF, TL, MF, TR, UD, MF, MF, TR, MF],
        "doorkey-6x6-shortcut.env": [PK, TL, TL, UD, MF, MF],
        "doorkey-8x8-direct.env": [MF, TL, MF, MF, MF, TL, MF],
        "doorkey-8x8-normal.env": [TR, MF, TL, MF, TR, MF, MF, MF, PK, TL, TL, MF, MF, MF, TR, UD, MF, MF, MF, TR, MF,
                                   MF, MF],
        "doorkey-8x8-shortcut.env": [TR, MF, TR, PK, TL, UD, MF, MF],
        "example-8x8.env": [TR, MF, PK, TL, UD, MF, MF, MF, MF, TR, MF]
    }
    for i, (env_name, seq) in enumerate(optimal_policies.items(), 1):
        env_path = env_folder + env_name
        env, info = load_env(env_path)
        draw_component_img_from_seq(seq, env, env_name)


def partB():
    """solve part B here"""

    seq = {
        1: [MF, MF, MF, TR, MF, MF, TL, MF],
        2: [MF, MF, MF, TR, MF, MF, TL, MF],
        3: [TR, MF, MF, TL, MF, MF, MF, MF],
        4: [MF, MF, MF, MF, TL, MF, PK, TL, MF, TL, MF, UD, MF, MF, TL, MF],
        5: [TR, MF, MF, MF, TL, MF, MF],
        6: [MF, MF, MF, TR, MF, MF, MF, TR, MF],
        7: [TR, MF, MF, MF, TL, MF, MF],
        8: [MF, MF, MF, MF, TL, MF, PK, TL, MF, TL, MF, UD, MF, MF, MF, TR, MF],
        9: [TR, MF, MF, TR, MF],
        10: [MF, MF, MF, TR, MF, MF, TR, MF, MF, MF, MF],
        11: [TR, MF, MF, TR, MF],
        12: [MF, MF, MF, MF, TL, MF, PK, TL, MF, MF, MF, MF, TL, MF, UD, MF, MF, TR, MF],
        13: [MF, MF, MF, TR, MF, MF, TL, MF],
        14: [MF, MF, MF, TR, MF, MF, TL, MF],
        15: [TR, MF, MF, TL, MF, MF, MF, MF],
        16: [MF, MF, TL, PK, TR, MF, TR, UD, MF, MF, TL, MF],
        17: [TR, MF, MF, MF, TL, MF, MF],
        18: [MF, MF, MF, TR, MF, MF, MF, TR, MF],
        19: [TR, MF, MF, MF, TL, MF, MF],
        20: [MF, MF, TL, PK, TR, MF, TR, UD, MF, MF, MF, TR, MF],
        21: [TR, MF, MF, TR, MF],
        22: [MF, MF, MF, TR, MF, MF, TR, MF, MF, MF, MF],
        23: [TR, MF, MF, TR, MF],
        24: [MF, MF, TL, PK, TL, MF, MF, TL, UD, MF, MF, TR, MF],
        25: [MF, MF, MF, TR, MF, MF, TL, MF],
        26: [MF, MF, MF, TR, MF, MF, TL, MF],
        27: [TR, MF, MF, TL, MF, MF, MF, MF],
        28: [TL, MF, MF, TL, PK, TL, MF, MF, UD, MF, MF, TL, MF, MF, MF, MF],
        29: [TR, MF, MF, MF, TL, MF, MF],
        30: [MF, MF, MF, TR, MF, MF, MF, TR, MF],
        31: [TR, MF, MF, MF, TL, MF, MF],
        32: [TL, MF, MF, TL, PK, TL, MF, MF, UD, MF, MF, MF, TL, MF, MF],
        33: [TR, MF, MF, TR, MF],
        34: [MF, MF, MF, TR, MF, MF, TR, MF, MF, MF, MF],
        35: [TR, MF, MF, TR, MF],
        36: [TL, MF, MF, TL, PK, TL, MF, MF, UD, MF, MF, TR, MF]
    }


    env_folder = "envs/random_envs/"
    all_env_name = [f"DoorKey-8x8-{i}.env" for i in range(1, 37)]
    for i, env_name in enumerate(all_env_name, 1):
        env_path = env_folder + env_name
        env, info = load_env(env_path)
        draw_component_img_from_seq(seq[i], env, env_name)


if __name__ == "__main__":
    partA()
    partB()

