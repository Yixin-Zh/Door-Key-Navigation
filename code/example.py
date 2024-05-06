from utils import *
import minigrid
MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def example_use_of_gym_env():
    """
    The Coordinate System:
        (0,0): Top Left Corner
        (x,y): x-th column and y-th row
    """

    print("<========== Example Usages ===========> ")
    env_path = "envs/random_envs/DoorKey-8x8-2.env"
    # env, info = load_env(env_path) # load an environment

    env, info = load_env(env_path)
    print("<Environment Info>\n")
    print(info)  # Map size
    # agent initial position & direction,
    # key position, door position, goal position
    print("<================>\n")

    # Visualize the environment
    plot_env(env)

    # Get the agent position
    agent_pos = env.agent_pos
    print(agent_pos)
    # Get the agent direction
    agent_dir = env.dir_vec  # or env.agent_dir
    print(agent_dir)
    # Get the cell in front of the agent
    front_cell = env.front_pos  # == agent_pos + agent_dir

    print(front_cell)
    # Access the cell at coord: (2,3)
    cell = env.grid.get(2, 3)  # NoneType, Wall, Key, Goal

    # Get the door status
    door = env.grid.get(info["door_pos"][0], info["door_pos"][1])
    is_open = door.is_open
    is_locked = door.is_locked
    print("##################")
    print("Is Open: {}".format(is_open))
    # Determine whether agent is carrying a key
    is_carrying = env.carrying is not None
    print("Is Carrying Key: {}".format(is_carrying))
    # Take actions
    cost, done = step(env, MF)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Moving Forward Costs: {}".format(cost))
    cost, done = step(env, TL)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Turning Left Costs: {}".format(cost))
    cost, done = step(env, TR)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Turning Right Costs: {}".format(cost))
    cost, done = step(env, PK)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Picking Up Key Costs: {}".format(cost))
    cost, done = step(env, UD)  # MF=0, TL=1, TR=2, PK=3, UD=4
    print("Unlocking Door Costs: {}".format(cost))

    # Determine whether we stepped into the goal
    if done:
        print("Reached Goal")

    # The number of steps so far
    print("Step Count: {}".format(env.step_count))
    print("<================>\n")

    # iterate and print the cell
    pos_list = []
    wall_list = []
    key_list = []
    door_list = []
    goal_list = []
    for i in range(env.height):
        for j in range(env.width):
            cell = env.grid.get(j, i)
            if cell == None:
                pos_list.append((j, i))
            elif isinstance(cell, minigrid.core.world_object.Wall):
                wall_list.append((j, i))
            elif isinstance(cell, minigrid.core.world_object.Key):
                key_list.append((j, i))
                pos_list.append((j, i))
            elif isinstance(cell, minigrid.core.world_object.Door):
                door_list.append((j, i))
                pos_list.append((j, i))
            elif isinstance(cell, minigrid.core.world_object.Goal):
                goal_list.append((j, i))
                pos_list.append((j, i))
    print("<Cell Info>\n")
    print("pos_list: ", pos_list)
    print("wall_list: ", wall_list)
    print("key_list: ", key_list)
    print("door_list: ", door_list)
    print("goal_list: ", goal_list)
    print("<================>\n")

if __name__ == "__main__":
    example_use_of_gym_env()

