from utils import *
import numpy as np
import minigrid
import time

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door
action_list = ["MF", "TL", "TR", "PK", "UD"]
WALL = 0
FLOOR = 1
KEY = 2
AGENT = 3
DOOR = 4
GOAL = 5

inf = float("inf")


class Agent:
    def __init__(self, env_path, random=False):

        self.random = random
        self.env_path = env_path
        self.env, self.info = load_env(env_path)

        self.pos_list = []
        self.wall_list = []
        self.key_list = []
        self.goal_list = []
        self.door_list = []
        self.get_cell_type()


    def get_state_space(self):
        """
        get all states and their init values
        return: list of dict, each dict represents a state(some may be impossible),
        and a list of float, each float represents the value of the corresponding state
        """
        all_state_list = []
        value_list = []
        for i in range(len(self.pos_list)): # all possible agent position
            for k in range(4): # all possible agent direction
                for l in range(2): # key status
                    for m in range(2): # door status
                        if self.pos_list[i] in self.goal_list:
                            state = {
                                "door_status": True if m == 1 else False,
                                "key_status": True if l == 1 else False,
                                "agent_pos": self.pos_list[i],
                                "agent_dir": [(0, 1), (1, 0), (0, -1), (-1, 0)][k],
                                "done": True,
                            }
                            value_list.append(0)
                        else:
                            state = {
                                "door_status": True if m == 1 else False,
                                "key_status": True if l == 1 else False,
                                "agent_pos": self.pos_list[i],
                                "agent_dir": [(0, 1), (1, 0), (0, -1), (-1, 0)][k],
                                "done": False,
                            }
                            value_list.append(inf)

                        all_state_list.append(state)

        return all_state_list, value_list

    def motion_model(self, state, action):
        """
        get the new state after taking the action in the current state.
        state: dict { "door_status": bool,
        "key_status": bool, "agent_pos": tuple, "agent_dir": tuple}
        action: int, 0: MF, 1: TL, 2: TR, 3: PK, 4: UD
        """
        new_state = state.copy()
        door_status = new_state["door_status"]
        key_status = new_state["key_status"]
        agent_pos = new_state["agent_pos"]
        agent_dir = new_state["agent_dir"]
        done = new_state["done"]

        # MF = 0, Move Forward
        if action == 0:
            new_agent_pos = (agent_pos[0]+agent_dir[0], agent_pos[1]+agent_dir[1])
            if new_agent_pos in self.wall_list:  # hit wall
                agent_pos = agent_pos
            elif new_agent_pos in self.door_list and not door_status:  # hit door
                agent_pos = agent_pos
            elif new_agent_pos in self.goal_list:  # reach goal
                agent_pos = new_agent_pos
                done = True
            else:  # move forward
                agent_pos = new_agent_pos

        # TL = 1, Turn Left
        if action == 1:
            agent_dir = (agent_dir[1], -agent_dir[0])

        # TR = 2, Turn Right
        if action == 2:
            agent_dir = (-agent_dir[1], agent_dir[0])

        # PK = 3, Pickup Key
        if action == 3:
            agent_front = (agent_pos[0]+agent_dir[0], agent_pos[1]+agent_dir[1])
            if agent_front in self.key_list:
                key_status = True

        # UD = 4, Unlock Door
        if action == 4:
            agent_front = (agent_pos[0]+agent_dir[0], agent_pos[1]+agent_dir[1])
            if agent_front in self.door_list and key_status:
                door_status = True
            elif agent_front in self.door_list and not key_status:
                door_status = False

        new_state = {
            "door_status": door_status,
            "key_status": key_status,
            "agent_pos": agent_pos,
            "agent_dir": agent_dir,
            "done": done,
        }
        return new_state

    def get_init_state(self):
        """
        get the initial state of the environment
        """

        agent_pos = self.env.agent_pos

        agent_dir = (self.env.dir_vec[0], self.env.dir_vec[1])

        # Get the door status
        door = self.env.grid.get(self.info["door_pos"][0], self.info["door_pos"][1])
        is_open = door.is_open

        # Determine whether agent is carrying a key
        is_carrying = self.env.carrying is not None

        done = False

        init_state = {
            "door_status": is_open,
            "key_status": is_carrying,
            "agent_pos": agent_pos,
            "agent_dir": agent_dir,
            "done": done,
        }

        return init_state

    def get_cell_type(self):
        """
        get the type of each cell in the environment,
        save the coordinates of each type of cell in the corresponding list
        """
        for i in range(self.env.height):
            for j in range(self.env.width):
                cell = self.env.grid.get(j, i)
                if cell == None:
                    self.pos_list.append((j, i))
                elif isinstance(cell, minigrid.core.world_object.Wall):
                    self.wall_list.append((j, i))
                elif isinstance(cell, minigrid.core.world_object.Key):
                    self.key_list.append((j, i))
                    self.pos_list.append((j, i))
                elif isinstance(cell, minigrid.core.world_object.Door):
                    self.door_list.append((j, i))
                    self.pos_list.append((j, i))
                elif isinstance(cell, minigrid.core.world_object.Goal):
                    self.goal_list.append((j, i))
                    self.pos_list.append((j, i))

def dp_algo(env_path):

    agent = Agent(env_path)

    all_state_list, value_list = agent.get_state_space()
    T = len(all_state_list) - 1

    policy_list = [inf] * len(all_state_list)

    for t in range(T): # for each time step
        value_list_copy = value_list.copy()
        for i in range(len(all_state_list)): # for each state
            state = all_state_list[i]
            if state["done"]:
                value_list[i] = 0
            else:
                q_value_list = []
                for k in range(5): # for each action
                    new_state = agent.motion_model(state, k)
                    j = all_state_list.index(new_state)
                    q_value_list.append(1 + value_list_copy[j])

                # update value
                value_list[i] = min(q_value_list)
                # update policy
                policy_list[i] = np.argmin(q_value_list)

        # early stop
        if value_list == value_list_copy:
            break

    state = agent.get_init_state()
    optimal_policy_seq = []
    while not state["done"]:
        state_index = all_state_list.index(state)
        action = policy_list[state_index]
        state = agent.motion_model(state, action)
        optimal_policy_seq.append(action_list[int(action)])
    print("the env_path is: ", env_path)
    print("Optimal Policy Sequence: ", optimal_policy_seq)


if __name__ == "__main__":
    import os
    directory_path = "envs/known_envs"
    env_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.env'):
            env_files.append(os.path.join(directory_path, filename))
    for env_path in env_files:
        dp_algo(env_path)