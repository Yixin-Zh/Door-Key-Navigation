from utils import *
import numpy as np
import minigrid
import time
import copy
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
    def __init__(self):

        self.wall_list = [(4, 0), (4, 1), (4, 3), (4, 4), (4, 6), (4, 7)]
        self.door_list = [(4, 2), (4, 5)]

        self.door_list = [(4, 2), (4, 5)]
        self.env_length = 8
        self.pos_list = self.get_pos_list()

        self.goal_list = [(5, 1), (6, 3), (5, 6)]
        self.key_list = [(1, 1), (2, 3), (1, 6)]
        self.door_status = [[i, j] for i in [True, False] for j in [True, False]]

    def get_pos_list(self):
        """
        get all possible agent positions
        """
        pos_list = []
        for i in range(self.env_length):
            for j in range(self.env_length):
                if (j, i) not in self.wall_list:
                    pos_list.append((j, i))
        return pos_list

    def get_state_space(self):
        """
        get all states and their init values
        return: list of dict, each dict represents a state(some may be impossible),
        and a list of float, each float represents the value of the corresponding state
        """
        all_state_list = []
        value_list = []
        for a in range(len(self.key_list)):
            key_pos = self.key_list[a]
            for b in range(len(self.door_status)):
                door_status = self.door_status[b]
                for c in range(len(self.goal_list)):
                    goal_pos = self.goal_list[c]
                    for i in range(len(self.pos_list)):  # all possible agent position
                        for k in range(4):  # all possible agent direction
                            for l in range(2):  # key status
                                if self.pos_list[i] == goal_pos:
                                    state = {
                                        "key_status": bool(l),
                                        "agent_pos": self.pos_list[i],
                                        "agent_dir": [(0, 1), (1, 0), (0, -1), (-1, 0)][k],
                                        "key_pos": key_pos,
                                        "door_status": door_status,
                                        "goal_pos": goal_pos,
                                    }
                                    value_list.append(0)
                                    all_state_list.append(state)
                                elif self.pos_list[i] in self.wall_list:
                                    pass
                                elif self.pos_list[i] == (4, 2) and not door_status[0]:
                                    pass
                                elif self.pos_list[i] == (4, 5) and not door_status[1]:
                                    pass
                                else:
                                    state = {
                                        "key_status": bool(l),
                                        "agent_pos":  self.pos_list[i],
                                        "agent_dir": [(0, 1), (1, 0), (0, -1), (-1, 0)][k],
                                        "key_pos": key_pos,
                                        "door_status": door_status,
                                        "goal_pos": goal_pos,
                                    }
                                    value_list.append(inf)
                                    all_state_list.append(state)

        return all_state_list, value_list

    def motion_model(self, state, action):
        """
        get the new state after taking the action in the current state.
        state: {"key_status": bool(l),
        "agent_pos": self.pos_list[i],
        "agent_dir": [(0, 1), (1, 0), (0, -1), (-1, 0)][k],
        "key_pos": key_pos,
        "door_status": door_status,
        action: int, 0: MF, 1: TL, 2: TR, 3: PK, 4: UD
        """
        new_state = copy.deepcopy(state)
        key_status = new_state["key_status"]
        agent_pos = new_state["agent_pos"]
        agent_dir = new_state["agent_dir"]
        key_pos = new_state["key_pos"]
        door_status = new_state["door_status"]
        goal_pos = new_state["goal_pos"]

        # MF = 0, Move Forward
        if action == 0:
            new_agent_pos = (agent_pos[0]+agent_dir[0], agent_pos[1]+agent_dir[1])
            if new_agent_pos in self.wall_list:  # hit wall
                agent_pos = agent_pos
            elif new_agent_pos in self.door_list and not door_status[self.door_list.index(new_agent_pos)]:  # hit door
                agent_pos = agent_pos
            elif new_agent_pos[0] < 0 or new_agent_pos[0] >= self.env_length\
                    or new_agent_pos[1] < 0 or new_agent_pos[1] >= self.env_length: # out of bound
                agent_pos = agent_pos
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
            if agent_front == key_pos:
                key_status = True

        # UD = 4, Unlock Door
        if action == 4:
            agent_front = (agent_pos[0]+agent_dir[0], agent_pos[1]+agent_dir[1])
            if agent_front in self.door_list and key_status:
                door_status[self.door_list.index(agent_front)] = True

        new_state = {
            "key_status": key_status,
            "agent_pos": agent_pos,
            "agent_dir": agent_dir,
            "key_pos": key_pos,
            "door_status": door_status,
            "goal_pos": goal_pos,
        }
        return new_state

    def get_init_state(self, env_path):
        """
        get the initial state of the environment
        """
        env, info = load_env(env_path)
        agent_pos = env.agent_pos

        agent_dir = (env.dir_vec[0], env.dir_vec[1])

        # Get the door status
        door1 = env.grid.get(4, 2)
        door1_status = door1.is_open
        door2 = env.grid.get(4, 5)
        door2_status = door2.is_open
        door_status = [door1_status, door2_status]

        key_status = False
        key_pos = info["key_pos"]
        key_pos = (key_pos[0], key_pos[1])

        goal_pos = info["goal_pos"]
        goal_pos = (goal_pos[0], goal_pos[1])
        init_state = {
            "key_status": key_status,
            "agent_pos": agent_pos,
            "agent_dir": agent_dir,
            "key_pos": key_pos,
            "door_status": door_status,
            "goal_pos": goal_pos,
        }

        return init_state, goal_pos


def get_optimal_policy_seq(env_path, all_state_list_copy, policy_list):
    agent_temp = Agent()
    init_state, goal_pos = agent_temp.get_init_state(env_path)
    optimal_policy_seq = []
    state = init_state
    while not state["agent_pos"] == state["goal_pos"]:
        state_index = all_state_list_copy.index(state)
        action = policy_list[state_index]
        state = agent_temp.motion_model(state, action)
        optimal_policy_seq.append(action_list[int(action)])
    return optimal_policy_seq

def dp_algo():
    """
    compute a single optimal policy sequence for each random environment
    """
    agent = Agent()
    all_state_list, value_list = agent.get_state_space()
    all_state_list_copy = copy.deepcopy(all_state_list)
    T = len(all_state_list) - 1
    print("T", T)
    policy_list = [inf] * len(all_state_list_copy)

    # compute the optimal policy sequence

    for t in range(T): # for each time step
        value_list_copy = value_list.copy()
        t1 = time.time()
        for i in range(len(all_state_list)): # for each state
            state = all_state_list[i]
            if state["agent_pos"] == state["goal_pos"]:
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
        print("time", time.time()-t1)
        # early stop
        if value_list == value_list_copy:
            break

    # get the optimal policy sequence
    environments = [f"envs/random_envs/DoorKey-8x8-{i}.env" for i in range(1, 37)]

    # Loop through each environment
    for i, env in enumerate(environments, 1):
        seq = get_optimal_policy_seq(env, all_state_list_copy, policy_list)
        print(f"the optimal seq for random{i} is:", seq)

if __name__ == "__main__":
    dp_algo()