import numpy as np
import pandas as pd
import torch
from DQN_Agent_lib import *
from RAT_env import *
from RL import *
import matplotlib.pyplot as plt
from collections import Counter


def run_heuristic(rat_env, n_episodes,h_value,p_switch,m):
    
    """
    Runs a heuristic user association algorithm in the RAT environment over multiple episodes.
    Users probabilistically switch RATs if throughput improvement exceeds a threshold.

    Parameters:
    - rat_env: The RAT environment instance.
    - n_episodes: Number of episodes to simulate.
    - h_value: Threshold multiplier for throughput improvement to consider switching.
    - p_switch: Base probability of switching to a better RAT.
    - m: Counter influencing switching probability to avoid frequent switches.

    Returns:
    - rewards_buffer: List of collected rewards for each step.
    - actions_buffer: List of actions taken at each step.
    """

    n_steps = rat_env.n_steps
    n_users = rat_env.n_users
    actions_buffer = []
    rewards_buffer = []
    for i in tqdm(range(n_episodes), desc="Simulation progress"):
        rat_env.reset()
        for step in range(n_steps):
            current_state,_, _ = rat_env.get_state()
            state = expand_list(current_state,n_users)[:,:rat_env.n_stations]
            new_rats = []
            switch_register = []
            for user in range(n_users):
                switch = False
                current_rat = rat_env.user_assignments[user]
                current_thr = rat_env.get_ue_throughput(user)
                user_rates = state[user,:]
                for rat in range(rat_env.n_stations):
                    rat_id = 0 if rat < rat_env.n_ltesn else 1
                    node_id = rat - rat_id * rat_env.n_ltesn
                    if ([rat_id,node_id] != current_rat) and (user_rates[rat] > 0):
                        rat_env.user_assignments[user] = [rat_id, node_id]
                        possible_thr = rat_env.get_ue_throughput(user)
                        if possible_thr > current_thr*h_value:
                            if np.random.random() < p_switch**(m+1):
                                new_rat = node_id + rat_id * rat_env.n_ltesn
                                changed_rats = [rat_id for rat_id, flag in zip(new_rats, switch_register) if flag]
                                if new_rat in changed_rats:
                                    m += 1
                                else:
                                    m = 0
                                new_rats.append(new_rat)
                                switch = True                                
                                break
                rat_env.user_assignments[user] = current_rat # For the computation of next user, rates stay the same
                switch_register.append(switch)
                if not switch:
                    new_rats.append(current_rat[1]+ current_rat[0] * rat_env.n_ltesn) 
            # Execute step in the environment
            actions = new_rats
            rat_env.step(actions)
            _,_, _, lr = rat_env.step(actions)
            actions_buffer.append(torch.tensor(actions))
            rewards_buffer.append(torch.tensor(lr))

    return rewards_buffer, actions_buffer