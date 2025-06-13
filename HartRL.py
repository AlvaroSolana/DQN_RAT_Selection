import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc
from RAT_env import *
from tqdm import tqdm

import os


def compute_regret(regret_vector,rat_env,user_idx,user_action,user_reward):
    # Obtain new regret vectors for the user
    regrets = [0.0] * rat_env.n_stations
    for rat in range(rat_env.n_stations):
        rat_id = 0 if rat < rat_env.n_ltesn else 1
        node_id = rat - rat_id * rat_env.n_ltesn
        rat_env.user_assignments[user_idx] = [rat_id, node_id]
        rewards = rat_env.r()
        new_reward = rewards[user_idx]
        regrets[rat] = max(0, new_reward - user_reward)

    # return the user to the original assignment
    rat_id = 0 if user_action < rat_env.n_ltesn else 1
    node_id = user_action - rat_id * rat_env.n_ltesn
    rat_env.user_assignments[user_idx] = [rat_id, node_id]
    regret_vector = regret_vector + torch.tensor(regrets, dtype=torch.float32).to(device)
    return regret_vector

def run_Hart_RL(rat_env, max_steps, sim_steps):
    """
    Runs the nash RL algothrim and outputs two files that hold the network parameters
    for the estimated action network and value network
    :param num_sim:           Number of Simulations
    :param batch_update_size: Number of experiences sampled at each time step
    :param buffersize:        Maximum size of replay buffer
    :return: Truncated Array
    """
    n_agents = rat_env.n_users
    p_vectors = [torch.full((rat_env.n_stations,), 1 / rat_env.n_stations) for _ in range(n_agents)]
    regret_vectors = [torch.zeros(rat_env.n_stations) for _ in range(n_agents)]
    actions = torch.zeros(n_agents, dtype=torch.int64)

    for _ in tqdm(range(0, sim_steps), desc="Simulation Progress"):
        # Select and take action
        for i,p_vector in enumerate(p_vectors):
            #print(f"p_vector for agent {i}: {p_vector}")
            action = np.random.choice(len(p_vector), p=p_vector.cpu().numpy() / p_vector.sum().item())
            actions[i] = action
        _,_,_,rewards = rat_env.step(actions)

        for i, (p_vector, regret_vector) in enumerate(zip(p_vectors, regret_vectors)):
            regret_vectors[i] = compute_regret(regret_vector,rat_env, i, actions[i], min(0,rewards[i]))
            total_regret = torch.sum(regret_vector)
            if total_regret > 0:
                p_vectors[i] = torch.clamp(regret_vectors[i] / total_regret, min=0.0)
            else:
                p_vectors[i] = torch.full_like(p_vectors[i], 1 / len(p_vectors[i]))
        
        if rat_env.iteration == (max_steps-1):
            rat_env.reset()
        else:
            rat_env.iteration = rat_env.iteration + 1

    #Simulation finished, save the p_vectors   
    return p_vectors

def evaluate_hart_rl(p_vectors, rat_env, n_episodes):
    reward_buffer = []
    action_buffer = []
    for i in range(n_episodes):
        rat_env.reset()
        for t in range(0, rat_env.n_steps):
            actions = torch.zeros(rat_env.n_users, dtype=torch.int64)
            for i,p_vector in enumerate(p_vectors):
                #print(f"p_vector for agent {i}: {p_vector}")
                action = np.random.choice(len(p_vector), p=p_vector.cpu().numpy() / p_vector.sum().item())
                actions[i] = action
            _,_,_,rewards = rat_env.step(actions)
            
            action_buffer.append(actions)
            reward_buffer.append(rewards)
    
    return reward_buffer, action_buffer