import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc
from RAT_env import *
from tqdm import tqdm
import os


def compute_regret(regret_vector,rat_env,user_idx,user_action,user_reward):
    """
    Computes and updates the regret vector for a given user based on the difference
    between the rewards of alternative actions and the user's current reward.

    Parameters:
    - regret_vector (torch.Tensor): Current regret vector for the user.
    - rat_env (object): The RAT environment instance.
    - user_idx (int): Index of the user.
    - user_action (int): Current action taken by the user.
    - user_reward (float): Reward received by the user for the current action.

    Returns:
    - torch.Tensor: Updated regret vector for the user.
    """
    regrets = [0.0] * rat_env.n_stations # Initialize regrets for all possible actions
    for rat in range(rat_env.n_stations):
        rat_id = 0 if rat < rat_env.n_ltesn else 1 # Determine RAT type
        node_id = rat - rat_id * rat_env.n_ltesn # Determine node index
        rat_env.user_assignments[user_idx] = [rat_id, node_id] # Temporarily assign alternative action
        rewards = rat_env.r() # Evaluate rewards under new assignment
        new_reward = rewards[user_idx]
        regrets[rat] = max(0, new_reward - user_reward) # Only positive regret is accumulated

    # Restore original assignment
    rat_id = 0 if user_action < rat_env.n_ltesn else 1
    node_id = user_action - rat_id * rat_env.n_ltesn
    rat_env.user_assignments[user_idx] = [rat_id, node_id]
    
    # Update regret vector with new values
    regret_vector = regret_vector + torch.tensor(regrets, dtype=torch.float32).to(device)
    return regret_vector

def run_Hart_RL(rat_env, max_steps, sim_steps):
    """
    Runs the Hart learning algorithm for multi-agent reinforcement learning in the RAT environment.
    Probability vectors are updated iteratively based on regret values for each user.

    Parameters:
    - rat_env (object): The RAT environment instance.
    - max_steps (int): Maximum number of steps before environment reset.
    - sim_steps (int): Total number of simulation steps to run.

    Returns:
    - List[torch.Tensor]: List of learned probability vectors for each user.
    """
    n_agents = rat_env.n_users
    # Initialize equal probability vectors and empty regret vectors
    p_vectors = [torch.full((rat_env.n_stations,), 1 / rat_env.n_stations) for _ in range(n_agents)]
    regret_vectors = [torch.zeros(rat_env.n_stations) for _ in range(n_agents)]
    actions = torch.zeros(n_agents, dtype=torch.int64)

    for _ in tqdm(range(0, sim_steps), desc="Simulation Progress"):
        # Sample actions from probability vectors
        for i,p_vector in enumerate(p_vectors):
            action = np.random.choice(len(p_vector), p=p_vector.cpu().numpy() / p_vector.sum().item())
            actions[i] = action
        _,_,_,rewards = rat_env.step(actions)

        # Update regret and adjust probability vectors
        for i, (p_vector, regret_vector) in enumerate(zip(p_vectors, regret_vectors)):
            regret_vectors[i] = compute_regret(regret_vector,rat_env, i, actions[i], min(0,rewards[i]))
            total_regret = torch.sum(regret_vector)
            if total_regret > 0:
                p_vectors[i] = torch.clamp(regret_vectors[i] / total_regret, min=0.0)
            else:
                p_vectors[i] = torch.full_like(p_vectors[i], 1 / len(p_vectors[i]))
        
        # Reset environment if needed
        if rat_env.iteration == (max_steps-1):
            rat_env.reset()
        else:
            rat_env.iteration = rat_env.iteration + 1

    return p_vectors

def evaluate_hart_rl(p_vectors, rat_env, n_episodes):
    """
    Evaluates the Hart RL policy by simulating a number of episodes.
    Actions are sampled from the learned probability distributions, and rewards collected.

    Parameters:
    - p_vectors (List[torch.Tensor]): Learned probability vectors for each user.
    - rat_env (object): The RAT environment instance.
    - n_episodes (int): Number of episodes to simulate for evaluation.

    Returns:
    - List[torch.Tensor], List[torch.Tensor] : Buffers of collected rewards and actions.
    """
    reward_buffer = []
    action_buffer = []
    for i in range(n_episodes):
        rat_env.reset()
        for t in range(0, rat_env.n_steps):
            actions = torch.zeros(rat_env.n_users, dtype=torch.int64)
            # Sample actions from probability vectors
            for i,p_vector in enumerate(p_vectors):
                action = np.random.choice(len(p_vector), p=p_vector.cpu().numpy() / p_vector.sum().item())
                actions[i] = action
            _,_,_,rewards = rat_env.step(actions)
            
            action_buffer.append(actions)
            reward_buffer.append(rewards)
    
    return reward_buffer, action_buffer