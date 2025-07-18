import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc
import math
from RAT_env import *
from tqdm import tqdm
import csv
from DQN_Agent_lib import *
import os

device=torch.device("cpu")

# -------------------------------------------------------------------
# This file executes the DQN Reinforcement Learning Algorithm
# -------------------------------------------------------------------

# Define truncation function
def expand_list(state, n_users):
    """
    Prepares input features for the neural network by converting
    a multi-agent environment state into a structured tensor.
    Each agent's state is separated and normalized to ensure compatibility
    with the neural network's input expectations
    :param state: State object (from environment) representing the system state.
    :param n_users: Total number of agents/users in the environment.
    :return: Tensor of shape (n_users, features_per_user)
    """
    states = []
    for i in range(0, n_users):
        s = state.to_sep_numpy(i)
        states.append(s)
    
    return torch.stack(states).float()

def expand_list(state, n_users):
    """
    Prepares input features for the neural network by converting
    a multi-agent environment state into a structured tensor.

    Each agent's state is separated and normalized to ensure compatibility
    with the neural network's input expectations.

    :param state: State object (from environment) representing the system state.
    :param n_users: Total number of agents/users in the environment.
    :return: Tensor of shape (n_users, features_per_user)
    """
    states = []
    for i in range(0, n_users):
        s = state.to_sep_numpy(i)
        states.append(s)
    
    return torch.stack(states).float()

def run_DQN_Agent(rat_env, max_steps, sim_steps, exploration_fraction, buffer_size, AN_file_name):
    """
    Executes the Deep Q-Network (DQN) algorithm on the given multi-RAT environment.
    Handles training, exploration/exploitation decisions, learning, and model saving.

    :param rat_env: An instance of the Multi_RAT_Network environment.
    :param max_steps: Number of steps per episode before reset.
    :param sim_steps: Total number of environment interactions (training steps).
    :param exploration_fraction: Fraction of training during which epsilon decays linearly.
    :param buffer_size: Size of the experience replay buffer.
    :param AN_file_name: Path prefix to save the trained Action Network weights.
    :return: Tuple containing:
             - dqn_agent: The trained DQN agent object.
             - sum_loss: List of loss values over time.
             - last_rats: Last selected stations by all users for visualization.
             - episode_rewards: List of cumulative rewards per episode.
             - best_actions: Most frequently selected actions over training.
             - eps_list: Epsilon values used over time.
             - episode_length: Length of each episode in terms of steps.
    """
    n_agents = rat_env.n_users

    # Visualization variables
    reward_values = []
    episode_rewards = []
    episode_reward = np.zeros(n_agents)
    last_rats = [0] * rat_env.n_stations
    best_actions = [0] * rat_env.n_stations

    # Initialize agent and replay buffer
    dqn_agent = DQN_NN(n_users=n_agents, n_stations=rat_env.n_stations)
    replay_buffer = ExperienceReplay(buffer_size)

    sum_loss = []
    ep = 1
    min_ep = 0.05  # Minimum epsilon
    eps_list = []
    episode_length = []

    # ---------- Main simulation and training loop -----------------
    for global_step in tqdm(range(0, sim_steps), desc="Simulation Progress"):
        # Linear epsilon decay
        eps = max(ep - (ep - min_ep) * global_step / sim_steps / exploration_fraction, min_ep)
        eps_list.append(eps)

        current_state, lr, _ = rat_env.get_state()
        state = expand_list(current_state, n_agents)

        # Choose action using epsilon-greedy strategy
        rand_action_flag = np.random.random() < eps
        if rand_action_flag:
            actions = torch.randint(0, rat_env.n_stations - 1, (n_agents,))
        else:
            q_values = dqn_agent.predict_action(state).detach()
            actions = torch.argmax(q_values, dim=1)
            best_actions = rat_env.get_rat_chosen(actions, best_actions)

        last_rats = rat_env.get_rat_chosen(actions, last_rats)

        # Perform a step in the environment
        current_state, _, new_state, lr = rat_env.step(actions.detach())
        cur_s = expand_list(current_state, n_agents)
        next_s = expand_list(new_state, n_agents)

        negative_reward = torch.any(torch.eq(lr, -0.1))
        episode_reward += lr.numpy()  # Track cumulative reward per episode

        # Check for terminal state
        if rat_env.iteration == (max_steps - 1) or negative_reward:
            isLastState = torch.ones(n_agents, dtype=torch.float32).to(device)

            episode_rewards.append(episode_reward)
            episode_reward = np.zeros(n_agents)
            episode_length.append(rat_env.iteration + 1)

            rat_env.reset()
        else:
            isLastState = torch.zeros(n_agents, dtype=torch.float32).to(device)
            rat_env.iteration += 1

        # Store transition in replay buffer
        rewards = lr.detach()
        next_s = next_s.detach()
        replay_buffer.add(cur_s, next_s, isLastState, rewards, actions)

        # Start training when the buffer is full
        learning_starts = buffer_size
        if global_step > learning_starts:
            if global_step % 1 == 0:
                gamma = 0.9
                sample = replay_buffer.sample(128)
                current_state_list, next_state_list, isLastState_list, rewards_list, act_list = sample

                with torch.no_grad():
                    target_max, _ = dqn_agent.predict_value(next_state_list).max(dim=2)
                    td_target = rewards_list.flatten() + gamma * target_max.flatten() * (1 - isLastState_list.flatten())

                act_list = act_list.long()
                old_value = dqn_agent.predict_action(current_state_list).gather(2, act_list.unsqueeze(-1)).squeeze(-1).flatten()

                loss = F.mse_loss(td_target, old_value)

                # Backpropagation
                dqn_agent.optimizer_DQN.zero_grad()
                loss.backward()
                dqn_agent.optimizer_DQN.step()

                sum_loss.append(loss.item())

            # Soft update target network (Polyak averaging)
            tau = 0.5
            for target_param, q_param in zip(dqn_agent.value_net.parameters(), dqn_agent.action_net.parameters()):
                target_param.data.copy_(
                    tau * q_param.data + (1.0 - tau) * target_param.data
                )

            if global_step % (sim_steps / 10) == 0:
                print(f"Iteration {global_step} A_Loss: {loss}")

    # Save trained action network to disk
    print("Saving final weights to disk")
    torch.save(dqn_agent.action_net.state_dict(), AN_file_name + ".pt")
    print("Weights saved to disk")

    # Save reward data for visualization
    print("csv imported")    
    with open('rewards.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["step"] + [f"reward_{i}" for i in range(1, rat_env.n_users + 1)])
        for i, reward in enumerate(reward_values):
            writer.writerow([i] + reward)

    # Return training artifacts
    return dqn_agent, sum_loss, last_rats, episode_rewards, best_actions, eps_list, episode_length


"""     # Save here till we run and make sure the commented code run by gpt works
n_agents = rat_env.n_users
#------------for debugging and for visualization-----------
    reward_values = [] # store training rewards
    episode_rewards = [] # store training rewards 
    episode_reward = np.zeros(n_agents)
    last_rats = [0]*rat_env.n_stations # store last rats chosen
    best_actions = [0]*rat_env.n_stations 
    #------------------------
    dqn_agent = DQN_NN(n_users=n_agents, n_stations = rat_env.n_stations)
    replay_buffer = ExperienceReplay(buffer_size)
    sum_loss = [] #list to store episode loss
    ep = 1
    min_ep = 0.05
    eps_list = []
    episode_length = []
    # ---------- Main simulation Block -----------------
    for global_step in tqdm(range(0, sim_steps), desc="Simulation Progress"):
        eps = max(ep-(ep-min_ep)*global_step/sim_steps/exploration_fraction, min_ep) #exploration rate
        eps_list.append(eps)
        current_state, lr, _ = rat_env.get_state()
        state = expand_list(current_state,n_agents)

        rand_action_flag = np.random.random() < eps # Choose exploration or exploitation
        if rand_action_flag:
            actions = torch.randint(0, rat_env.n_stations - 1 ,(n_agents,))
        else: # choose best action
            q_values = dqn_agent.predict_action(state).detach()
            actions = torch.argmax(q_values, dim=1)
            best_actions = rat_env.get_rat_chosen(actions,best_actions) # For visualization
        
        last_rats = rat_env.get_rat_chosen(actions,last_rats) # For visualization
        
        # Take an step with the chosen action
        current_state,_, new_state, lr = rat_env.step(actions.detach())
        cur_s = expand_list(current_state,n_agents)
        next_s = expand_list(new_state, n_agents)
        negative_reward = torch.any(torch.eq(lr, -0.1))

        # -------- For visualization --------
        episode_reward += lr.numpy()
        #---------------------------------- 
        if rat_env.iteration == (max_steps-1) or negative_reward:
            isLastState = torch.ones(n_agents,dtype=torch.float32).to(device)
            # -------- For visualization --------
            episode_rewards.append(episode_reward)
            episode_reward = np.zeros(n_agents)
            episode_length.append(rat_env.iteration+1)
            #---------------------------------- 
            rat_env.reset()
        else:
            isLastState = torch.zeros(n_agents,dtype=torch.float32).to(device)
            rat_env.iteration = rat_env.iteration + 1
            
    
        # Add step results to the buffer
        rewards = lr.detach()
        next_s = next_s.detach()
        replay_buffer.add(cur_s, next_s, isLastState, rewards, actions)

        learning_starts = buffer_size
        if global_step > learning_starts: # Buffer is full, we can start learning
            if global_step % 1 == 0:
                gamma = 0.9
                buffer_sample = replay_buffer.sample(128)
                current_state_list, next_state_list, isLastState_list, rewards_list, act_list = buffer_sample                
                with torch.no_grad():
                    target_max,_ = dqn_agent.predict_value(next_state_list).max(dim=2)
                    td_target = rewards_list.flatten() + gamma * target_max.flatten() * (1 - isLastState_list.flatten())
                act_list = act_list.long()
                old_value = dqn_agent.predict_action(current_state_list).gather(2, act_list.unsqueeze(-1)).squeeze(-1).flatten()
                loss = F.mse_loss(td_target,old_value)
                    
                # Optimization step
                dqn_agent.optimizer_DQN.zero_grad()
                loss.backward()
                dqn_agent.optimizer_DQN.step()

                sum_loss.append(loss.item()) # store loss for visualization
            
            # Update target network
            tau = 0.5
            for target_param, q_param in zip(dqn_agent.value_net.parameters(), dqn_agent.action_net.parameters()):
                target_param.data.copy_(
                    tau * q_param.data + (1.0 - tau) * target_param.data
                )   
            if  global_step %(sim_steps/10) == 0:
                print(f"Iteration {global_step} A_Loss: {loss}")         

    print("Saving final weights to disk")
    torch.save(dqn_agent.action_net.state_dict(), AN_file_name + ".pt") # Save action network
    print("Weights saved to disk")

    # ------------For visualization-------------
    print("csv imported")    
    with open('rewards.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["step"] + [f"reward_{i}" for i in range(1, rat_env.n_users +1 )])
        for i, reward in enumerate(reward_values):
            writer.writerow([i] + reward)
    # ---------------------------------------------



    return dqn_agent, sum_loss, last_rats, episode_rewards, best_actions,eps_list,episode_length

    """