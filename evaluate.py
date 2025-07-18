import torch
import numpy as np
from RL import expand_list
import matplotlib.pyplot as plt
from collections import Counter
from RAT_env import *


def predict_action(nn,states):
    """
    Predicts the actions for given input states using the provided neural network.
    Handles batch inputs by flattening and reshaping appropriately.
    Returns the predicted actions as network outputs.
    """

    if len(states.shape) > 2:
        B, A, D = states.shape 
        flat_states = states.view(B * A, D) 
        action_list = nn.forward(input=flat_states)
        action_list = action_list.view(B, A, -1)
    else:
        action_list = nn.forward(input=states) 
        
    return action_list

def evaluate_action_network(rat_env, action_network, n_episodes):
    """
    Evaluates the performance of the action network over multiple episodes in the given RAT environment.
    Runs episodes, collects rewards and actions taken by users.
    Returns lists of rewards and actions recorded during evaluation.
    """
    action_network.eval()

    reward_buffer = []
    action_buffer = []
    for _ in range(n_episodes):
        rat_env.reset()
        for _ in range(0, rat_env.n_steps):
            current_state, _, _ = rat_env.get_state()
            state = expand_list(current_state, rat_env.n_users)
            actions = torch.argmax(predict_action(action_network, state), dim=1)
            _, _, _, reward = rat_env.step(actions.detach())
            
            action_buffer.append(actions)
            reward_buffer.append(reward)
    
    return reward_buffer, action_buffer


def plot_reward(reward_buffer):
    """
    Plots the evolution of rewards for each user across multiple runs/episodes.
    Displays separate subplots for each user showing their reward progression over time.
    """
    n_runs = len(reward_buffer)
    n_agents = len(reward_buffer[0])
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i in range(n_agents):
        ax = axes[i // 5, i % 5]
        agent_rewards = np.array([reward_buffer[run][i] for run in range(n_runs)])
    
        ax.plot(range(n_runs), agent_rewards, label=f"User {i+1}")
        ax.set_title(f"User {i+1}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Reward Evolution for Each User Across Runs", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_actions(actions_buffer):
    """
    Plots the station choices made by each user across multiple runs/episodes.
    Converts actions to integer station indices and visualizes trends per user.
    """    
    n_runs = len(actions_buffer)
    n_agents = len(actions_buffer[0])
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(n_agents):
        ax = axes[i // 5, i % 5]
        agent_actions = np.array([int(torch.round(actions_buffer[run][i])) for run in range(n_runs)])
        ax.plot(range(n_runs), agent_actions, label=f"Agent {i+1}")
        ax.set_title(f"User {i+1}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Station chosen")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Stations Chosen by Each User Across Runs", fontsize=16)
    plt.tight_layout()
    plt.show()

def print_stats(rewards, actions,n_users):
    """
    Prints statistical summaries of actions and rewards after evaluation.
    Counts how many AP and LTE stations were chosen and how often.
    Calculates average rewards per RAT type and disconnection occurrences.
    Outputs results formatted for clear interpretation.
    """
    APs_chosen = [value.item() for action in actions for value in action if value.item() > 3]
    counted_aps = Counter(APs_chosen)
    LTEs_chosen = [value.item() for action in actions for value in action if value.item() <= 3]
    counted_ltes = Counter(LTEs_chosen)
    average_LTE_reward = np.mean([reward for reward, action in zip(rewards, actions) if any(value.item() <= 3 for value in action)])
    average_AP_reward = np.mean([reward for reward, action in zip(rewards, actions) if any(value.item() > 3 for value in action)])
    all_rewards = torch.cat(rewards).numpy()
    total_sn_chosen =  len(APs_chosen) + len(LTEs_chosen)
    disconnections = np.sum(all_rewards < 0)

    print(f"TEST RESULTS with {n_users} users in 150 episodes ({total_sn_chosen} stations chosen)")
    print("-----------------------------------------")
    print(f"{len(counted_aps)} APs chosen {len(APs_chosen)} times ({100 * len(APs_chosen) / (len(APs_chosen) + len(LTEs_chosen)):.2f}%) | {len(counted_ltes)} LTEs chosen {len(LTEs_chosen)} times ({100 * len(LTEs_chosen) / (len(APs_chosen) + len(LTEs_chosen)):.2f}%)")
    print(f"User disconnected {disconnections} times ({(disconnections / total_sn_chosen*100):.5f} %)")
    print(f"Average reward in LTEs: {average_LTE_reward:.3f} --> {average_LTE_reward * 480 :.3f} Mb/s")
    print(f"Average reward in APs: {average_AP_reward:.3f} --> {average_AP_reward * 480:.3f} Mb/s")
    print("-----------------------------------------")
    print(f"LTEs chosen :", counted_ltes)
    print(f"APs chosen :", counted_aps)


def plot_episode_reward(reward_buffer, n_step):
    """
    Plots the average reward per episode for each user.
    Calculates average rewards over fixed step intervals (episodes).
    Shows user-wise performance trends over episodes.
    """
    n_timesteps = len(reward_buffer)
    n_users = len(reward_buffer[0])
    n_episodes = n_timesteps // n_step
    rewards_np = np.array(reward_buffer)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for user in range(n_users):
        ax = axes[user // 5, user % 5]
        user_rewards = rewards_np[:, user]

        episodes_avg = [
            np.mean(user_rewards[i*n_step:(i+1)*n_step])
            for i in range(n_episodes)
        ]

        ax.plot(range(n_episodes), episodes_avg, label=f"User {user+1}")
        ax.set_title(f"User {user+1}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward")
        ax.grid(True)
        ax.legend()

    fig.suptitle("Average Reward per Episode for Each User", fontsize=16)
    plt.tight_layout()
    plt.show()
