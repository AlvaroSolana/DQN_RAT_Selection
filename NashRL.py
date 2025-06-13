import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc
import math
from RAT_env import *
from tqdm import tqdm

from NashAgent_lib import *

import os

device=torch.device("cpu")

# -------------------------------------------------------------------
# This file executes the Nash-DQN Reinforcement Learning Algorithm
# -------------------------------------------------------------------

# Define truncation function
def expand_list(state, n_users):
    """
    Creates a matrix of features for a batch of input states of the environment.
    Specifically, given a list of states, returns a matrix of features structured as follows:
        - Blocks of matrices stacked vertically where each block represents the features for one element
          in the batch
        - Each block is separated into N rows where N is the number of players
        - Each i'th row in each block is structured such that the first three elements are the non-permutation-invariant features
    :param state:    List of states to pass pass into NN later
    :return:         Matrix of the batch of features structured to be pass into NN
    """
    states = []
    for i in range(0, n_users):
        s = state.to_sep_numpy(i)
        states.append(s)
    
    return torch.stack(states).float()


def run_Nash_Agent(rat_env, max_steps, sim_steps, exploration_fraction, buffer_size, AN_file_name, VN_file_name):
    """
    Runs the nash RL algothrim and outputs two files that hold the network parameters
    for the estimated action network and value network
    :param num_sim:           Number of Simulations
    :param batch_update_size: Number of experiences sampled at each time step
    :param buffersize:        Maximum size of replay buffer
    :return: Truncated Array
    """
    n_agents = rat_env.n_users
#------------for debugging and for visualization-----------
    negative_reward_count = 0  # 
    reward_values = [] # store training rewards
    episode_rewards = [] # store training rewards 
    episode_reward = np.zeros(n_agents)
    last_rats = [0]*rat_env.n_stations # store last rats chosen
    best_actions = [0]*rat_env.n_stations 
    #------------------------
    nash_agent = NashNN(n_users=n_agents, n_stations = rat_env.n_stations)
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
        else: # choose best action (nash action)
            q_values = nash_agent.predict_action(state).detach()#[:,:rat_env.n_stations]).detach()
            actions = torch.argmax(q_values, dim=1)
            best_actions = rat_env.get_rat_chosen(actions,best_actions) # For visualization
        
        last_rats = rat_env.get_rat_chosen(actions,last_rats) # For visualization
        
        # Take an step with the chosen action
        current_state,_, new_state, lr = rat_env.step(actions.detach())
        cur_s = expand_list(current_state,n_agents)
        next_s = expand_list(new_state, n_agents)
        negative_reward = torch.any(torch.eq(lr, -0.1))

        # -------- For debugging --------
        episode_reward += lr.numpy()
        #---------------------------------- 
        #   
        if rat_env.iteration == (max_steps-1) or negative_reward:
            isLastState = torch.ones(n_agents,dtype=torch.float32).to(device)
            # -------- For debugging --------
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
        #replay_buffer.add(cur_s[:, :rat_env.n_stations], next_s[:, :rat_env.n_stations], isLastState, rewards, actions)
        replay_buffer.add(cur_s, next_s, isLastState, rewards, actions)

        learning_starts = buffer_size
        if global_step > learning_starts: # Buffer is full, we can start learning
            if global_step % 1 == 0:
                gamma = 0.9 # tuned
                buffer_sample = replay_buffer.sample(128)
                current_state_list, next_state_list, isLastState_list, rewards_list, act_list = buffer_sample                
                with torch.no_grad():
                    target_max,_ = nash_agent.predict_value(next_state_list).max(dim=2)
                    td_target = rewards_list.flatten() + gamma * target_max.flatten() * (1 - isLastState_list.flatten())
                act_list = act_list.long()
                old_value = nash_agent.predict_action(current_state_list).gather(2, act_list.unsqueeze(-1)).squeeze(-1).flatten()
                loss = F.mse_loss(td_target,old_value)
                    
                # Optimization step
                nash_agent.optimizer_DQN.zero_grad()
                loss.backward()
                nash_agent.optimizer_DQN.step()

                sum_loss.append(loss.item()) # store loss for visualization
            
            # Update target network
            tau = 5*1e-1 #1e-2  #5e-2 1e-3 or 5e-3, higher tau high variabilty and more error
            for target_param, q_param in zip(nash_agent.value_net.parameters(), nash_agent.action_net.parameters()):
                target_param.data.copy_(
                    tau * q_param.data + (1.0 - tau) * target_param.data
                )   
            if  global_step %(sim_steps/10) == 0:
                print(f"Iteration {global_step} A_Loss: {loss}")         

    # ------------For visualization-------------
    import csv
    print("csv imported")    
    with open('rewards.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["step"] + [f"reward_{i}" for i in range(1, rat_env.n_users +1 )])
        for i, reward in enumerate(reward_values):
            writer.writerow([i] + reward)

    print("Saving final weights to disk")
    torch.save(nash_agent.action_net.state_dict(), AN_file_name + ".pt")
    torch.save(nash_agent.value_net.state_dict(), VN_file_name + ".pt")
    print("Weights saved to disk")
    # ---------------------------------------------

    return nash_agent, sum_loss, last_rats, episode_rewards, best_actions,eps_list,episode_length


## Previously used code that may be useful in the future

'''

            # Save weights
            save_flag = not (global_step+1) % (sim_steps/5)
            if save_flag:
                torch.save(nash_agent.action_net.state_dict(),
                            AN_file_name + "_" + str(global_step) + ".pt")
                torch.save(nash_agent.value_net.state_dict(), VN_file_name + "_" + str(global_step) + ".pt")
                print(f"Weights saved to disk (Checkpoint)")  


            # -------- For debugging --------
            # This was under choosing actions (both random and best)
                    lte_select = False
                ap_select = False
                for action in actions:  
                    if action < rat_env.n_ltesn:
                        best_lte_chosen += 1
                        lte_select = True
                    else:
                        best_aps_chosen += 1
                        ap_select = True               
            if ap_select:
                ap_step+=1
            if lte_select:
                lte_step+=1
            
            # This was at the end of the loop (after the step)
            if (global_step%print_idx == 0) or global_step==(sim_steps-1):
                aps_chosen = best_aps_chosen + rand_aps_chosen
                lte_chosen = best_lte_chosen + rand_lte_chosen
                disconnected = rand_disconnected + best_disconnected
                if aps_chosen == 0:
                    aps_chosen = 1
                if disconnected == 0:
                    disconnected =1
                if lte_chosen == 0:
                    lte_chosen =1
                print(f"Iteration {global_step - print_idx} to {global_step}. Total Steps = {step_counter}. Total stations selected = {aps_chosen + lte_chosen}(APs + LTEs)")
                print()
                print(f"APs were selected {aps_chosen} times across {ap_step} steps. In {disconnected} steps we had disconnection ")
                print(f"Only {aps_chosen-negative_reward_count} APs didn´t got negative reward(user connected)")
                print(f"--> {rand_aps_chosen / aps_chosen * 100:.2f}% of APs selected randomly | {best_aps_chosen / aps_chosen * 100:.2f}% selected by the NN as best action")
                print(f"--> {rand_disconnected / disconnected * 100:.2f}% of disconnections are from random actions | {best_disconnected / disconnected * 100:.2f}% from best actions")
                print()
                print(f"LTE were selected {lte_chosen} times across {lte_step} steps")
                print(f"--> {rand_lte_chosen / lte_chosen * 100:.2f}% selected randomly | {best_lte_chosen / lte_chosen * 100:.2f}% selected by the NN as best action" )
                print()
                print("-----------------------------------------------------------------")

                rand_aps_chosen, rand_lte_chosen, best_aps_chosen, best_lte_chosen, best_disconnected, rand_disconnected, step_counter, ap_step, lte_step,negative_reward_count = [0] * 10
            
            # --------------------------------------

            best_action_net = None
            best_value_net = None
            best_loss = None
            best_idx = None
            impv_counter = 0

            # Early stop (not used)
            if early_stop:
                if best_loss is None or loss < best_loss:
                    print("New best loss: " + loss)
                    best_loss = dc(loss)
                    best_action_net = dc(nash_agent.action_net.state_dict())
                    best_value_net = dc(nash_agent.value_net.state_dict())
                    best_idx = global_step
                    impv_counter = 0    
                    if global_step > 100:
                        torch.save(best_action_net, AN_file_name +"_" + str(best_idx) + "_best.pt")
                        torch.save(best_value_net, VN_file_name + "_" + str(best_idx) + "_best.pt")
                        print("Weights saved to disk")   
                    else:
                        impv_counter += 1
                        if impv_counter > early_lim:
                            print("EARLY STOPPING ON ITERATION " + str(global_step))
                            print("Saving final weights to disk")
                            torch.save(best_action_net, AN_file_name +"_" + str(best_idx) + "_best.pt")
                            torch.save(best_value_net, VN_file_name + "_" + str(best_idx) + "_best.pt")
                            print("Weights saved to disk")
                            return nash_agent, sum_loss

'''

''' # Check if no AP was selected by any agent
no_APs = 0
for agent in range(0,n_agents):
    aps = state[agent, 4:rat_env.n_stations]
    if (aps == 0).all() :
        print("all zeros")
        no_APs +=1
    '''

'''#Action masking no longer used
random_a = torch.zeros(n_agents)
for i in range(n_agents):
    available = (rat_env.rate[i] != 0).nonzero(as_tuple=True)[0].tolist()
    if len(available) > 1:
        choices = [x for x in available]
        selected = random.choice(choices)
        random_a[i] = selected
    else:
        random_a[i] = random.randint(0,rat_env.n_stations)
a = random_a

        negative_reward_count += torch.sum(torch.eq(lr, -0.1)).item()
        if negative_reward:
            if rand_action_flag:
                rand_disconnected+=1
            else:
                best_disconnected+=1
        reward_values.append(lr.tolist())

'''