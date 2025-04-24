import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc
import math
from RAT_env import *

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


def run_Nash_Agent(rat_env, max_steps, nash_agent, sim_steps, exploration_fraction, buffer_size, AN_file_name, VN_file_name,rv_min, rv_max, path, early_stop, early_lim):
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
    rand_aps_chosen, rand_lte_chosen, best_aps_chosen, best_lte_chosen, best_disconnected, rand_disconnected, step_counter, ap_step, lte_step,negative_reward_count = [0] * 10
    print_idx = sim_steps/5
    reward_values = [] # store training rewards
    episode_rewards = [] # store training rewards 
    episode_reward = np.zeros(n_agents)
    last_rats = [0]*rat_env.n_stations # store last rats chosen
    best_actions = [0]*rat_env.n_stations 
    #------------------------

    #os.makedirs(os.path.dirname(path), exist_ok=True)
    if nash_agent is None:
        nash_agent = NashNN(n_users=n_agents, n_stations = rat_env.n_stations)
    replay_buffer = ExperienceReplay(buffer_size)
    sum_loss = [] #list to store episode loss
    ep = 1
    min_ep = 0.05
    eps_list = []
    # ---------- Main simulation Block -----------------
    for global_step in range(0, sim_steps):
        eps = max(ep-(ep-min_ep)*global_step/sim_steps/exploration_fraction, min_ep) #exploration rate
        eps_list.append(eps)
        current_state, lr, _ = rat_env.get_state()
        state = expand_list(current_state,n_agents)
        
        rand_action_flag = np.random.random() < eps # Choose exploration or exploitation
        if rand_action_flag:
            nash_a = nash_agent.predict_action(state[:,:rat_env.n_stations])[:,4].detach()
            noise = (torch.rand(n_agents)-0.5)*(sim_steps - global_step)/(sim_steps)
            a = nash_a + noise
            a = torch.clamp(a, min=0, max=1)
            #------------- FOR DEBUGGING--------------
            lte_select = False
            ap_select = False
            for i in range(n_agents):  
                rat_choice = int(torch.round(a[i]*(rat_env.n_stations-1)))
                if rat_choice < rat_env.n_ltesn:
                    rand_lte_chosen += 1
                    lte_select = True
                else:
                    rand_aps_chosen += 1
                    ap_select = True               
            # --------------------------------------- 
        
        else: # choose best action (nash action)
            nash_a = nash_agent.predict_action(state[:,:rat_env.n_stations])[:,4].detach()
            a = torch.clamp(nash_a, min=0, max=1)
        #---------------Debugging--------------------------
            rat_env.get_rat_chosen(a,best_actions)
            lte_select = False
            ap_select = False
            for i in range(n_agents):   
                rat_choice = int(torch.round(a[i]*(rat_env.n_stations-1)))
                if rat_choice < rat_env.n_ltesn:
                    best_lte_chosen += 1
                    lte_select = True
                else:
                    best_aps_chosen += 1
                    ap_select = True               
        if ap_select:
            ap_step+=1
        if lte_select:
            lte_step+=1

        last_rats = rat_env.get_rat_chosen(a,last_rats)
        #---------------------------------- 
         
        a = a.to(device)
        # Take an step with the chosen action
        current_state, a, new_state, lr = rat_env.step(a.detach())
        cur_s = expand_list(current_state,n_agents)
        next_s = expand_list(new_state, n_agents)
        negative_reward = torch.any(torch.eq(lr, -1))
        # -------- For debugging --------
        negative_reward_count += torch.sum(torch.eq(lr, -1)).item()
        if negative_reward:
            if rand_action_flag:
                rand_disconnected+=1
            else:
                best_disconnected+=1
        reward_values.append(lr.tolist())
        episode_reward += lr.numpy()
        #----------------------------------   
        if rat_env.iteration == (max_steps-1) or negative_reward:
            isLastState = torch.ones(n_agents,dtype=torch.float32).to(device)
            # -------- For debugging --------
            episode_reward = episode_reward/(rat_env.iteration+1)
            episode_rewards.append(episode_reward)
            episode_reward = np.zeros(n_agents)
            #---------------------------------- 
            rat_env.reset()

        else:
            isLastState = torch.zeros(n_agents,dtype=torch.float32).to(device)
            rat_env.iteration = rat_env.iteration + 1
            
    
        # Add step results to the buffer
        rewards = lr.detach()
        action = a.detach()
        cur_s = cur_s.detach()
        next_s = next_s.detach()
        replay_buffer.add(cur_s[:,:rat_env.n_stations], next_s[:,:rat_env.n_stations], isLastState, rewards, action)
        
        learning_starts = buffer_size
        if global_step > learning_starts: # Buffer is full, we can start learning
            if global_step % 10 == 0:
                replay_sample = replay_buffer.sample(128)
                # Train Value Network
                nash_agent.value_net.train()
                nash_agent.action_net.eval()
                nash_agent.optimizer_value.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                vloss = nash_agent.compute_value_Loss(replay_sample)
                vloss.backward()
                torch.nn.utils.clip_grad_norm_(nash_agent.value_net.parameters(), max_norm=1.0)
                nash_agent.optimizer_value.step()

                # Train action network
                nash_agent.action_net.train()
                nash_agent.value_net.eval()
                nash_agent.optimizer_DQN.zero_grad()
                loss = nash_agent.compute_action_Loss(replay_sample)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nash_agent.action_net.parameters(), max_norm=1.0)
                nash_agent.optimizer_DQN.step()
                
                # Return networks to train mode
                nash_agent.value_net.train()  
                nash_agent.action_net.train()
            
                sum_loss.append(loss.item())
            
            # Update slow_network
            if global_step % 50 == 0:
                last_loss = sum_loss[-1]
                if last_loss < 1e4:
                    nash_agent.update_slow()

            if  global_step %(sim_steps/10) == 0:
                print(f"Iteration {global_step} A_Loss: {loss} | V_Loss: {vloss}")
    
            # Save weights
            save_flag = not (global_step+1) % (sim_steps/5)
            if save_flag:
                torch.save(nash_agent.action_net.state_dict(),
                            AN_file_name + "_" + str(global_step) + ".pt")
                torch.save(nash_agent.value_net.state_dict(), VN_file_name + "_" + str(global_step) + ".pt")
                print(f"Weights saved to disk (Checkpoint)")           
            '''
            # -------- For debugging --------
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
            '''
            # --------------------------------------

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

    return nash_agent, sum_loss, last_rats, episode_rewards, best_actions,eps_list


## Previously used code that may be useful in the future

'''

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
'''