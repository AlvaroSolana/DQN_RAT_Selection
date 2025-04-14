import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc
import math

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


def run_Nash_Agent(rat_env, n_steps, nash_agent, n_episodes, exploration_fraction, AN_file_name, VN_file_name,rv_min, rv_max, path, early_stop, early_lim):
    """
    Runs the nash RL algothrim and outputs two files that hold the network parameters
    for the estimated action network and value network
    :param num_sim:           Number of Simulations
    :param batch_update_size: Number of experiences sampled at each time step
    :param buffersize:        Maximum size of replay buffer
    :return: Truncated Array
    """

    n_agents = rat_env.n_users
    #os.makedirs(os.path.dirname(path), exist_ok=True)

    if nash_agent is None:
        nash_agent = NashNN(n_users=n_agents, n_stations = rat_env.n_stations)

    good_action_net_w = nash_agent.action_net.state_dict()
    good_value_net_w = nash_agent.value_net.state_dict()
    
    best_action_net = None
    best_value_net = None
    best_loss = None
    best_idx = None
    impv_counter = 0

    sum_loss = np.zeros(n_episodes)
    total_l = 0

    reward_values = [] # store training rewards (for visualization)
    episode_rewards = [] # store training rewards (for visualization)
    last_rats = [0]*rat_env.n_stations # store last rats chosen (for visualization)
    best_actions = [0]*rat_env.n_stations 

    #-----for debugging------
    rand_aps_chosen, rand_lte_chosen, best_aps_chosen, best_lte_chosen, best_disconnected, rand_disconnected, step_counter, ap_step, lte_step,negative_reward_count = [0] * 10
    print_idx = n_episodes/5
    #------------------------

    ep = 1
    min_ep = 0.05
    eps_list = []
    # ---------- Main simulation Block -----------------
    for k in range(0, n_episodes):
        rat_env.reset() # Start a new episode
        total_l = 0
        # Update slow_network if needed
        update_flag = (k % 50) and k > 0
        if update_flag:
            last_loss = sum_loss[k-1]
            if k<1000 or (last_loss < 1e4 and last_loss > 100):
                nash_agent.update_slow()
            elif last_loss < 100:
                nash_agent.update_slow()
                good_value_net_w = dc(nash_agent.value_net.state_dict())
                good_action_net_w = dc(nash_agent.action_net.state_dict())
                
        # Initialize buffers  
        cur_s_buffer=torch.empty(0).to(device)
        next_s_buffer=torch.empty(0).to(device)
        term_flag_buffer=torch.empty(0).to(device)
        rewards_buffer=torch.empty(0).to(device)
        action_buffer=torch.empty(0).to(device)
        
        episode_reward = np.zeros(n_agents) # for visualization

        for step in range(0, n_steps): # 100 steps
            current_steps = k* n_steps + step
            eps = max(max(ep - (ep- min_ep )*((current_steps+1)/(n_episodes*n_steps))/exploration_fraction, 0), min_ep) #exploration rate
            eps_list.append(eps)
            rat_env.iteration = step
            current_state, lr, _ = rat_env.get_state()
            state = expand_list(current_state,n_agents)
         
            
            ''' # Previously used for debugging
            no_APs = 0
            for agent in range(0,n_agents):
                aps = state[agent, 4:rat_env.n_stations]
                if (aps == 0).all() :
                    print("all zeros")
                    no_APs +=1
                '''

            rand_action_flag = np.random.random() < eps # Choose exploration or exploitation
            if rand_action_flag: # Select random action
                #noise = torch.randn(nash_a.size()) * (rv_max - (rv_max-rv_min)*k/n_episodes)
                random_a = torch.zeros(n_agents)
                
                for i in range(n_agents):
                    
                    available = (rat_env.rate[i] != 0).nonzero(as_tuple=True)[0].tolist()
                    if len(available) > 1:
                        choices = [x for x in available]
                        selected = random.choice(choices)
                        random_a[i] = selected
                    #    if rat_env.rate[i,selected] == 0:
                    #        print(" ERROR : station with rate 0 was selected")
                    else:
                    #    random_a[i] = available[0]
                    #    print("No random action available")
                        random_a[i] = random.randint(0,rat_env.n_stations)
                a = random_a
               
                
                #------------- FOR DEBUGGING--------------
                lte_select = False
                ap_select = False
                for i in range(n_agents):   
                    rat_choice = random_a[i]
                    if rat_choice < rat_env.n_ltesn:
                        rand_lte_chosen += 1
                        lte_select = True
                    else:
                        rand_aps_chosen += 1
                        ap_select = True               
                # --------------------------------------- 
            else: # choose best action (nash action)
                
                nash_a = nash_agent.predict_action(state)[:,4].detach()
                a = nash_a + torch.randn(nash_a.size()) * (rv_max - (rv_max-rv_min)*step/k)
                print(nash_a,a)
                #print(torch.round(a))
                rat_env.get_rat_chosen(a,best_actions)

                lte_select = False
                ap_select = False
                actions = torch.clamp(a, min=0, max=(rat_env.n_stations-1))
                for i in range(n_agents):   
                    rat_choice = int(torch.round(actions[i]))
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
            # --------------------------------------- 
            
            a = torch.clamp(a, min=0, max=(rat_env.n_stations-1))
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
            if rat_env.iteration == (n_steps-1) or negative_reward:
                isLastState = torch.ones(n_agents,dtype=torch.float32).to(device)
                episode_reward = episode_reward/(step+1) # for debugging
            else:
                isLastState = torch.zeros(n_agents,dtype=torch.float32).to(device)
        
            rewards = lr.detach()
            action = a.detach()
            cur_s = cur_s.detach()
            next_s = next_s.detach()

            # Add step results to the buffers
            cur_s_buffer=torch.cat([cur_s_buffer, cur_s], dim = 0)
            next_s_buffer=torch.cat([next_s_buffer, next_s], dim = 0)
            term_flag_buffer=torch.cat([term_flag_buffer, torch.unsqueeze(isLastState,0)], dim = 0)
            rewards_buffer=torch.cat([rewards_buffer, torch.unsqueeze(rewards,0)], dim = 0)
            action_buffer=torch.cat([action_buffer, torch.unsqueeze(action,0)], dim = 0)

            #------for debugging---------
            if (n_episodes-k) < n_episodes:
                last_rats = rat_env.get_rat_chosen(a,last_rats)
            #----------------------------------           

            if negative_reward:
                break

        episode_rewards.append(episode_reward) # for visualization

        # -------- For debugging --------
        if (k%print_idx == 0 and k>0) or k==(n_episodes-1):
            aps_chosen = best_aps_chosen + rand_aps_chosen
            lte_chosen = best_lte_chosen + rand_lte_chosen
            disconnected = rand_disconnected + best_disconnected
            if aps_chosen == 0:
                aps_chosen = 1
            if disconnected == 0:
                disconnected =1
            if lte_chosen == 0:
                lte_chosen =1
            print(f"Iteration {k - print_idx} to {k}. Total Steps = {step_counter}. Total stations selected = {aps_chosen + lte_chosen}(APs + LTEs)")
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


        replay_sample = (cur_s_buffer, next_s_buffer, term_flag_buffer, rewards_buffer, action_buffer) 
        nash_agent.value_net.train()
        nash_agent.action_net.eval()
        nash_agent.optimizer_value.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        vloss = nash_agent.compute_value_Loss(replay_sample)
        vloss.backward()
        torch.nn.utils.clip_grad_norm_(nash_agent.value_net.parameters(), max_norm=1.0)
        nash_agent.optimizer_value.step()

        nash_agent.action_net.train()
        nash_agent.value_net.eval()

            # Computes action loss and updates Action network
        nash_agent.optimizer_DQN.zero_grad()
        loss = nash_agent.compute_action_Loss(replay_sample)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nash_agent.action_net.parameters(), max_norm=1.0)
        nash_agent.optimizer_DQN.step()

        nash_agent.value_net.train()  
        nash_agent.action_net.train()

            # Calculat Current Step's final Total Loss
        total_l += loss
        sum_loss[k] = total_l


        if  k%(n_episodes/10) == 0:
            print(f"Iteration {k} Loss: {total_l} | V_Loss: {vloss} | A_Loss: {loss}")


        # Set Save Flag
        save_flag = not (k+1) % 500
        if save_flag:
            torch.save(nash_agent.action_net.state_dict(),
                       AN_file_name + "_" + str(k) + ".pt")
            torch.save(nash_agent.value_net.state_dict(), VN_file_name + "_" + str(k) + ".pt")
            print("Weights saved to disk")           
        if early_stop:
            if best_loss is None or total_l.item() < best_loss:
                print("New best loss: " + str(total_l.item()))
                best_loss = dc(total_l.item())
                best_action_net = dc(nash_agent.action_net.state_dict())
                best_value_net = dc(nash_agent.value_net.state_dict())
                best_idx = k
                impv_counter = 0    
                if k > 100:
                    torch.save(best_action_net, AN_file_name +"_" + str(best_idx) + "_best.pt")
                    torch.save(best_value_net, VN_file_name + "_" + str(best_idx) + "_best.pt")
                    print("Weights saved to disk")   
                else:
                    impv_counter += 1
                    if impv_counter > early_lim:
                        print("EARLY STOPPING ON ITERATION " + str(k))
                        print("Saving final weights to disk")
                        torch.save(best_action_net, AN_file_name +"_" + str(best_idx) + "_best.pt")
                        torch.save(best_value_net, VN_file_name + "_" + str(best_idx) + "_best.pt")
                        print("Weights saved to disk")
                        return nash_agent, sum_loss

    
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
