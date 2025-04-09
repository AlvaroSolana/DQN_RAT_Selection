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


def run_Nash_Agent(rat_env, max_steps, nash_agent, num_sim, AN_file_name, VN_file_name,rv_min, rv_max, path, early_stop, early_lim, mini_batch):
    """
    Runs the nash RL algothrim and outputs two files that hold the network parameters
    for the estimated action network and value network
    :param num_sim:           Number of Simulations
    :param batch_update_size: Number of experiences sampled at each time step
    :param buffersize:        Maximum size of replay buffer
    :return: Truncated Array
    """

    n_agents = rat_env.n_users
    max_T = max_steps

    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    

    if nash_agent is None:
        # Set number of output variables needed from net: ( c1 + c2 + c3 + mu)
        #parameter_number = 4 used before for the output dimensions of the NN
        nash_agent = NashNN(n_users=n_agents, n_stations = rat_env.n_stations,max_steps=max_steps)

    good_action_net_w = nash_agent.action_net.state_dict()
    good_value_net_w = nash_agent.value_net.state_dict()
    
    best_action_net = None
    best_value_net = None
    best_loss = None
    best_idx = None
    impv_counter = 0

    sum_loss = np.zeros(num_sim)
    total_l = 0

    reward_values = [] # To store the rewards during the training
    episode_rewards = []
    last_rats = [0]*rat_env.n_stations

    ep = 0.9
    min_ep = 0.1

    # for debugging
    rand_aps_chosen, rand_lte_chosen, best_aps_chosen, best_lte_chosen, best_disconnected, rand_disconnected, step_counter, ap_step, lte_step,negative_reward_count = [0] * 10
    print_idx = num_sim/5
    # ---------- Main simulation Block -----------------
    for k in range(0, num_sim):
        # Decays Exploration rate Linearly and Resets Loss
        eps = max(max(ep - (ep- min_ep )*(k/(num_sim-1)), 0), min_ep)
        total_l = 0

        print_flag = (not k % 50) and k > 0  # Sets Print Flag - Prints simulation results every 20 simuluations
        if print_flag:
            #update slow value network  UPDATE TARGET NETWORK
            last_loss = sum_loss[k-1]
            if k<1000 or (last_loss < 1e4 and last_loss > 100):
                # update slow network
                nash_agent.update_slow()
            elif last_loss < 100:
                # record last good point
                nash_agent.update_slow()
                good_value_net_w = dc(nash_agent.value_net.state_dict())
                good_action_net_w = dc(nash_agent.action_net.state_dict())
            else:
                # reset
                print("RESETTING WEIGHTS")
                if good_action_net_w is not None and good_value_net_w is not None:
                    nash_agent.value_net.load_state_dict(good_value_net_w)
                    nash_agent.action_net.load_state_dict(good_action_net_w)
                    nash_agent.update_slow()
                    print(nash_agent.value_net.state_dict())
                    print(nash_agent.action_net.state_dict())
                else:
                    print("CANNOT RESET, NO SAVE POINT")
                
            
        cur_s_buffer=torch.empty(0).to(device)
        next_s_buffer=torch.empty(0).to(device)
        term_flag_buffer=torch.empty(0).to(device)
        rewards_buffer=torch.empty(0).to(device)
        action_buffer=torch.empty(0).to(device)

        for i in range(0, mini_batch): # 10 episodes
            rat_env.reset()
            episode_reward = np.zeros(n_agents)
            for t in range(0, max_T):  # 10 step
                step_counter+=1 # just for debugging        
                rat_env.iteration = t
                current_state, lr, _ = rat_env.get_state()
                state = expand_list(current_state,n_agents)
                no_APs = 0
                for agent in range(0,n_agents):
                    aps = state[agent, 4:rat_env.n_stations]
                    if (aps == 0).all() :
                        print("all zeros")
                        no_APs +=1


                nash_a = nash_agent.predict_action(state)[:,4].detach()

                rand_action_flag = np.random.random() < eps
                if rand_action_flag:
                    noise = torch.randn(nash_a.size()) * (rv_max - (rv_max-rv_min)*k/num_sim)
                    a = nash_a + noise                   
                    #------------- FOR DEBUGGING---------------------------
                    lte_select = False
                    ap_select = False
                    for i in range(n_agents):   
                        a = torch.clamp(a, min=0, max=(rat_env.n_stations-1))
                        rat_choice = int(torch.round(a[i]))
                        if rat_choice < rat_env.n_ltesn:
                            rand_lte_chosen += 1
                            lte_select = True
                        else:
                            rand_aps_chosen += 1
                            ap_select = True               
                    # --------------------------------------- 
                else:
                    a = nash_a

                    #------------- FOR DEBUGGING---------------------------
                    lte_select = False
                    ap_select = False
                    for i in range(n_agents):   
                        a = torch.clamp(a, min=0, max=(rat_env.n_stations-1))
                        rat_choice = int(torch.round(a[i]))
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
                negative_reward_count += torch.sum(torch.eq(lr, -1)).item()

                # -------- For debugging --------
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
                    episode_reward = episode_reward/(t+1) ## for debugging
                else:
                    isLastState = torch.zeros(n_agents,dtype=torch.float32).to(device)
          
                rewards = lr.detach()
                action = a.detach()
                cur_s = cur_s.detach()
                next_s = next_s.detach()

                cur_s_buffer=torch.cat([cur_s_buffer, cur_s], dim = 0)
                next_s_buffer=torch.cat([next_s_buffer, next_s], dim = 0)
                term_flag_buffer=torch.cat([term_flag_buffer, torch.unsqueeze(isLastState,0)], dim = 0)
                rewards_buffer=torch.cat([rewards_buffer, torch.unsqueeze(rewards,0)], dim = 0)
                action_buffer=torch.cat([action_buffer, torch.unsqueeze(action,0)], dim = 0)

                #### DEbugging block ####

                if (num_sim-k) < 300:
                    last_rats = rat_env.get_rat_chosen(a,last_rats)

                if negative_reward:
                    break

                # Prints Some Information
                #if (print_flag) and i == print_idx:
               #     print("Rewards: {}".format(rewards))
            episode_rewards.append(episode_reward)

        if (k%print_idx == 0 and k>0) or k==(num_sim-1):
            aps_chosen = best_aps_chosen + rand_aps_chosen
            lte_chosen = best_lte_chosen + rand_lte_chosen
            disconnected = rand_disconnected + best_disconnected
            if aps_chosen == 0:
                aps_chosen = 1
            if disconnected == 0:
                disconnected =1
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



        #print(f"Epoch total reward {rewards_buffer.sum()}")
        #print(f"Iteration {k} : All users chose station 0 {all_zeros}/100 times before adding noise")

        nash_agent.value_net.train()
        #nash_agent.action_net.train()
        nash_agent.action_net.eval()
        
        # Computes value loss and updates Value network
        replay_sample = (cur_s_buffer, next_s_buffer, term_flag_buffer, rewards_buffer, action_buffer)
            
        nash_agent.optimizer_value.zero_grad()
        vloss = nash_agent.compute_value_Loss(replay_sample)
        vloss.backward()
        torch.nn.utils.clip_grad_norm_(nash_agent.value_net.parameters(), 1e-1)
        nash_agent.optimizer_value.step()

        nash_agent.action_net.train()  # train the action net
        nash_agent.value_net.eval()

        # Computes action loss and updates Action network
        nash_agent.optimizer_DQN.zero_grad()
        loss = nash_agent.compute_action_Loss(replay_sample)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nash_agent.action_net.parameters(), 1e-1)
        nash_agent.optimizer_DQN.step()
        
        nash_agent.value_net.eval()  
        nash_agent.action_net.eval()

        # Calculat Current Step's final Total Loss
        total_l += vloss + loss

        sum_loss[k] = total_l
        '''
        if print_flag:
            print(f"Iteration {k} Loss: {total_l} | V_Loss: {vloss} | A_Loss: {loss}")
        '''
           

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
                
                if k > 1000:
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

    ### Debugging block ###


    return nash_agent, sum_loss, last_rats, episode_rewards
