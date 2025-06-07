import numpy as np
import torch
from datetime import date
from copy import deepcopy as dc
from RAT_env import *
from tqdm import tqdm
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device=torch.device("cpu")

# -------------------------------------------------------------------
# This file executes the Nash-DQN Reinforcement Learning Algorithm
# -------------------------------------------------------------------

# Define truncation function
def expand_list(state, n_users):
    states = []
    for i in range(0, n_users):
        s = state.to_sep_numpy(i)
        states.append(s)
    
    return torch.stack(states).float()

class PermInvariantQNN(torch.nn.Module):  # invariant features --> do not depend on the order of the agents

    n_users: int
    n_stations: int
    out_dim: int

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.5)
                nn.init.uniform_(m.bias, a=0.5, b=0.5)
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.zeros_(m.bias)
    '''
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.out_dim:  # final layer
                    nn.init.xavier_uniform_(m.weight, gain=5.0)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    '''

    def __init__(self,n_users, n_stations,input_dim, out_dim, lat_dims, layers):
        super(PermInvariantQNN, self).__init__()
        # Store input and output dimensions
        self.n_users = n_users
        self.n_stations = n_stations
        self.out_dim = out_dim 
        nets = []
        nets.append(nn.Linear(input_dim, lat_dims)) # lat_dims = number of neurons per layer
        #nets.append(nn.LeakyReLU(0.1))
        nets.append(nn.ReLU())
        nets.append(nn.LayerNorm(lat_dims))

        for i in range(layers):
            next_lat_dims = max(int(lat_dims // 2), out_dim)  # floor division and enforce a minimum size of out_dim
            nets.append(nn.Linear(lat_dims, next_lat_dims))
            nets.append(nn.LayerNorm(next_lat_dims))
            #nets.append(nn.LeakyReLU(0.1))
            nets.append(nn.ReLU())
            lat_dims = next_lat_dims  # update for next loop

            
        nets.append(nn.Linear(lat_dims, self.out_dim))
        #nets.append(ScaledTanh(n_stations))
        self.decoder_net = nn.Sequential(*nets)
        
        self.initialize_weights()

    def forward(self, input):
        x = self.decoder_net(input)  # Output shape: (n_users, out_dim)
        return x


class NashNN():

    def __init__(self, n_users, n_stations, lat_dims= 512 , c_cons=0.1, c2_cons=True, c3_pos=True, layers=1, weighted_adam=True):
        # Simulation Parameters
        self.n_users = n_users
        self.n_stations = n_stations

        # Initialize Networks
        self.actor_net = PermInvariantQNN(
            self.n_users, self.n_stations, input_dim=self.n_stations ,out_dim=self.n_stations, lat_dims=lat_dims, layers=layers)
        self.critic_net = PermInvariantQNN(
            self.n_users, self.n_stations, input_dim=(self.n_stations+1),out_dim=1, lat_dims=lat_dims, layers=layers)
        self.target_actor = PermInvariantQNN(
            self.n_users, self.n_stations, input_dim=self.n_stations ,out_dim=self.n_stations, lat_dims=lat_dims, layers=layers)
        self.target_critic = PermInvariantQNN(
            self.n_users, self.n_stations, input_dim=(self.n_stations+1),out_dim=1, lat_dims=lat_dims, layers=layers)
        ## Add this at the beggining
        #self.critic_net.load_state_dict(self.action_net.state_dict())

        #self.slow_val_net=PermInvariantQNN(
         #   n_users = self.n_users, n_stations = self.n_stations, out_dim=1, lat_dims=lat_dims, layers=layers)
               
        # Define optimizer used (SGD, etc)
        self.lr = 2.5e-4
        if weighted_adam:
            self.actor_optimizer = optim.AdamW(
                self.actor_net.parameters(), lr=self.lr)

            self.critic_optimizer = optim.AdamW(
                self.critic_net.parameters(), lr=self.lr)
        else:
            self.actor_optimizer = optim.Adam(
                self.actor_net.parameters(), lr=self.lr)

            self.critic_optimizer = optim.Adam(
                self.critic_net.parameters(), lr=self.lr)
        
        #optimizer = optim.Adam(self.action_net.parameters(), lr=args.learning_rate)

        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()


    def predict_action(self, states):
        """
        Predicts the parameters of the advantage function of a batch of environmental states
        :param states:    nm List of environmental state objects
        :return:          List of NashFittedValue objects representing the estimated parameters
        """
        #print(states.shape)
        if len(states.shape) > 2:
            B, A, D = states.shape  # B = batch size (10), A = agents (3), D = 255
            flat_states = states.view(B * A, D)  # Flatten to shape (30, 255)
            # Forward pass through action_net
            action_list = self.actor_net.forward(input=flat_states)
            # Reshape back to (10, 3, -1)
            action_list = action_list.view(B, A, -1)
            return action_list
        else:
            action_list = self.actor_net.forward(input=states) 
            return action_list
        
    def target_action(self, states):
        """
        Predicts the parameters of the advantage function of a batch of environmental states
        :param states:    nm List of environmental state objects
        :return:          List of NashFittedValue objects representing the estimated parameters
        """
        #print(states.shape)
        if len(states.shape) > 2:
            B, A, D = states.shape  # B = batch size (10), A = agents (3), D = 255
            flat_states = states.view(B * A, D)  # Flatten to shape (30, 255)
            # Forward pass through action_net
            action_list = self.target_actor.forward(input=flat_states)
            # Reshape back to (10, 3, -1)
            action_list = action_list.view(B, A, -1)
            return action_list
        else:
            action_list = self.target_actor.forward(input=states) 
            return action_list
        

    def predict_value(self, states,actions):
        """
        Predicts the nash value of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          Tensor of estimated nash values of all agents for the batch of states
        """
        #print("Predicting value")
        #print(f"States shape: {states.shape}, Actions shape: {actions.shape}")
        input = torch.cat((states, actions.unsqueeze(-1)), dim=-1)  # Concatenate states and actions
        #print("Input shape: ", input.shape)
        if len(input.shape) > 2:
            B, A, D = input.shape  # B = batch size (10), A = agents (3), D = 255
            flat_input = input.view(B * A, D)  # Flatten to shape (30, 255)
            # Forward pass through action_net
            critic_values = self.critic_net.forward(input=flat_input)
            # Reshape back to (10, 3, -1)
            critic_values = critic_values.view(B, A, -1)
            return critic_values
        else:
            critic_values = self.critic_net.forward(input=states) 
            return critic_values
    
    def target_value(self, states,actions):
        """
        Predicts the nash value of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          Tensor of estimated nash values of all agents for the batch of states
        """
        #print("Predicting target value")
        #print(f"States shape: {states.shape}, Actions shape: {actions.shape}")
        actions = actions.unsqueeze(-1)  # Now shape is [128, 10, 1]
        input = torch.cat((states, actions), dim=-1)  # Concatenate states and actions
        #print(f"Input shape: {input.shape}")
        if len(input.shape) > 2:
            B, A, D = input.shape  # B = batch size (10), A = agents (3), D = 255
            flat_input = input.view(B * A, D)  # Flatten to shape (30, 255)
            # Forward pass through action_net
            critic_values = self.target_critic.forward(input=flat_input)
            # Reshape back to (10, 3, -1)
            critic_values = critic_values.view(B, A, -1)
            return critic_values
        else:
            critic_values = self.target_critic.forward(input=states) 
            return critic_values



def run_maddpg(rat_env, max_steps, sim_steps, exploration_fraction, buffer_size, AN_file_name, VN_file_name):
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
    reward_values = [] # store training rewards
    episode_rewards = [] # store training rewards 
    episode_reward = np.zeros(n_agents)
    last_rats = [0]*rat_env.n_stations # store last rats chosen
    best_actions = [0]*rat_env.n_stations 
    #------------------------

    agents = [NashNN(n_users=n_agents, n_stations = rat_env.n_stations)]* n_agents # Create a list of NashNN agents, one for each user
    
    #optimizer = torch.optim.Adam(nash_agent.actor_net.parameters(), lr=2.5e-4)
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
        actions = torch.zeros(n_agents, dtype=torch.int64)
        for i,actor in enumerate(agents):
            rand_action_flag = np.random.random() < eps # Choose exploration or exploitation
            if rand_action_flag:
                actions[i] = torch.randint(0, rat_env.n_stations - 1, (1,))
            else: # choose best action (nash action)
                action_values = actor.predict_action(state[i,:rat_env.n_stations]).detach()
                actions[i] = torch.argmax(action_values, dim=0)
                
        best_actions = rat_env.get_rat_chosen(actions,best_actions) # For visualization
        last_rats = rat_env.get_rat_chosen(actions,last_rats) # For visualization
        
        # Take an step with the chosen action
        current_state,_, new_state, lr = rat_env.step(actions.detach())
        cur_s = expand_list(current_state,n_agents)
        next_s = expand_list(new_state, n_agents)
        negative_reward = torch.any(torch.eq(lr, -0.1))

        # -------- For debugging --------
        negative_reward_count += torch.sum(torch.eq(lr, -0.1)).item()
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
        #print(f"Current state: {cur_s[:, :rat_env.n_stations]} with rewards {rewards} and actions {actions}")
        replay_buffer.add(cur_s[:, :rat_env.n_stations], next_s[:, :rat_env.n_stations], isLastState, rewards, actions)

        learning_starts = buffer_size
        if global_step > learning_starts: # Buffer is full, we can start learning
            #if global_step % 10 == 0:
            gamma = 0.99
            buffer_sample = replay_buffer.sample(128)
            current_state_list, next_state_list, isLastState_list, rewards_list, act_list = buffer_sample                
            for i, agent in enumerate(agents):
                with torch.no_grad():
                    target_actions = []
                    for j, other_agent in enumerate(agents):
                        next_state = next_state_list[:, j, :]
                        target_act = other_agent.target_action(next_state)  
                        target_actions.append(torch.argmax(target_act, dim=-1, keepdim=True))  # Shape: [batch_size, 1]
                    joint_target_actions = torch.cat(target_actions, dim=1)
                    # Use target critic
                    critic_value = agent.target_value(next_state_list, joint_target_actions)
                    td_target = rewards_list.flatten() + gamma * critic_value.flatten() * (1 - isLastState_list.flatten())
                act_list = act_list.long()
                current_value = agent.predict_value(current_state_list,act_list)
                # Critic loss calculation
                critic_loss = F.mse_loss(td_target,current_value)                
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                agent.critic_optimizer.step()
                # Actor loss calculation
                agent_states = current_state_list[:, i, :].unsqueeze(1)  # shape: [128, 1, 13]
                action_values = agent.predict_action(agent_states).squeeze(0)
                # ---- ACTOR UPDATE ----
                agent_state = current_state_list[:, i, :]  # shape: [batch_size, state_dim]
                # Predict this agent's action from its actor
                predicted_action = agent.predict_action(agent_state).squeeze(0)  # shape: [batch_size, action_dim]
                all_actions = []
                for j, other_agent in enumerate(agents):
                    if j == i:
                        all_actions.append(torch.argmax(predicted_action, dim=-1, keepdim=True))
                    else:
                        # Detach other agents' actions (no gradient needed)
                        other_state = current_state_list[:, j, :]
                        other_action = other_agent.predict_action(other_state).detach()
                        all_actions.append(torch.argmax(other_action, dim=-1, keepdim=True))
                # Concatenate joint actions
                joint_actions = torch.cat(all_actions, dim=1)  # shape: [batch_size, total_action_dim]
                # Evaluate joint action using this agent's critic
                actor_loss = -agent.predict_value(current_state_list, joint_actions).mean()
                # Optimize actor
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                agent.actor_optimizer.step()
                                
                if i==3: # Only print the loss for one agent
                    sum_loss.append(actor_loss.item()) # store loss for visualization
                    if  global_step %(sim_steps/10) == 0:
                        print(f"Iteration {global_step} Actor_Loss: {actor_loss}")

                # ---- TARGET NETWORK SOFT UPDATE ----
                # You can tune tau (e.g., 0.005 or 0.01)
                tau = 0.01
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for target_param, param in zip(agent.target_critic.parameters(), agent.critic_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                


    # ------------For visualization-------------
    import csv
    print("csv imported")    
    with open('rewards.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["step"] + [f"reward_{i}" for i in range(1, rat_env.n_users +1 )])
        for i, reward in enumerate(reward_values):
            writer.writerow([i] + reward)
    # ---------------------------------------------

    print("Saving final weights to disk")
    for i, agent in enumerate(agents):
        torch.save(agent.actor_net.state_dict(), AN_file_name + f"{i}.pt")
    print("Weights saved to disk")

    return agents, sum_loss, last_rats, episode_rewards, best_actions,eps_list,episode_length

