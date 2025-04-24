import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random
from copy import deepcopy as dc

device=torch.device("cpu")
import torch.nn.functional as F
'''
class ScaledSigmoid(nn.Module):
    def __init__(self, N):
        super(ScaledSigmoid, self).__init__()
        self.N = N - 1  # The scaling factor to scale last column to [0, N]

    def forward(self, x):
        x = torch.sigmoid(x)  # Apply sigmoid to the entire output
        # Make a copy and scale only the last column safely
        x_scaled = x.clone()
        x_scaled[:, -1] = x[:, -1] * self.N
        return x_scaled
    
class ScaledTanh(nn.Module):
    def __init__(self):
        super(ScaledTanh, self,N).__init__()
        self.N=N
        
    def forward(self, x):
       
        return (torch.tanh(x)+1)*self.N/2
    
class BoundedReLU(nn.Module):
    def __init__(self, upper_bound):
        super().__init__()
        self.upper_bound = upper_bound

    def forward(self, x):
        return torch.clamp(F.relu(x), 0, self.upper_bound)
'''
    
class PermInvariantQNN(torch.nn.Module):  # invariant features --> do not depend on the order of the agents
    """
    Permutation Invariant Network
    :param n_users:        Number of agents in the scenario
    :param n_stations:     Number of stations in the scenario
    :param out_dim:        Dimension of output
    :param block_size:     Number of invariant features of each agent
    :param num_moments:    Number of features/moments to summarize invariant features of each agent
    """
    n_users: int
    n_stations: int
    out_dim: int
    block_size: int
    num_moments: int

         
    '''
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, mean=0.0, std=0.5)
                #nn.init.uniform_(m.bias, a=0.5, b=0.5)
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    '''
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.out_dim:  # final layer
                    nn.init.xavier_uniform_(m.weight, gain=5.0)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def __init__(self,n_users, n_stations, out_dim, lat_dims, layers):
        super(PermInvariantQNN, self).__init__()
        # Store input and output dimensions
        self.n_users = n_users
        self.n_stations = n_stations
        self.out_dim = out_dim 
        nets = []
        input_dim = n_stations #* 3 # number of features/columns of the state space (rate , station_id , rate_type)
        nets.append(nn.Linear(input_dim, lat_dims)) # lat_dims = number of neurons per layer
        #nets.append(nn.LeakyReLU(0.1))
        nets.append(nn.ReLU())
        nets.append(nn.LayerNorm(lat_dims))

        for i in range(layers):
            next_lat_dims = max(int(lat_dims // 2), 16)  # floor division and enforce a minimum size of 16
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
        #x = torch.tanh(x)
        #return (torch.tanh(x)+1)/2
        x[:, :-1] = torch.tanh(x[:, :-1]) # Advantage outputs [-1, 1]
        x[:, -1] = torch.sigmoid(x[:, -1]) # Action output [0, 1]
        return x


class NashNN():
    """
    Object summarizing estimated parameters of the advantage function, initiated 
    through a vector of inputs
    :param non_invar_dim:Number of total non invariant (i.e. market state) input features
    :param output_dim:   Number of total parameters to be estimated via NN
    :param nump:         Number of agents
    :param t:            Number of total time steps
    :param t_cost:       Transaction costs (estimated or otherwise)
    :param term_cost:    Terminal costs (estimated or otherwise)
    """

    def __init__(self, n_users, n_stations, lr=1e-4, lat_dims= 128, c_cons=0.1, c2_cons=True, c3_pos=True, layers=4, weighted_adam=True):
        # Simulation Parameters
        self.lr = lr
        self.n_users = n_users
        self.n_stations = n_stations
        self.output_dim = 4 + 1

        # Initialize Networks
        self.action_net = PermInvariantQNN(
            n_users = self.n_users, n_stations = self.n_stations, out_dim=self.output_dim, lat_dims=lat_dims, layers=layers)
        self.value_net = PermInvariantQNN(
            n_users = self.n_users, n_stations = self.n_stations, out_dim=1, lat_dims=lat_dims, layers=layers)
        self.slow_val_net=PermInvariantQNN(
            n_users = self.n_users, n_stations = self.n_stations, out_dim=1, lat_dims=lat_dims, layers=layers)
                
        # Define optimizer used (SGD, etc)
        if weighted_adam:
            self.optimizer_DQN = optim.AdamW(
                self.action_net.parameters(), lr=self.lr)

            self.optimizer_value = optim.AdamW(
                self.value_net.parameters(), lr=self.lr)
        else:
            self.optimizer_DQN = optim.Adam(
                self.action_net.parameters(), lr=self.lr)

            self.optimizer_value = optim.Adam(
                self.value_net.parameters(), lr=self.lr)

        # Define loss function (Mean-squared, etc)
        self.criterion = nn.MSELoss()
        
        # Define constant L-2 penalty
        self.c_cons = c_cons
        self.c2_cons = c2_cons
        self.c3_pos = c3_pos


    def matrix_slice(self, X):
        """
        Returns a matrix where each row in X is replicated N number of times where N
        is the number of total agents. Then the value of the j'th element of the 
        (i*N + j)'th row is removed. Effectively creating
        a stacked version of the u^(-1) or mu(-1) for batched inputs.
        :param X:    Matrix of actions/nash actions where each row corresponds to one transition from a batch input
        :return:     Matrix of u^(-1) or mu(-1) as described above
        """
        num_entries = len(X)
        arr = X.repeat_interleave(self.n_users, dim = 0)
        ids = torch.arange(self.n_users).clone().detach().tile(num_entries)
        #ids = torch.tensor(torch.arange(self.n_users)).tile(num_entries) #previous version (didn´t work)
        mask = torch.ones_like(arr).scatter_(1, ids.unsqueeze(1), 0.)
        res = arr[mask.bool()].view(-1, self.n_users - 1)
        
        return res


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
            action_list = self.action_net.forward(input=flat_states)

            # Process the output
            action_list = torch.hstack([
                torch.abs(action_list[:, 0]).view(-1, 1),
                action_list[:, 1].view(-1, 1),
                torch.abs(action_list[:, 2]).view(-1, 1),
                action_list[:, 3:]
            ])

            # Reshape back to (10, 3, -1)
            action_list = action_list.view(B, A, -1)
            return action_list
        else:
        
            action_list = self.action_net.forward(input=states) 
            action_list = torch.hstack([torch.abs(action_list[:, 0]).view(-1,1), action_list[:, 1].view(-1,1), torch.abs(action_list[:, 2]).view(-1,1), action_list[:, 3:]])

            return action_list
        

    # ---- Previously used for exploration -------------
    '''
    def get_random_action(self,nash_a ,state):
        """
        Selects random action different from nash_action
        """
        random_action = torch.zeros_like(nash_a, dtype=torch.long)
        rates = state[:,:20]
        for i in range(self.n_users):
            agent_rate = rates[i,:]
            available_stations = torch.nonzero(agent_rate, as_tuple=True)[0]
            original_action = nash_a[i].item()  # Get the original action
            valid_choices = available_stations[available_stations != original_action] # Exclude the original action
            if len(valid_choices) > 0:
                action_chosen = random.choice(valid_choices.tolist())
                random_action[i] = action_chosen
            else:
                random_action[i] = original_action  # If no alternative, keep the original action
        return random_action
    '''

    def update_slow(self):
        self.slow_val_net.load_state_dict(dc(self.value_net.state_dict()))
        # Idea to improve results (didn´t work)
        '''
        tau = 0.05
        for target_param, param in zip(self.slow_val_net.parameters(), self.value_net.parameters()):
           target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        '''

    def predict_value(self, states, slow=False):
        """
        Predicts the nash value of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          Tensor of estimated nash values of all agents for the batch of states
        """
        if slow:
            values = self.slow_val_net.forward(input=states)
        else:
            values = self.value_net.forward(input=states)
                
        return values


    def compute_value_Loss(self, state_tuples):
        """
        Computes the loss function for the value network
        :param state_tuples:    List of a batch of transition tuples
        :return:                Total loss of batch
        """

        cur_state_list = state_tuples[0]
        next_state_list = state_tuples[1]
        isLastState = state_tuples[2].view(-1)
        reward_list = state_tuples[3].view(-1)
        act_list = state_tuples[4].view(-1)

        curAct = self.predict_action(cur_state_list).detach().view(-1,5)
        curVal = self.predict_value(cur_state_list).view(-1) # Nash Value of Current state
        nextVal = self.predict_value(next_state_list, slow=True).detach().view(-1) # Nash Value of Next state

        # Create Lists for predicted Values
        c1_list = curAct[:, 0]
        c2_list = curAct[:, 1]
        c3_list = curAct[:, 2]
        c4_list = curAct[:, 3]
        mu_list = curAct[:, 4]
        
        
        uNeg_list = self.matrix_slice(act_list.view(-1, self.n_users))
        muNeg_list = self.matrix_slice(mu_list.view(-1, self.n_users))

        # Computes the Advantage Function using matrix operations
        A = - c1_list * (act_list-mu_list)**2 - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list, dim=1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2, dim=1) + c4_list * torch.sum((uNeg_list - muNeg_list), dim=1)
        
        # Add entropy regularization on action (mu)
        base_loss = torch.sum(((torch.ones(len(isLastState))-isLastState) * nextVal + reward_list - curVal - A)**2) + torch.sum(c4_list**2)
        action_entropy = - (mu_list * torch.log(mu_list + 1e-8) + (1 - mu_list) * torch.log(1 - mu_list + 1e-8))
        λ = 1e-1  # You can tune this
        
        return base_loss + λ * action_entropy.mean()
        #return torch.sum(((torch.ones(len(isLastState))-isLastState) * nextVal + reward_list - curVal - A)**2) + torch.sum(c4_list**2)
    
    def compute_action_Loss(self, state_tuples):
        """
        Computes the loss function for the action/advantage-function network
        :param state_tuples:    List of a batch of transition tuples
        :return:                Total loss of batch
        """
        
        cur_state_list = state_tuples[0]
        next_state_list = state_tuples[1]
        isLastState = state_tuples[2].view(-1)
        reward_list = state_tuples[3].view(-1)
        act_list = state_tuples[4].view(-1)

        curAct = self.predict_action(cur_state_list).view(-1,5)
        curVal = self.predict_value(cur_state_list).detach().view(-1) # Nash Value of Current state
        nextVal = self.predict_value(next_state_list, slow=True).detach().view(-1) # Nash Value of Next state
        
        c1_list = curAct[:, 0]
        c2_list = curAct[:, 1]
        c3_list = curAct[:, 2]
        c4_list = curAct[:, 3]
        mu_list = curAct[:, 4]
        

        uNeg_list = self.matrix_slice(act_list.view(-1, self.n_users))
        muNeg_list = self.matrix_slice(mu_list.view(-1, self.n_users))
        A = - c1_list * (act_list-mu_list)**2  - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list, dim=1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2, dim=1) + c4_list * torch.sum((uNeg_list - muNeg_list), dim=1)
        
        # Add entropy regularization on action (mu)
        base_loss = torch.sum(((torch.ones(len(isLastState))-isLastState) * nextVal + reward_list - curVal - A)**2) + torch.sum(c4_list**2)
        action_entropy = - (mu_list * torch.log(mu_list + 1e-8) + (1 - mu_list) * torch.log(1 - mu_list + 1e-8))
        λ = 1e-2  # You can tune this
        
        return base_loss + λ * action_entropy.mean()
        #return torch.sum(((torch.ones(len(isLastState))-isLastState) * nextVal + reward_list - curVal - A)**2) + torch.sum(c4_list**2)