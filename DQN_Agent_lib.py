import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random
from copy import deepcopy as dc

device=torch.device("cpu")
import torch.nn.functional as F


class PermInvariantQNN(torch.nn.Module): 
    """
    Permutation-Invariant Neural Network used as the Q-function approximator.
    Processes the input state of each user independently with shared weights.
    
    :param n_users:     Number of users (agents)
    :param n_stations:  Number of stations (LTE + Wi-Fi)
    :param out_dim:     Output dimension, equals to number of possible actions (stations)
    :param lat_dims:    Latent dimension of the hidden layers
    :param layers:      Number of hidden layers
    """
    n_users: int
    n_stations: int
    out_dim: int
    lat_dims: int
    layers: int

    def initialize_weights(self):
        """Initializes the network weights using normal and uniform distributions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.5)
                nn.init.uniform_(m.bias, a=0.5, b=0.5)

    def __init__(self,n_users, n_stations, out_dim, lat_dims, layers):
        super(PermInvariantQNN, self).__init__()
        # Store input and output dimensions
        self.n_users = n_users
        self.n_stations = n_stations
        self.out_dim = out_dim 
        
        nets = []
        input_dim = n_stations * 3  # number of features/columns of the state space (rate , station_id , rate_type)
        
        # First layer
        nets.append(nn.Linear(input_dim, lat_dims))  # lat_dims = number of neurons per layer
        nets.append(nn.ReLU())
        nets.append(nn.LayerNorm(lat_dims))

        # Hidden layers
        for i in range(layers):
            next_lat_dims = max(int(lat_dims // 2), out_dim) 
            nets.append(nn.Linear(lat_dims, next_lat_dims))
            nets.append(nn.LayerNorm(next_lat_dims))
            nets.append(nn.ReLU())
            lat_dims = next_lat_dims
        
        # Output layer
        nets.append(nn.Linear(lat_dims, self.out_dim))
        self.decoder_net = nn.Sequential(*nets)
        
        self.initialize_weights()

    def forward(self, input):
        """
        Forward pass through the network.
        :param input: Input state vector of shape (n_users, features)
        :return: Q-values of shape (n_users, out_dim)
        """
        x = self.decoder_net(input)  # Output shape: (n_users, out_dim)
        return x


class DQN_NN():
    """
    Class for DQN Agent holding the action and target (value) networks.
    
    :param n_users:     Number of users (agents)
    :param n_stations:  Number of possible station choices
    :param lat_dims:    Latent dimension in the neural network
    :param c_cons:      Coefficient for optional constraints (unused in code)
    :param c2_cons:     Flag for applying constraint 2 (unused in code)
    :param c3_pos:      Flag for constraint 3 positivity (unused in code)
    :param layers:      Number of layers in the network
    :param weighted_adam: Whether to use AdamW instead of Adam
    """
    def __init__(self, n_users, n_stations, lat_dims= 512, layers=1, weighted_adam=True):
        # Simulation Parameters
        self.n_users = n_users
        self.n_stations = n_stations
        self.output_dim = n_stations    

        # Initialize action and value networks
        self.action_net = PermInvariantQNN(
            n_users = self.n_users, n_stations = self.n_stations, out_dim=self.output_dim, lat_dims=lat_dims, layers=layers)
        self.value_net = PermInvariantQNN(
            n_users = self.n_users, n_stations = self.n_stations, out_dim=self.output_dim, lat_dims=lat_dims, layers=layers)

        # Choose optimizer
        self.lr = 2.5e-4
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
        
        # Mean Squared Error loss for Q-value targets
        self.criterion = nn.MSELoss()

    def predict_action(self, states):
        """
        Predict Q-values from the value (target) network.
        :param states: Input state tensor
        :return: Q-values predicted by the value network
        """
        if len(states.shape) > 2:
            B, A, D = states.shape
            flat_states = states.view(B * A, D)  # Flatten to (B*A, D)
            action_list = self.action_net.forward(input=flat_states)  # Predict Q-values
            action_list = action_list.view(B, A, -1)  # Reshape back to (B, A, out_dim)
        else:
            action_list = self.action_net.forward(input=states) 
        
        return action_list
        
    def predict_value(self, states, ):
        """
        Predict Q-values from the value (target) network.

        :param states: Input state tensor
        :return: Q-values predicted by the value network
        """
        values = self.value_net.forward(input=states)     
        return values



