from collections import namedtuple
import random
import torch
from copy import deepcopy as dc
import numpy as np
import matplotlib.pyplot as plt
import math
from channel_model import *
device = torch.device("cpu")

"""
Transition object summarizing changes to the environment at each time step
:param state:       State object representing observable features of the current state
:param action:      Array of actions of all agents
:param next_state:  State object representing observable features of the resultant state
:param reward:      Array of rewards obtained by all agents
"""
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

"""
State object summarizing observable features of the environment at each time step
:param rate:  Rate available in that RAT
:param station_id: Station to which the user is connected
:param rat_type: If its a Wifi or LTE station
"""
ProtoState = namedtuple('State', ('rate', 'station_id', 'rat_type'))


class State(ProtoState):

    def to_sep_numpy(self, idx):
        """
        Extract and normalize the state for a specific agent.
        Parameters:
        - idx (int): Index of the agent.
        Returns:
        - torch.Tensor: Flattened and normalized state vector for the agent.
        """
        agent_rate = self.rate[idx]
        normalized_rate = torch.tensor([(rate - 0)/(480 - 0) if rate > 0 else rate for rate in agent_rate])
        return torch.cat([
            normalized_rate,
            self.station_id[idx],
            self.rat_type[idx]
        ], dim=-1).float().to(device)
        
import copy

def clone(self):
    """
    Creates a deep copy of the object.
    """
    return copy.deepcopy(self)


class Multi_RAT_Network:
  """
    Class representing the multi RAT environment.
    Parameters:
    - area_width (float): Width of the simulation area.
    - n_users (int): Number of users.
    - n_aps (int): Number of Wi-Fi APs.
    - n_steps (int): Number of movement steps.
    - plot (bool): Whether to plot the environment initially.
    - train (bool): If True, training logic will apply (e.g. RAT switching penalty).
  """
  def __init__(self, area_width, n_users, n_aps, n_steps,  plot, train ):
    """
    Initializes the Multi-RAT environment with LTE and Wi-Fi stations,
    users, their assignments, rates, and paths.
    """
    self.area_width = area_width
    self.n_ltesn = 4
    self.n_aps = n_aps
    self.n_stations =  self.n_ltesn + self.n_aps
    self.n_users = n_users  
    self.n_steps = n_steps
    self.iteration = 0
    self.train = train

    self.reset() # Initialize the enviroment

    if plot: 
       self.plot_environment() 
       

  def reset(self):
    """
    Resets the environment, initializes users, stations, positions, paths,
    assignments, and computes initial rates and states.
    """
    # Reset rewards
    self.last_reward = torch.zeros(self.n_users).to(device)
    self.total_reward = torch.zeros(self.n_users).to(device)
    #Reset time parameters
    self.iteration = 0

    self.ltesn_positions = []  # List to save LTE Serving Nodes positions
    self.aps_positions = []  # List to save LTE Serving Nodes positions
    self.users_positions = [] # List of user positions
    self.users_destinations = [] # List of points where the user will move towards
    self.user_assignments = []  # List to save user assignments with SN information
    self.rat_change = [0] * self.n_users # List to save when a user changes to a different RAT
    
    margin = self.area_width/2 * 0.5
    half_width = self.area_width / 2

    corners = [
        (-half_width, -half_width),  
        ( half_width, -half_width),  
        (-half_width,  half_width),  
        ( half_width,  half_width),  
    ]

    for x_corner, y_corner in corners:
        if x_corner < 0:
            x_min = x_corner
            x_max = x_corner + margin
        else:
            x_min = x_corner - margin
            x_max = x_corner

        if y_corner < 0:
            y_min = y_corner
            y_max = y_corner + margin
        else:
            y_min = y_corner - margin
            y_max = y_corner

        self.ltesn_positions.append([
            random.uniform(x_min, x_max),
            random.uniform(y_min, y_max)
        ])



    # Define AREA for APs parameters
    grid_size = self.area_width/math.sqrt(self.n_aps)  # Size of each square
    num_rows = int(math.sqrt(self.n_aps))  # NxN grid
    num_cols = int(math.sqrt(self.n_aps))
    start_x = -self.area_width/2  # Bottom-left X boundary
    start_y = -self.area_width/2  # Bottom-left Y boundary

    # Initialize APs, ensuring one per square
    for row in range(num_rows):
        for col in range(num_cols):
            # Define the boundaries of the current grid square
            x_min = start_x + col * grid_size
            x_max = x_min + grid_size
            y_min = start_y + row * grid_size
            y_max = y_min + grid_size
            # Generate a random position within the square
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            self.aps_positions.append([x, y])
    
    
    #Function to know if a user is in range of at least one AP
    def in_AP_range(user_pos):
        for ap_pos in self.aps_positions:
            distance = math.sqrt((user_pos[0] - ap_pos[0]) ** 2 + (user_pos[1] - ap_pos[1]) ** 2)
            if distance <= 12:
                return True
        return False
    
    # Initialize users randomly within the user area and in range of at least 1 AP
    for _ in range(self.n_users):
        while True:
            x = random.uniform(-self.area_width/2, self.area_width/2)
            y = random.uniform(-self.area_width/2, self.area_width/2)
            user_pos = [x, y]
            if in_AP_range(user_pos):
                self.users_positions.append(user_pos)
                break
    
    # Assign users to the closest Serving Node LTE or WiFI
    for user_pos in self.users_positions:
        closest_distance = float('inf')
        assignment = None
        # Check distances to LTE SNs
        for i, ltesn_pos in enumerate(self.ltesn_positions):
            distance = math.sqrt((user_pos[0] - ltesn_pos[0]) ** 2 + (user_pos[1] - ltesn_pos[1]) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                assignment = [0, i]  # [RAT=0 (LTE SN), Node ID=i]

        for i, ap_pos in enumerate(self.aps_positions):
            distance = math.sqrt((user_pos[0] - ap_pos[0]) ** 2 + (user_pos[1] - ap_pos[1]) ** 2)
            if distance < closest_distance and distance < 12:
                closest_distance = distance
                assignment = [1, i]  # [RAT=1 (AP), Node ID=i]

        self.user_assignments.append(assignment)

    # Initialize rate, station_id, and rat_type needed for the state    
    num_stations = len(self.ltesn_positions) + len(self.aps_positions)
    self.rate = torch.zeros((self.n_users, num_stations)).to(device)
    self.station_id = torch.zeros((self.n_users, num_stations)).to(device)

    for user_idx, assignment in enumerate(self.user_assignments):
        rat, station_idx = assignment
        # Determine the global station index (LTE stations first, then Wi-Fi APs)
        global_station_idx = station_idx if rat == 0 else station_idx + len(self.ltesn_positions)
        # Update the station_id matrix
        self.station_id[user_idx, global_station_idx] = 1

        # Get user and station positions to calculate the rate
        user_position = self.users_positions[user_idx]
        rates_lte = []
        rates_wifi = []
        for lte_position in self.ltesn_positions:
            rate_value = get_rate(user_position, lte_position,rat_type=0)
            rates_lte.append(rate_value)
        for wifi_position in self.aps_positions:
            rate_value = get_rate(user_position, wifi_position,rat_type=1)
            rates_wifi.append(rate_value)
        # Update the rate matrix
        self.rate[user_idx, :len(rates_lte)] = torch.tensor(rates_lte).to(device)  # LTE rates
        self.rate[user_idx, len(rates_lte):] = torch.tensor(rates_wifi).to(device)  # Wi-Fi rates

    
    self.rat_type = torch.cat([
        torch.zeros(len(self.ltesn_positions)),  # LTE stations: 0
        torch.ones(len(self.aps_positions))     # Wi-Fi APs: 1
    ]).to(device)
    self.rat_type = self.rat_type.unsqueeze(0).repeat(self.n_users, 1)

    # Generate the straight-line path with equidistant points for each user
    self.users_destinations = []
    self.users_path = []    
    min_distance = self.area_width / 3  # Minimum required path length
    for user_pos in self.users_positions:
        while True:
            x = random.uniform(-self.area_width / 2, self.area_width / 2)
            y = random.uniform(-self.area_width / 2, self.area_width / 2)
            user_dest = [x, y]
            # Check minimum distance
            if np.linalg.norm(np.array(user_pos) - np.array(user_dest)) < min_distance:
                continue  # Try again if too short

            x1, y1 = user_pos
            x2, y2 = user_dest
            valid_points = []
            for step in range(self.n_steps + 1):
                alpha = step / self.n_steps
                x = x1 + alpha * (x2 - x1)
                y = y1 + alpha * (y2 - y1)
                valid_points.append([x, y])

            self.users_destinations.append(user_dest)
            self.users_path.append(valid_points)
            break  # Valid destination found


  def update_users_positions(self):
    """
    Moves all users one step forward along their predefined straight-line paths.
    Updates `self.users_positions` in-place.
    """
    next_idx = self.iteration + 1
    self.users_positions = [user_path[next_idx] for user_path in self.users_path]

  def update_user_rate(self):
    """
    Updates the achievable data rates between users and all stations
    based on the updated positions of the users.
    """
    for user_idx, user_pos in enumerate(self.users_positions):
        rates_lte = []
        for station_pos in self.ltesn_positions:
            rate = get_rate(user_pos, station_pos, rat_type=0)  # LTE
            rates_lte.append(rate)
        rates_wifi = []
        for station_pos in self.aps_positions:
            rate = get_rate(user_pos, station_pos, rat_type=1)  # Wi-Fi
            rates_wifi.append(rate)
        # Update rate matrix (LTE first, then Wi-Fi)
        self.rate[user_idx, :len(rates_lte)] = torch.tensor(rates_lte).to(device)
        self.rate[user_idx, len(rates_lte):] = torch.tensor(rates_wifi).to(device)


  def update_station_id(self):
    """
    Updates the station assignment matrix indicating the currently connected
    station for each user using one-hot encoding.
    """
    # Reinitialize the station_id tensor to zeros before updating
    num_stations = len(self.ltesn_positions) + len(self.aps_positions)
    self.station_id = torch.zeros((self.n_users, num_stations)).to(device)
    # Iterate over the updated user assignments
    for user_idx, assignment in enumerate(self.user_assignments):
        rat, node_id = assignment
        # Determine the global station index (LTE stations first, then Wi-Fi APs)
        global_station_idx = node_id if rat == 0 else node_id + self.n_ltesn
        self.station_id[int(user_idx),int(global_station_idx)] = 1


  def update_state(self):
    """
    Updates the environment by moving users, updating rates,
    and refreshing user-to-station assignments.
    """
    self.update_users_positions() # Move the user
    self.update_user_rate() # Update available rate in every station
    self.update_station_id() #Updates the station_id matrix based on the current user assignments.


  def plot_environment(self):
    """
    Plots a visual representation of the environment including:
    - LTE base stations
    - Wi-Fi APs
    - User positions and movement paths (for 5 random users)
    """
    plt.figure(figsize=(5, 5))

    # Plot LTE SNs as red triangles
    for position in self.ltesn_positions:
        plt.scatter(position[0], position[1], c='red', marker='^', label='LTE BS', s=25)

    # Plot APs as blue triangles
    for position in self.aps_positions:
        plt.scatter(position[0], position[1], c='blue', marker='^', label='AP', s=25)

    # Plot users and color them based on RAT assignment
    for user_idx, position in enumerate(self.users_positions):
        rat, node_id = self.user_assignments[user_idx]
        color = "tab:green"  # Light red for RAT=0, light blue for RAT=1
        plt.scatter(position[0], position[1], c=color, label="User", s=20)

    # Select 5 random users for path plotting
    num_users = len(self.users_positions)
    sample_users = random.sample(range(num_users), min(5, num_users))

    # Plot user paths
    for i,user_idx in enumerate(sample_users):
            path = self.users_path[user_idx]  # List of (x, y) positions
            if len(path) > 1:
                path_x, path_y = zip(*path)  # Extract x and y coordinates
                plt.plot(path_x, path_y, linestyle='--', marker='x', markersize=3, 
                         color='orange', label='User Path')#path_colors[i % len(path_colors)])

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Add grid, labels, and plot limits
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Physical Scenario',fontsize = 14)
    plt.xlabel('X Position (m)',fontsize = 14)
    plt.ylabel('Y Position (m)',fontsize = 14)
    axis_limit = self.area_width / 2
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

     
  def get_ue_throughput(self, user_id):
    """
    Calculates the throughput for a specific user based on their current assignment.
    """
    rat, station_idx = self.user_assignments[user_id]
    shared_users = sum([1 for assignment in self.user_assignments if assignment == (rat, station_idx)])
    
    base_rate = self.rate[user_id, (station_idx + rat* self.n_ltesn)].item()  # Get the base rate for the user at the assigned station
    throughput = 0
    if base_rate>0:
        if rat == 0:  # LTE user
            if shared_users == 0:
                throughput = base_rate
            else:
                throughput = base_rate / shared_users
        else:  # Wi-Fi user
            inverse_rate_sum = 0
            for i in range(self.n_users):
                other_rat, other_station_idx = self.user_assignments[i]
                if other_rat == rat and other_station_idx == station_idx:
                    other_rate = self.rate[i, (other_station_idx + self.n_ltesn)].item()
                    #print("other_rate = ", other_rate)
                    if other_rate > 0:
                        inverse_rate_sum += 1 / other_rate
            throughput = 1/inverse_rate_sum
    else:
        throughput = 0

    return throughput              


  def r(self):  
    """
    Computes the reward for all users:
    - Encourages high throughput
    - Penalizes unnecessary RAT switches
    - Penalizes disconnection
    Returns:
    - torch.Tensor: Reward vector for all users.
    """
    users_thr = [self.get_ue_throughput(i) for i in range(self.n_users)]
    self.max_thr = 480
    min_thr = 0
    norm_thr = [(throughput - min_thr) / (self.max_thr - min_thr) for throughput in users_thr]
    users_reward = []
    for i in range(self.n_users): # Compute reward for each user
        reward = norm_thr[i]
        if reward > 0 :
            
            if self.train and self.rat_change[i]:
                last_thr = self.last_reward[i].item() * (self.max_thr)
                current_thr = users_thr[i]
                reward = norm_thr[i]
                if last_thr!=0:
                    if current_thr/last_thr < 1.1:
                        reward = ( (current_thr - 0.1*last_thr) - min_thr) / (self.max_thr - min_thr)
            else:
                reward = norm_thr[i]

        else: # User disconnected
            reward = -0.1
        users_reward.append(reward)
    return torch.tensor(users_reward, dtype=torch.float)
  

  def step(self,actions):
    
    """
    Executes a simulation step based on agent actions.
    Parameters:
    - actions (torch.Tensor): Action tensor of shape [n_users, 2].
    Returns:
    - Transition: Tuple of (state, action, next_state, reward).
    """
    
    last_state, _, _ =self.get_state()
    # Transform NN output
    rats_chosen = [] # List of [RAT,node_ID] chosen by every user.
    for rat_choice in actions:        
        rat_id = 0 if rat_choice < self.n_ltesn else 1
        node_id = rat_choice - rat_id * self.n_ltesn
        rats_chosen.append([rat_id,node_id])
    # Update self.RAT_change
    current_rats = [rat[0] for rat in self.user_assignments]
    new_rats = [action[0] for action in rats_chosen]
    self.rat_change = [0] * self.n_users
    for i in range(self.n_users):
        if current_rats[i] != new_rats[i]:
            self.rat_change[i] = 1

    # Update self.user_assignments
    self.user_assignments = rats_chosen
    # Compute Reward for each agent in this enviroment

    reward = self.r()
    self.last_reward = reward

    # Update state
    self.update_state()
    cur_state, _, _=self.get_state()
        
    return Transition(last_state, actions, cur_state, self.last_reward)
       

  def get_state(self):
        """
        Returns the observable features of the current state
        :return: State object summarizing observable features of the current state
        """
        return State(
            self.rate.clone(), 
            self.station_id.clone(), 
            self.rat_type.clone(), 
        ), self.last_reward.clone(), self.total_reward.clone()
    
  def __str__(self):
        state, last_reward, total_reward=self.get_state()
        str="Simulation -- Last State: {}, \
        Last Reward: {}, Total Reward: {}".format(state, last_reward, total_reward)
        return str

  def __repr__(self):
        return self.__str__()    
  
  #-----visualization-------------
  def get_rat_chosen(self,actions,last_rats):
    for action in actions:        
        last_rats[action]+=1    
    return last_rats
  #--------------------------
  def clone(self):
    return copy.deepcopy(self)


class ExperienceReplay:
    """
    Class for storing objects in the experience replay buffer
    :param buffer:          List containing all objects in the replay buffer
    :param max_buffer_size: Max size of the buffer
    :param buffer_size:     Current size of the buffer
    """
    def __init__(self, buffer_size):
        self.cur_s_buffer=torch.empty(0).to(device)
        self.next_s_buffer=torch.empty(0).to(device)
        self.term_flag_buffer=torch.empty(0).to(device)
        self.rewards_buffer=torch.empty(0).to(device)
        self.action_buffer=torch.empty(0).to(device)
        self.max_buffer_size=buffer_size
        self.buffer_size=0

    def add(self, cur_s, next_s, term_flag, rewards, action):
        if self.buffer_size >= self.max_buffer_size:
            self.cur_s_buffer=self.cur_s_buffer[1:,:]
            self.next_s_buffer=self.next_s_buffer[1:,:]
            self.term_flag_buffer=self.term_flag_buffer[1:]
            self.rewards_buffer=self.rewards_buffer[1:,:]
            self.action_buffer=self.action_buffer[1:,:]
            
            self.buffer_size -= 1
        
        self.cur_s_buffer=torch.cat([self.cur_s_buffer, torch.unsqueeze(cur_s,0)], dim = 0)
        self.next_s_buffer=torch.cat([self.next_s_buffer, torch.unsqueeze(next_s,0)], dim = 0)
        self.term_flag_buffer=torch.cat([self.term_flag_buffer, torch.unsqueeze(term_flag,0)], dim = 0)
        self.rewards_buffer=torch.cat([self.rewards_buffer, torch.unsqueeze(rewards,0)], dim = 0)
        self.action_buffer=torch.cat([self.action_buffer, torch.unsqueeze(action,0)], dim = 0)
        self.buffer_size += 1

    def sample(self, size):
        sample_idx = torch.tensor(np.random.choice(self.buffer_size, min(size, self.buffer_size), replace=False), dtype = torch.long)
        
        return self.cur_s_buffer[sample_idx], self.next_s_buffer[sample_idx], self.term_flag_buffer[sample_idx], self.rewards_buffer[sample_idx], self.action_buffer[sample_idx]
    
    def query(self, idx):
        sample_idx = idx
        return self.cur_s_buffer[sample_idx], self.next_s_buffer[sample_idx], self.term_flag_buffer[sample_idx], self.rewards_buffer[sample_idx], self.action_buffer[sample_idx]

    def __len__(self):
        return self.buffer_size

    def reset(self):
        self.cur_s_buffer=torch.empty(0).to(device)
        self.next_s_buffer=torch.empty(0).to(device)
        self.term_flag_buffer=torch.empty(0).to(device)
        self.rewards_buffer=torch.empty(0).to(device)
        self.action_buffer=torch.empty(0).to(device)
        self.buffer_size=0
        

