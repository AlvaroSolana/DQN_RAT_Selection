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

device = torch.device("cpu")

class State(ProtoState):

    def to_sep_numpy(self, idx):
        """
        Extract and normalize the state for a specific agent.
        :param idx: Index of the agent whose state is being extracted.
        """
        agent_rate = self.rate[idx]
        normalized_rate = torch.tensor([(rate - 0)/(480 - 0) if rate > 0 else rate for rate in agent_rate])
        return torch.cat([
            normalized_rate,
            self.station_id[idx],
            self.rat_type[idx]
        ], dim=-1).float().to(device)
        

class Multi_RAT_Network:
  """
  Class representing the multi RAT environment

  :param user_area_width: Width of the area of users
  :param ltesn_area_width : Width of the area of lte serving nodes
  :param n_aps : Number of APs in the scenario
  :param n_users: Number of user terminals in the scenario
  :param plot: To plot the enviroment intial distribution of its elements
  """

  def __init__(self, area_width, n_users, n_aps, n_steps,  plot ):
    """
    Initialize the environment (multiple networks) with its parameters
    """
    self.area_width = area_width
    self.n_ltesn = 4
    self.n_aps = n_aps
    self.n_stations =  self.n_ltesn + self.n_aps
    self.n_users = n_users  
    self.n_steps = n_steps
    self.iteration = 0

    self.reset() # Initialize the enviroment

    if plot: 
       self.plot_environment() 
       

  def reset(self):
    """
        Reset enviroment
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

        #if random.random() < 0.5:  # 50% probability to check distances to APs (they will probably be closer than LTEs) 
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
    # Move the user from intial position to destination
    next_idx = self.iteration + 1
    self.users_positions = [user_path[next_idx] for user_path in self.users_path]

  def update_user_rate(self):
    # Update the rate for each user based on their new position and achievable rates from all stations
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

    self.update_users_positions() # Move the user
    self.update_user_rate() # Update available rate in every station
    self.update_station_id() #Updates the station_id matrix based on the current user assignments.


  def plot_environment(self):
    """
    Plot the positions of LTE SNs, APs, and users in the environment.
    Additionally, plot the movement paths of 5 random users.
    """
    print(f"Aps positions: {self.aps_positions}")
    print(f"LTE SNs positions: {self.ltesn_positions}")
    plt.figure(figsize=(5, 5))

    # Plot LTE SNs as red triangles
    for position in self.ltesn_positions:
        plt.scatter(position[0], position[1], c='red', marker='^', label='LTE SN', s=20)

    # Plot APs as blue triangles
    for position in self.aps_positions:
        plt.scatter(position[0], position[1], c='blue', marker='^', label='AP', s=20)

    # Plot users and color them based on RAT assignment
    for user_idx, position in enumerate(self.users_positions):
        rat, node_id = self.user_assignments[user_idx]
        color = (1, 0, 0, 0.5) if rat == 0 else (0, 0, 1, 0.5)  # Light red for RAT=0, light blue for RAT=1
        plt.scatter(position[0], position[1], c=color, label=f'User (RAT {rat})', s=10)

    # Select 5 random users for path plotting
    num_users = len(self.users_positions)
    sample_users = random.sample(range(num_users), min(10, num_users))

    # Define colors for paths
    path_colors = ['green', 'magenta', 'cyan', 'orange', 'purple']

    # Plot user paths
    for i,user_idx in enumerate(sample_users):
            path = self.users_path[user_idx]  # List of (x, y) positions
            if len(path) > 1:
                path_x, path_y = zip(*path)  # Extract x and y coordinates
                plt.plot(path_x, path_y, linestyle='--', marker='x', markersize=3, 
                         color=path_colors[i % len(path_colors)])

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Add grid, labels, and plot limits
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Environment Layout with User Assignments and Paths')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    axis_limit = self.area_width / 2
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)

    plt.show()

     
  def get_ue_throughput(self, user_id):
    """
    Calculate the user's throughput based on physical position and station type.
    """
    rat, station_idx = self.user_assignments[user_id]
    shared_users = sum([1 for assignment in self.user_assignments if assignment == (rat, station_idx)])
    
    base_rate = self.rate[user_id, (station_idx + rat* self.n_ltesn)].item()  # Get the base rate for the user at the assigned station
    #print("base_rate = ", base_rate,"rat = ",rat, "station_idx = ",station_idx)
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
    '''
    reward function : rewards UE throughput, total system throughput, fairness between users and penalizes RAT changing
    '''
    users_thr = [self.get_ue_throughput(i) for i in range(self.n_users)]
    #print("users_thr = ", users_thr)
    self.max_thr = 480
    min_thr = 0
    norm_thr = [(throughput - min_thr) / (self.max_thr - min_thr) for throughput in users_thr]
    #print("norm_thr = ", norm_thr)
    users_reward = []
    for i in range(self.n_users): # Compute reward for each user
        reward = norm_thr[i]
        if reward > 0 :
            '''
            if self.rat_change[i]:
                last_thr = self.last_reward[i].item() * (max_thr - min_thr) + min_thr
                current_thr = users_thr[i]
                if last_thr!=0:
                    if current_thr/last_thr < 1.05:
                        reward = ( (current_thr - 0.1*last_thr) - min_thr) / (max_thr - min_thr)
            '''
            reward = norm_thr[i]

        else: # User disconnected
            reward = -0.1
        #print("reward = ", reward)
        users_reward.append(reward)
    return torch.tensor(users_reward, dtype=torch.float)
  

  def step(self,actions):
    
    """
    Calculate and update all environment parameters given all agent's actions.
    Returns a Transition tuple holding observable state values of the previous
    state of the environment, actions executed by all agents, observable state
    values of the resultant state and the rewards obtained by all agents.')
    :param actions: Output of the NN, torch.Size([N_users,N_stations])
    """
    
    last_state, _, _ =self.get_state()
    # Transform NN output
    rats_chosen = [] # List of [RAT,node_ID] chosen by every user.
    for rat_choice in actions:        
        rat_id = 0 if rat_choice < self.n_ltesn else 1
        node_id = rat_choice - rat_id * self.n_ltesn
        rats_chosen.append([rat_id,node_id])
    # Udpadate self.RAT_change
    current_rats = [rat[0] for rat in self.user_assignments]
    new_rats = [action[0] for action in rats_chosen]
    self.rat_change = [0] * self.n_users
    for i in range(self.n_users):
        if current_rats[i] != new_rats[i]:
            self.rat_change[i] = 1

    '''     # DEBUGGING BLOCK
    # Check distance between AP chosen and user
    for i,user_pos in enumerate(self.users_positions):
        if new_rats[i] == 1: #If we have chosen an AP
            AP_id = rats_chosen[i][1] # Select ID of the AP of
            #print("AP_id = ",AP_id)
            ap_pos = self.aps_positions[AP_id]
            distance = math.sqrt((user_pos[0]-ap_pos[0])**2 + (user_pos[1]-ap_pos[1])**2)
            if distance > 12 and self.iteration==1:
                print("AP chosen OUT OF RANGE")
                print(f"User position [x,y] = [{user_pos[0]:.3f}, {user_pos[1]:.3f}] Chosen AP position [x,y] = [{self.aps_positions[AP_id][0]:.3f}, {self.aps_positions[AP_id][1]:.3f}]")
    '''
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
  
  #-----debugging-------------
  def get_rat_chosen(self,actions,last_rats):
    for action in actions:        
        last_rats[action]+=1    
    return last_rats
  #--------------------------


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
        

