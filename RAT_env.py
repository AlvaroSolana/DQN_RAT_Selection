from collections import namedtuple
import random
import torch
from copy import deepcopy as dc
import numpy as np
import matplotlib.pyplot as plt
import math
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
        normalized_rate = torch.tensor([(rate - 0)/(600 - 0) if rate > 0 else rate for rate in agent_rate])
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
  :param cqi_dataset : Dataset of CQI and distance pairs
  :param rssi_dataset : Dataset of RSSI and distance pairs
  :param plot: To plot the enviroment intial distribution of its elements
  """

  def __init__(self, user_area_width, ltesn_area_width,n_users, n_aps, cqi_dataset, rssi_dataset, n_steps,  plot ):
    """
    Initialize the environment (multiple networks) with its parameters
    """
    self.user_area_width = user_area_width
    self.ltesn_area_width = ltesn_area_width
    self.n_ltesn = 4
    self.n_aps = n_aps
    self.n_stations =  self.n_ltesn + self.n_aps
    self.n_users = n_users  
    self.cqi_dataset = cqi_dataset  
    self.rssi_dataset = rssi_dataset
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
    self.users_cqi = [] # List of CQI values for every user for every LTE Serving Node. List of n_users len with 4 Lists for each user
    self.users_rssi = [] # List of RSSI values for every user for every AP.
    self.user_assignments = []  # List to save user assignments with SN information
    self.rat_change = [0] * self.n_users # List to save when a user changes to a different RAT
    
    # Initialize LTE SNs within their respective areas
    half_ltesn_area = self.ltesn_area_width / 2
    for i in range(2):
        for j in range(2):
            x_min = -half_ltesn_area + i * half_ltesn_area
            x_max = x_min + half_ltesn_area
            y_min = -half_ltesn_area + j * half_ltesn_area
            y_max = y_min + half_ltesn_area
            self.ltesn_positions.append([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max)
            ])

    # Define area for APs and users 
    restricted_area_width = self.ltesn_area_width + 2 * 50
    half_restricted_area = restricted_area_width / 2
    full_area_width = 2* self.user_area_width + restricted_area_width
    half_full_area = full_area_width / 2

    # Define grid parameters
    grid_size = 350/math.sqrt(self.n_aps)  # Size of each square
    num_rows = int(math.sqrt(self.n_aps))  # NxN grid
    num_cols = int(math.sqrt(self.n_aps))
    start_x = -175  # Bottom-left X boundary
    start_y = -175  # Bottom-left Y boundary

    # Initialize APs, ensuring one per square
    for row in range(num_rows):
        for col in range(num_cols):
            num_tries = 0
            while True:
                # Define the boundaries of the current grid square
                x_min = start_x + col * grid_size
                x_max = x_min + grid_size
                y_min = start_y + row * grid_size
                y_max = y_min + grid_size

                # Generate a random position within the square
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                
                num_tries+=1
                if num_tries>5:#in case the square is inside the restricted area
                    x = random.uniform(start_x, -start_x)
                    y = random.uniform(start_y, -start_y)

                # Ensure AP is outside the restricted area
                if not (-half_restricted_area <= x <= half_restricted_area and -half_restricted_area <= y <= half_restricted_area):
                    self.aps_positions.append([x, y])
                    break  # Exit loop once a valid AP position is found
    
    if len(self.aps_positions)!=self.n_aps: # Just to make sure, shuold never happened
        print("wrong initialization of the APs")
        self.reset()
    
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
            x = random.uniform(-half_full_area, half_full_area)
            y = random.uniform(-half_full_area, half_full_area)
            user_pos = [x, y]
            if not (-half_restricted_area <= x <= half_restricted_area and -half_restricted_area <= y <= half_restricted_area) and in_AP_range(user_pos):
                self.users_positions.append(user_pos)
                break

    self.update_cqi() # Assign cqi values to the users in their positions
    self.update_rssi() # Assign rssi values to the users in their positions
    
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

        # With a 50% probability, check distances to APs (they will always be closer than LTEs) 
        #if random.random() < 0.5:  # 50% probability 
        # Now we always start from APs
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

        rates_lte = [self.get_lte_rate(cqi) for cqi in self.users_cqi[user_idx]]
        rates_wifi = [self.get_wifi_rate(rssi) for rssi in self.users_rssi[user_idx]]

        # Update the rate matrix
        self.rate[user_idx, :len(rates_lte)] = torch.tensor(rates_lte).to(device)  # LTE rates
        self.rate[user_idx, len(rates_lte):] = torch.tensor(rates_wifi).to(device)  # Wi-Fi rates
    
    self.rat_type = torch.cat([
        torch.zeros(len(self.ltesn_positions)),  # LTE stations: 0
        torch.ones(len(self.aps_positions))     # Wi-Fi APs: 1
    ]).to(device)
    self.rat_type = self.rat_type.unsqueeze(0).repeat(self.n_users, 1)

    # Establish the destination position to which the user moves towards
    half_restricted_area = (self.ltesn_area_width + 2 * 50) / 2
    self.users_destinations = []
    self.users_path = []
    def crosses_restricted_area(x1, y1, x2, y2): # To avoid users to cross through the LTE stations where we have no data
        """ Check if a straight-line path crosses the restricted area """
        if (x1 < -half_restricted_area and x2 > half_restricted_area) or (x1 > half_restricted_area and x2 < -half_restricted_area):
            if (y1 < -half_restricted_area and y2 > half_restricted_area) or (y1 > half_restricted_area and y2 < -half_restricted_area):
                return True  # Crosses in Y direction
        else:
            return False 
    
    for user_pos in self.users_positions:
        while True:
            x = random.uniform(-half_full_area, half_full_area)
            y = random.uniform(-half_full_area, half_full_area)
            user_dest = [x, y]

            # Ensure the destination is outside the restricted area and does not create a diagonal crossing path
            if not (-half_restricted_area <= x <= half_restricted_area and -half_restricted_area <= y <= half_restricted_area) and not crosses_restricted_area(user_pos[0], user_pos[1], x, y):
                # Generate the straight-line path for each user with equidistant points
                x1, y1 = user_pos
                x2, y2 = user_dest
                valid_points = []
                valid_path = True
                for step in range(self.n_steps + 1):
                    alpha = step / self.n_steps  # Ensures equidistant points
                    x = x1 + alpha * (x2 - x1)
                    y = y1 + alpha * (y2 - y1)
                    if (-half_restricted_area <= x <= half_restricted_area and -half_restricted_area <= y <= half_restricted_area):
                        valid_path = False
                        break # we need new destination
                    valid_points.append([x, y])
                
                if valid_path:
                    self.users_destinations.append(user_dest)
                    self.users_path.append(valid_points)
                    break  # Valid destination found


  def update_users_positions(self):
    # Move the user from intial position to destination
    next_idx = self.iteration + 1
    self.users_positions = [user_path[next_idx] for user_path in self.users_path]


  def update_cqi(self):
    # Obtain the CQI for each user for every LTE station
    for user_pos in self.users_positions:
        user_cqi = []
        for ltesn_pos in self.ltesn_positions:
            distance = math.sqrt((user_pos[0] - ltesn_pos[0]) ** 2 + (user_pos[1] - ltesn_pos[1]) ** 2)
            closest_row = self.cqi_dataset.iloc[(self.cqi_dataset['Distance'] - distance).abs().idxmin()]
            user_cqi.append(closest_row['CQI'])
        self.users_cqi.append(user_cqi)

  def update_rssi(self):
        # Obtain RSSI value for each user to every AP (if not connected RSSI = -100)
    for user_pos in self.users_positions:
        user_rssi = []
        for ap_pos in self.aps_positions:
            distance = math.sqrt((user_pos[0] - ap_pos[0]) ** 2 + (user_pos[1] - ap_pos[1]) ** 2)
            if distance > 12:
                user_rssi.append(-100)
            else:
                closest_row = self.rssi_dataset.iloc[(self.rssi_dataset['Distance'] - distance).abs().idxmin()]
                user_rssi.append(closest_row['RSSI'])
        self.users_rssi.append(user_rssi)

  def update_user_rate(self):
    #Update the rate for each user based on their new position, CQI, and RSSI values.    
    for user_idx, user_pos in enumerate(self.users_positions):
        # Calculate the rates for LTE and Wi-Fi
        rates_lte = [self.get_lte_rate(cqi) for cqi in self.users_cqi[user_idx]]
        rates_wifi = [self.get_wifi_rate(rssi) for rssi in self.users_rssi[user_idx]]

        # Update the rate for the corresponding user and station
        self.rate[user_idx, :len(rates_lte)] = torch.tensor(rates_lte).to(device)  # LTE rates
        self.rate[user_idx, len(rates_lte):] = torch.tensor(rates_wifi).to(device)  # Wi-Fi rates

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
    self.update_cqi() # Obtain CQI values to each station in the new position
    self.update_rssi() # Obtain RSSI values to each station in the new position
    self.update_user_rate() # Update available rate in every station with the new RSSI / CQI values
    self.update_station_id() #Updates the station_id matrix based on the current user assignments.



  def plot_environment(self):
    """
    Plot the positions of LTE SNs, APs, and users in the environment.
    Additionally, plot the movement paths of 5 random users.
    """
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
    axis_limit = self.user_area_width + self.ltesn_area_width / 2 + 50
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)

    plt.show()


  def get_wifi_rate(self,rssi_value):
    """
    Returns the user rate according to the RSSI value
    """ 
    if rssi_value >= -55:
        return 600
    elif rssi_value >= -60:
        return 300  
    elif rssi_value >= -65:
        return 150  
    elif rssi_value >= -70:
        return 75 
    elif rssi_value >= -75:
        return 30
    else:
        return -1


  def get_lte_rate(self,cqi_value):
    """
    Returns the user rate acccording to the CQI value
    """
    cqi_to_rate = {
        15: 300, 14: 240, 13: 200, 12: 150, 11: 120,
        10: 100, 9: 80, 8: 60, 7: 45, 6: 30,
        5: 20, 4: 15, 3: 10, 2: 5, 1: 1
    }
    return cqi_to_rate.get(cqi_value, 0)

     
  def get_ue_throughput(self,user_id):
      """
      Calculate the user's throughput
      :param user_id : The user for which we want to compute the throughput
      """
      shared_users = sum([1 for assignment in self.user_assignments if assignment == self.user_assignments[user_id]])
      rat = self.user_assignments[user_id][0]
      station_id = self.user_assignments[user_id][1]  # Convert tensor to integer
      throughput = 0
      distance_to_ap_list = []
      if rat == 0: # LTE user
        cqi = self.users_cqi[int(user_id)][int(station_id)]
        throughput = self.get_lte_rate(cqi) / shared_users
      else: # if the user is connected to an AP
        distance_to_ap_list.append(math.sqrt((self.users_positions[user_id][0]-self.aps_positions[station_id][0])**2 +(self.users_positions[user_id][1]-self.aps_positions[station_id][1])**2))
        rssi = self.users_rssi[user_id][station_id]
        if rssi == -100: # If the user is not connected to any AP, we set the throughput to 0
            throughput = 0
        else:
            inverse_rate_sum = 0
            for i in range(self.n_users):
                if self.user_assignments[i] == self.user_assignments[user_id]:
                    if rssi != -100:
                        rate = self.get_wifi_rate(rssi)
                    if rate > 0 :    
                        inverse_rate_sum += 1/rate
            else:
                throughput = 1/inverse_rate_sum

      return throughput
              


  def r(self):  
    '''
    reward function : rewards UE throughput, total system throughput, fairness between users and penalizes RAT changing
    '''
    users_thr = [self.get_ue_throughput(i) for i in range(self.n_users)]
    max_thr = 600
    min_thr = 0
    norm_thr = [(throughput - min_thr) / (max_thr - min_thr) for throughput in users_thr]
    
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
            reward = -1
        
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
        

