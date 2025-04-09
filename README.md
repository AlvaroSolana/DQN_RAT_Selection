# Nash_DQN_RAT

## Overview

This repository contains both python files and Jupyter Notebooks that are used in a simulation where the goal is to improve overall user throughput using Reinforcement Learning

## Files

- **RAT_training.ipynb**:In this notebook we are able to set the parameters of the scenario where the model will be trained, as well as peform this training.
- **test.ipynb**: In this notebook we take the action network obtained from the training to test it in new scenarios and measure its performance, i.e. throguhput achieved by the users.
- **Nash_RL.py**: Among other things it contains the main training algorithm, uses the rest of the files to return the action network among other things.
- **Nash_Agent_lib.py**: Contains the NN required to perform the training, and functions to perform feed forward, back_propagation, loss computation...
- **RAT_env.py**: Contains the enviroment definition and functions required for the Reinforcement Learning
- **Action_Net & Value net**: Output of the training, the action net will be used for testing the model in the test.ipynb.

## Datasets Used

The following CSV file were used in this project:
- `cqi_distance.csv` – Contains pairs of cqi values from users at a certain distance from the LTE Base Station.
- `rssi_distance` – Contains pairs of rssi values from users at a certain distance from the Access points.
- `rewards.csv` – Created in the training for visualizing the reward of each user during the training.
- `Data_analysis` – Contains csv files and jupyter notebooks that were used to analyse and create the datasets cqi_distance.csv and rssi_distance.csv .
  
## Requirements

To run the notebooks, ensure you have the following dependencies installed:

```bash
pip install numpy pandas matplotlib random csv
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AlvaroSolana/Nash_DQN_RAT
   ```
2. Navigate to the project directory:
   ```bash
   cd Nash_DQN_RAT
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter lab
   ```
   If you need to install jupyter lab: https://jupyter.org/install
   
5. Open and run the notebooks as needed.

## License

This project is open-source and available under the MIT License.

## Contact

For any questions or contributions, feel free to open an issue or reach out!

