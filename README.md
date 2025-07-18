# DQN-RAT-Selection:  Improving User Throughput in Multi-RAT Networks Using Reinforcement Learning

## 📡 Overview

With the exponential growth of connected mobile devices, the efficient use of wireless infrastructure has become increasingly important. Radio Access Technology (RAT) selection is a critical challenge in heterogeneous networks (HetNets), where users must choose efficiently between multiple available technologies, such as WiFi and LTE, in order to maximize their throughput.

The work was developed as part of a Bachelor's Thesis in Telecommunications Engineering, and it enables training, evaluation, and visualization of different metrics. For more information check the thesis report.

## 🎯 Project Goal

To evaluate the effectiveness of DQN-based RAT-Selection compared to other baseline methods in different key metrics such as throughput, fairness,  managing congestion, and user disconnection rate, demonstrating the effectiveness of this approach in managing the complex and dynamic RAT selection problem.

## 🗂️ File Descriptions

| File | Description |
|------|-------------|
| `RAT_env.py` | Core simulation environment modeling users, RATs, mobility, reward function, and interactions. |
| `channel_model.py` | Defines wireless channel behavior (path loss, Rayleigh fading, SNR, spectral efficiency). |
| `DQN_Agent_lib.py` | Deep Q-Network architecture with permutation-invariant layers and target/value nets. |
| `RL.py` | Main script for training the DQN agent, managing epsilon decay, buffer sampling, and network updates. |
| `evaluate.py` | Runs trained models in the environment and generates performance plots and statistics. |
| `heuristic.py` | Baseline algorithm: users probabilistically switch RATs based on throughput improvements. |
| `HartRL.py` | Stateless regret-based algorithm for decentralized user selection. |
| `RAT_training.ipynb` | Jupyter notebook to train the DQN agent and export model weights. |
| `test.ipynb` | Notebook to evaluate trained models and compare them with heuristics and Hart RL across metrics. |
| `Action_Net.pt` | Pre-trained PyTorch model weights for the final DQN agent used in testing. |
| `LICENSE` | MIT License for open-source use and distribution. |
| `requirements.txt` | Python dependencies needed to run the project. |
| `thesis_report.pdf` | Detailed explanation of this project. |


## 📦 Installation

Ensure Python 3.8+ and PyTorch are installed. Then clone the repository and install dependencies:

```bash
git clone https://github.com/AlvaroSolana/DQN_RAT_Selection.git
cd DQN_RAT_Selection
pip install -r requirements.txt
```

## 🚀 Usage

### 📓 Notebook Mode (Recommended)

```bash
jupyter lab
```
Use the provided notebooks to train and evaluate the DQN

## 📄 License

This project is open-source under the MIT License. Feel free to use, modify, and share.

## 📬 Contact

Developed by Álvaro Solana Lambán for his Telecommunications BSc Thesis. For questions, reach out via LinkedIn or check the thesis report included in the repository.
