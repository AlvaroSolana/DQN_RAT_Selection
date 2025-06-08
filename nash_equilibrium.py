import numpy as np
import torch
from tqdm import tqdm
from NashRL import expand_list
from evaluate import *




def dqn_nash_eq(rat_env, action_network, n_episodes, k):
    """
    Evaluates empirical Nash Equilibrium rate using actual environment rewards.
    Aggregates results every k episodes and plots the average NE rate.
    """

    action_network.eval()
    total_users = rat_env.n_users
    total_equilibrium_cases = 0
    total_users_checked = 0
    aggregated_rates = []

    for episode in tqdm(range(n_episodes), desc="Evaluating Dqn Episodes", position=0):
        rat_env.reset()
        for _ in range(30):  # or rat_env.max_steps
            state, _, _ = rat_env.get_state()
            state_expanded = expand_list(state, total_users)

            # Get actions via policy
            q_values = predict_action(action_network, state_expanded[:, :rat_env.n_stations])
            actions = torch.argmax(q_values, dim=1)

            # Step with joint action to get actual rewards
            _, _, _, base_rewards = rat_env.step(actions.detach())

            # Evaluate NE by comparing actual reward to each user's unilateral deviation
            for user in range(total_users):
                actual_action = actions[user].item()
                actual_reward = base_rewards[user].item()

                best_alt_reward = actual_reward
                for alt_action in range(rat_env.n_stations):
                    if alt_action == actual_action:
                        continue

                    # Create a new action vector with one deviation
                    alt_actions = actions.clone()
                    alt_actions[user] = alt_action

                    # You will need to implement environment cloning or state save/restore here
                    env_clone = rat_env.clone()  # or use save/load state method
                    _, _, _, alt_rewards = env_clone.step(alt_actions)
                    alt_reward = alt_rewards[user].item()

                    if alt_reward > best_alt_reward + 1e-6:
                        break
                else:
                    total_equilibrium_cases += 1  # No unilateral deviation improved reward

                total_users_checked += 1

        # Aggregate every k episodes
        if (episode + 1) % k == 0:
            rate = 100 * total_equilibrium_cases / total_users_checked if total_users_checked > 0 else 0
            aggregated_rates.append(rate)
            total_equilibrium_cases = 0
            total_users_checked = 0

    return aggregated_rates


def heuristic_nash_eq(rat_env, n_episodes, k, h_value, p_switch, m):
    """
    Evaluates empirical Nash Equilibrium rate for heuristic approach using actual environment rewards.
    Aggregates results every k episodes and returns list of average NE rates.

    Params:
        rat_env: environment object
        n_episodes: total episodes to simulate
        k: aggregation window (number of episodes to average over)
        h_value, p_switch, m: heuristic parameters (same as used in run_heuristic)
    Returns:
        aggregated_rates: list of Nash Equilibrium rates (%) per k episodes
    """
    total_users = rat_env.n_users
    total_equilibrium_cases = 0
    total_users_checked = 0
    aggregated_rates = []

    for episode in tqdm(range(n_episodes), desc="Evaluating Heuristic Episodes"):
        rat_env.reset()
        n_steps = rat_env.n_steps
        
        for step in range(n_steps):
            # Run heuristic logic to get actions at this step
            current_state, _, _ = rat_env.get_state()
            state = expand_list(current_state, total_users)[:, :rat_env.n_stations]
            new_rats = []
            switch_register = []

            for user in range(total_users):
                switch = False
                current_rat = rat_env.user_assignments[user]
                current_thr = rat_env.get_ue_throughput(user)
                user_rates = state[user, :]
                
                for rat in range(rat_env.n_stations):
                    rat_id = 0 if rat < rat_env.n_ltesn else 1
                    node_id = rat - rat_id * rat_env.n_ltesn
                    if ([rat_id, node_id] != current_rat) and (user_rates[rat] > 0):
                        rat_env.user_assignments[user] = [rat_id, node_id]
                        possible_thr = rat_env.get_ue_throughput(user)
                        if possible_thr > current_thr * h_value:
                            if np.random.random() < p_switch ** (m + 1):
                                new_rat = node_id + rat_id * rat_env.n_ltesn
                                changed_rats = [rat_id for rat_id, flag in zip(new_rats, switch_register) if flag]
                                if new_rat in changed_rats:
                                    m += 1
                                else:
                                    m = 0
                                new_rats.append(new_rat)
                                switch = True
                                break
                rat_env.user_assignments[user] = current_rat
                switch_register.append(switch)
                if not switch:
                    new_rats.append(current_rat[1] + current_rat[0] * rat_env.n_ltesn)

            actions = torch.tensor(new_rats)

            # Step environment with heuristic actions and get base rewards
            _, _, _, base_rewards = rat_env.step(actions)

            # --- Nash Equilibrium check ---
            for user in range(total_users):
                actual_action = actions[user].item()
                actual_reward = base_rewards[user].item()
                best_alt_reward = actual_reward

                for alt_action in range(rat_env.n_stations):
                    if alt_action == actual_action:
                        continue
                    
                    alt_actions = actions.clone()
                    alt_actions[user] = alt_action
                    env_clone = rat_env.clone()
                    _, _, _, alt_rewards = env_clone.step(alt_actions)
                    alt_reward = alt_rewards[user].item()
                    
                    if alt_reward > best_alt_reward + 1e-6:
                        break
                else:
                    total_equilibrium_cases += 1

                total_users_checked += 1

        # Aggregate results every k episodes
        if (episode + 1) % k == 0:
            rate = 100 * total_equilibrium_cases / total_users_checked if total_users_checked > 0 else 0
            aggregated_rates.append(rate)
            total_equilibrium_cases = 0
            total_users_checked = 0

    return aggregated_rates


def hartrl_nash_eq(rat_env, p_vectors, n_episodes, k):
    """
    Evaluates empirical Nash Equilibrium rate using actual environment rewards.
    Aggregates results every k episodes and plots the average NE rate.
    """

    total_users = rat_env.n_users
    total_equilibrium_cases = 0
    total_users_checked = 0
    aggregated_rates = []

    for episode in tqdm(range(n_episodes), desc="Evaluating HartRL Episodes", position=0):
        rat_env.reset()
        for t in range(0, 30):
            actions = torch.zeros(rat_env.n_users, dtype=torch.int64)
            for i,p_vector in enumerate(p_vectors):
                #print(f"p_vector for agent {i}: {p_vector}")
                action = np.random.choice(len(p_vector), p=p_vector.cpu().numpy() / p_vector.sum().item())
                actions[i] = action
            _,_,_,base_rewards = rat_env.step(actions)

            # Evaluate NE by comparing actual reward to each user's unilateral deviation
            for user in range(total_users):
                actual_action = actions[user].item()
                actual_reward = base_rewards[user].item()
                best_alt_reward = actual_reward
                for alt_action in range(rat_env.n_stations):
                    if alt_action == actual_action:
                        continue
                    # Create a new action vector with one deviation
                    alt_actions = actions.clone()
                    alt_actions[user] = alt_action
                    # You will need to implement environment cloning or state save/restore here
                    env_clone = rat_env.clone()  # or use save/load state method
                    _, _, _, alt_rewards = env_clone.step(alt_actions)
                    alt_reward = alt_rewards[user].item()
                    if alt_reward > best_alt_reward + 1e-6:
                        break
                else:
                    total_equilibrium_cases += 1  # No unilateral deviation improved reward

                total_users_checked += 1

        # Aggregate every k episodes
        if (episode + 1) % k == 0:
            rate = 100 * total_equilibrium_cases / total_users_checked if total_users_checked > 0 else 0
            aggregated_rates.append(rate)
            total_equilibrium_cases = 0
            total_users_checked = 0

    return aggregated_rates
