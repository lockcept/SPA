import numpy as np

from data_loading import load_dataset, extract_trajectory_indices
from utils import get_new_dataset_path


def make_artificial_dataset(env_name, exp_name):
    """
    change reward and save it to new_dataset_path
    """

    dataset = load_dataset(env_name)
    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    print("obs_dim:", obs_dim, "act_dim:", act_dim)

    reward_std = np.std(dataset["rewards"])

    # original
    new_dataset_path = get_new_dataset_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="artificial",
        reward_model_algo="original",
    )
    true_rewards = dataset["rewards"]
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]

    save_data = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "true_rewards": true_rewards,
    }
    np.savez(new_dataset_path, **save_data)

    # noise
    for noise in [0.25, 0.5, 0.75]:
        new_dataset_path = get_new_dataset_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="artificial",
            reward_model_algo=f"noise_{noise}",
        )

        # add noise for noise 확률
        rewards = true_rewards.copy()
        has_noise = np.random.rand(len(rewards)) < noise

        noise_values = np.zeros_like(rewards)
        noise_values[has_noise] = np.random.normal(
            loc=0.0, scale=reward_std, size=rewards[has_noise].shape
        )

        rewards += noise_values

        save_data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": terminals,
            "true_rewards": true_rewards,
        }
        np.savez(new_dataset_path, **save_data)

    indices = extract_trajectory_indices(dataset=dataset)

    # clean
    for ratio in [0.25, 0.5, 0.75]:
        new_dataset_path = get_new_dataset_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="artificial",
            reward_model_algo=f"clean_{ratio}",
        )

        num_trajectories = len(indices)
        num_selected = int(num_trajectories * ratio)

        selected_indices = np.random.choice(
            num_trajectories, num_selected, replace=False
        )

        selected_observations = []
        selected_actions = []
        selected_rewards = []
        selected_terminals = []
        selected_true_rewards = []

        for idx in selected_indices:
            start, end = indices[idx]
            selected_observations.append(observations[start:end])
            selected_actions.append(actions[start:end])
            selected_rewards.append(rewards[start:end])
            selected_terminals.append(terminals[start:end])
            selected_true_rewards.append(true_rewards[start:end])

        selected_observations = np.concatenate(selected_observations, axis=0)
        selected_actions = np.concatenate(selected_actions, axis=0)
        selected_rewards = np.concatenate(selected_rewards, axis=0)
        selected_terminals = np.concatenate(selected_terminals, axis=0)
        selected_true_rewards = np.concatenate(selected_true_rewards, axis=0)

        save_data = {
            "observations": selected_observations,
            "actions": selected_actions,
            "rewards": selected_rewards,
            "terminals": selected_terminals,
            "true_rewards": selected_true_rewards,
        }
        np.savez(new_dataset_path, **save_data)
