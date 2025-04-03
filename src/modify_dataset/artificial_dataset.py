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

    reward_mean = np.mean(dataset["rewards"])
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
    terminals = dataset["terminals"] | dataset["timeouts"]

    save_data = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "true_rewards": true_rewards,
    }
    np.savez(new_dataset_path, **save_data)

    # positive noise
    new_dataset_path = get_new_dataset_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="artificial",
        reward_model_algo="positive-noise",
    )
    true_rewards = dataset["rewards"]
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = true_rewards.copy()
    terminals = dataset["terminals"] | dataset["timeouts"]
    noise_start = np.percentile(
        rewards, q=75
    )
    noise_end = np.percentile(
        rewards, q=95
    )

    noise_mask = (rewards >= noise_start) & (rewards < noise_end)
    noise = np.abs(np.random.normal(loc=0.0, scale=reward_std, size=rewards[noise_mask].shape))
    rewards[noise_mask] += noise

    save_data = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "true_rewards": true_rewards,
    }
    np.savez(new_dataset_path, **save_data)

    # negative noise
    new_dataset_path = get_new_dataset_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="artificial",
        reward_model_algo="negative-noise",
    )
    true_rewards = dataset["rewards"]
    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = true_rewards.copy()
    terminals = dataset["terminals"] | dataset["timeouts"]
    noise_start = np.percentile(
        rewards, q=75
    )
    noise_end = np.percentile(
        rewards, q=95
    )

    noise_mask = (rewards >= noise_start) & (rewards < noise_end)
    noise = np.abs(np.random.normal(loc=0.0, scale=reward_std, size=rewards[noise_mask].shape))
    rewards[noise_mask] -= noise


    save_data = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "true_rewards": true_rewards,
    }
    np.savez(new_dataset_path, **save_data)

    # noise
    # for noise in [0.25, 0.5, 0.75]:
    #     new_dataset_path = get_new_dataset_path(
    #         env_name=env_name,
    #         exp_name=exp_name,
    #         pair_algo="artificial",
    #         reward_model_algo=f"noise_{noise}",
    #     )

    #     rewards = true_rewards.copy()

    #     has_noise = np.random.rand(len(rewards)) < noise

    #     rewards[has_noise] = np.random.normal(
    #         loc=reward_mean, scale=reward_std, size=rewards[has_noise].shape
    #     )

    #     save_data = {
    #         "observations": observations,
    #         "actions": actions,
    #         "rewards": rewards,
    #         "terminals": terminals,
    #         "true_rewards": true_rewards,
    #     }
    #     np.savez(new_dataset_path, **save_data)

    indices = extract_trajectory_indices(dataset=dataset)

    # clean
    for ratio in [0.25, 0.5]:
        new_dataset_path = get_new_dataset_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="artificial",
            reward_model_algo=f"clean-{ratio}",
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
            selected_rewards.append(true_rewards[start:end])
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
    
    # front
    for ratio in [0.25, 0.5]:
        new_dataset_path = get_new_dataset_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="artificial",
            reward_model_algo=f"front-{ratio}",
        )

        num_trajectories = len(indices)
        num_selected = int(num_trajectories * ratio)

        selected_indices = range(num_selected)

        selected_observations = []
        selected_actions = []
        selected_rewards = []
        selected_terminals = []
        selected_true_rewards = []

        for idx in selected_indices:
            start, end = indices[idx]
            selected_observations.append(observations[start:end])
            selected_actions.append(actions[start:end])
            selected_rewards.append(true_rewards[start:end])
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
