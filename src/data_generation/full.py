import numpy as np
import random


def load_dataset(file_path):
    dataset = np.load(file_path)
    return dataset


def extract_trajectory_indices(terminals, timeouts):
    indices = []
    start = 0
    for i in range(len(terminals)):
        if terminals[i] or timeouts[i]:
            end = i
            indices.append((start, end))
            start = i + 1
    return indices


def sample_trajectory(dataset, indices):
    start, end = random.choice(indices)
    trajectory = {
        "observations": dataset["observations"][start : end + 1],
        "actions": dataset["actions"][start : end + 1],
        "rewards": dataset["rewards"][start : end + 1],
    }
    return trajectory, (start, end)


def compare_trajectories(traj1, traj2):
    reward_sum_1 = np.sum(traj1["rewards"])
    reward_sum_2 = np.sum(traj2["rewards"])

    if reward_sum_1 > reward_sum_2:
        return 0
    else:
        return 1


def generate_preference_pair(dataset, indices):
    traj1, (start1, end1) = sample_trajectory(dataset, indices)
    traj2, (start2, end2) = sample_trajectory(dataset, indices)

    winner = compare_trajectories(traj1, traj2)
    preference_pair = {"s0": (start1, end1), "s1": (start2, end2), "winner": winner}

    print(
        f"Trajectory 1: {end1-start1+1}, Trajectory 2: {end2-start2+1}, Winner: {winner}"
    )

    return preference_pair


def save_preference_pairs(file_path, preference_pairs):
    np.savez(file_path, preference_pairs=preference_pairs)
    print(f"Preference pairs saved at {file_path}")


def scripted_teacher(env, num_pairs=10):
    dataset_file_path = f"dataset/d4rl/{env}/dataset.npz"
    dataset = load_dataset(dataset_file_path)

    indices = extract_trajectory_indices(dataset["terminals"], dataset["timeouts"])

    length_stat = int(np.mean([end - start for start, end in indices]))
    print(length_stat)

    preference_pairs = []
    for _ in range(num_pairs):
        preference_pair = generate_preference_pair(dataset, indices)
        preference_pairs.append(preference_pair)

    save_path = f"dataset/pbrl/{env}/preference_pairs.npz"
    save_preference_pairs(save_path, preference_pairs)


if __name__ == "__main__":
    env = "maze2d-medium-dense-v1"
    scripted_teacher(env)
