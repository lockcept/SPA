import csv
import os
from data_generation.utils import extract_trajectory_indices
from data_loading.load_data import load_dataset, load_pair


def analyze_pair(env_name, exp_name, pair_type, pair_algo):
    dataset = load_dataset(env_name)
    indices = extract_trajectory_indices(dataset)
    pair = load_pair(env_name, exp_name, pair_type, pair_algo)

    # 3: good trajectory selection count
    success_count = 0
    count = 0

    for s0, s1, _ in pair:
        s0_traj = next(
            (traj for traj in indices if traj[0] <= s0[0] and traj[1] >= s0[1]), None
        )
        s1_traj = next(
            (traj for traj in indices if traj[0] <= s1[0] and traj[1] >= s1[1]), None
        )

        s0_success = (
            any(dataset["success"][s0_traj[0] : s0_traj[1]] > 0) if s0_traj else False
        )
        s1_success = (
            any(dataset["success"][s1_traj[0] : s1_traj[1]] > 0) if s1_traj else False
        )

        if s0_success:
            success_count += 1
        if s1_success:
            success_count += 1

        count += 2

    print("success_count", success_count, count)

    # 4: goal approach index ratio

    cut_after_success_count = 0
    count = 0

    for s0, s1, _ in pair:
        s0_traj = next(
            (traj for traj in indices if traj[0] <= s0[0] and traj[1] >= s0[1]), None
        )
        s1_traj = next(
            (traj for traj in indices if traj[0] <= s1[0] and traj[1] >= s1[1]), None
        )

        s0_cut_after_success = (
            any(dataset["success"][s0_traj[0] : s0[0]] > 0) if s0_traj else False
        )
        s1_cut_after_success = (
            any(dataset["success"][s1_traj[0] : s1[0]] > 0) if s1_traj else False
        )

        if s0_cut_after_success:
            cut_after_success_count += 1
        if s1_cut_after_success:
            cut_after_success_count += 1

        count += 2

    print("cut_after_success_count", cut_after_success_count, count)

    log_path = "log/good_pair.csv"
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_path, "a", encoding="utf-8", newline="") as log_file:
        writer = csv.writer(log_file)

        if log_file.tell() == 0:
            writer.writerow(
                [
                    "EnvName",
                    "ExpName",
                    "PairAlgo",
                    "SuccessTrajRatio",
                    "CutAfterSuccessTrajRatio",
                ]
            )

        success_traj_ratio = f"{success_count / count:.4f}"
        cut_after_success_traj_ratio = f"{cut_after_success_count / count:.4f}"

        writer.writerow(
            [
                env_name,
                exp_name,
                pair_algo,
                success_traj_ratio,
                cut_after_success_traj_ratio,
            ]
        )
