import csv
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from data_generation import RNNModel, LSTMModel
from data_loading import get_dataloader, load_pair, load_dataset
from utils import get_score_model_path, get_score_model_log_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_score_model(env_name, exp_name, pair_algo, test_pair_type, test_pair_algo):
    data_loader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=test_pair_type,
        pair_algo=test_pair_algo,
        shuffle=False,
        drop_last=False,
    )
    print(f"evaluate pair {exp_name} {pair_algo}")

    obs_dim, act_dim = data_loader.dataset.get_dimensions()

    raw_pair_algo = "-".join(pair_algo.split("-")[1:])
    score_model_algo = pair_algo.split("-")[0]

    model_path = get_score_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=raw_pair_algo,
        score_model=score_model_algo,
        ensemble_num=0,
    )

    if score_model_algo == "rnn":
        model = RNNModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
        )
    elif score_model_algo == "lstm.exp":
        model, _ = LSTMModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
            skip_if_exists=False,
        )
    elif score_model_algo == "lstm.linear":
        model, _ = LSTMModel.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
            skip_if_exists=False,
            linear_loss=True,
        )
    else:
        raise ValueError(f"Invalid score model algo: {score_model_algo}")

    model.eval()

    score_list = []

    answer_count = 0

    filtered_s0_scores = []
    filtered_s1_scores = []

    with torch.no_grad():
        for batch in data_loader:
            (
                s0_obs_batch,
                s0_act_batch,
                s1_obs_batch,
                s1_act_batch,
                mu_batch,
                mask0_batch,
                mask1_batch,
            ) = [x.to(device) for x in batch]

            s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1)
            s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1)

            lengths_s0 = (1 - mask0_batch.squeeze()).sum(dim=1)
            lengths_s1 = (1 - mask1_batch.squeeze()).sum(dim=1)

            s0_score = model(s0_batch, lengths_s0)
            s1_score = model(s1_batch, lengths_s1)

            condition = (lengths_s0 > 0) & (lengths_s1 > 0)
            filtered_s0_scores.extend(s0_score[condition].detach().cpu().numpy())
            filtered_s1_scores.extend(s1_score[condition].detach().cpu().numpy())

            scores = torch.cat((s0_score, s1_score), dim=1)
            score_list.append(scores.detach().cpu().numpy())

            mu_batch = mu_batch.unsqueeze(1)

            condition = ((s0_score <= s1_score) & (0.5 <= mu_batch)) | (
                (s0_score >= s1_score) & (0.5 >= mu_batch)
            )
            answer_count += torch.sum(condition).item()

    # Plotting the relationship between s0_score and s1_score
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_s0_scores, filtered_s1_scores, alpha=0.7, edgecolors="k")
    plt.xlabel("s0_score")
    plt.ylabel("s1_score")
    plt.title("Relationship between s0_score and s1_score")
    plt.grid(True)
    plt.savefig(
        get_score_model_log_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=raw_pair_algo,
            score_model=pair_algo.split("-")[0],
            ensemble_num=0,
            log_file=f"score_s0_s1_{test_pair_type}.png",
        ),
        format="png",
    )
    plt.close()

    # Plotting the relationship between score and sum of rewards
    score_list = np.concatenate(score_list, axis=0)
    score_list = np.concatenate(score_list, axis=0)

    dataset = load_dataset(env_name)
    pairs = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=test_pair_type,
        pair_algo=test_pair_algo,
    )
    reward_sum_list = []
    for s0, s1, _ in pairs["data"]:
        reward_sum_0 = np.sum(dataset["rewards"][s0[0] : s0[1]])
        reward_sum_1 = np.sum(dataset["rewards"][s1[0] : s1[1]])
        reward_sum_list.append(reward_sum_0)
        reward_sum_list.append(reward_sum_1)

    pearson_corr = np.corrcoef(reward_sum_list, score_list)[0, 1]

    label = f"{env_name}: {exp_name}, {pair_algo}"
    plt.figure(figsize=(8, 6))
    plt.scatter(reward_sum_list, score_list, alpha=0.5, label=label)
    plt.xlabel("Sum of Rewards")
    plt.ylabel("Trajectory Score")
    plt.title(f"Pearson Correlation: {pearson_corr:.2f}")
    plt.legend()
    plt.grid(True)

    output_path = get_score_model_log_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=raw_pair_algo,
        score_model=pair_algo.split("-")[0],
        ensemble_num=0,
        log_file=f"true_reward_{test_pair_type}.png",
    )
    plt.savefig(output_path, format="png")
    plt.close()

    log_path = "log/main_evaluate_score.csv"
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
                    "TestPairType",
                    "ScoreModelAlgo",
                    "Accuracy",
                    "PCC",
                ]
            )

        formatted_accuracy = f"{answer_count/len(data_loader.dataset):.4f}"
        formatted_pcc = f"{pearson_corr:.4f}"

        writer.writerow(
            [
                env_name,
                exp_name,
                raw_pair_algo,
                test_pair_type,
                score_model_algo,
                formatted_accuracy,
                formatted_pcc,
            ]
        )
