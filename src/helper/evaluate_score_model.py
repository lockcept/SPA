from matplotlib import pyplot as plt
import numpy as np
import torch
from data_generation import RNN
from data_loading import get_dataloader, load_pair, load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_score_model(env_name, model_path, pair_name):
    data_loader, obs_dim, act_dim = get_dataloader(
        env_name=env_name, pair_name=pair_name, shuffle=False, drop_last=False
    )
    print(f"evaluate {model_path} with pair {pair_name}")

    model, _ = RNN.initialize(
        config={"obs_dim": obs_dim, "act_dim": act_dim},
        path=model_path,
        skip_if_exists=False,
    )

    model.eval()

    score_list = []

    answer_count = 0

    with torch.no_grad():
        for batch in data_loader:
            (
                s0_obs_batch,
                s0_act_batch,
                s1_obs_batch,
                s1_act_batch,
                mu_batch,
                mask_batch,
            ) = [x.to(device) for x in batch]

            s0_batch = torch.cat((s0_obs_batch, s0_act_batch), dim=-1)
            s1_batch = torch.cat((s1_obs_batch, s1_act_batch), dim=-1)

            lengths = (1 - mask_batch.squeeze()).sum(dim=1)

            s0_score = model(s0_batch, lengths)
            s1_score = model(s1_batch, lengths)

            scores = torch.cat((s0_score, s1_score), dim=1)
            score_list.append(scores.detach().cpu().numpy())

            mu_batch = mu_batch.unsqueeze(1)

            condition = ((s0_score <= s1_score) & (0.5 <= mu_batch)) | (
                (s0_score >= s1_score) & (0.5 >= mu_batch)
            )
            answer_count += torch.sum(condition).item()

    print("ACC:", answer_count / len(data_loader.dataset))
    score_list = np.concatenate(score_list, axis=0)
    score_list = np.concatenate(score_list, axis=0)

    dataset = load_dataset(env_name)
    pairs = load_pair(env_name, pair_name)
    reward_sum_list = []
    for s0, s1, _ in pairs["data"]:
        reward_sum_0 = np.sum(dataset["rewards"][s0[0] : s0[1]])
        reward_sum_1 = np.sum(dataset["rewards"][s1[0] : s1[1]])
        reward_sum_list.append(reward_sum_0)
        reward_sum_list.append(reward_sum_1)

    pearson_corr = np.corrcoef(reward_sum_list, score_list)[0, 1]

    model_name = model_path.split("/")[-1].split(".")[0]
    pair_name_base = pair_name.split("_")[0]

    label = f"{env_name}: {model_name}, {pair_name_base}"
    plt.figure(figsize=(8, 6))
    plt.scatter(reward_sum_list, score_list, alpha=0.5, label=label)
    plt.xlabel("Sum of Rewards")
    plt.ylabel("Trajectory Score")
    plt.title(f"Pearson Correlation: {pearson_corr:.2f}")
    plt.legend()
    plt.grid(True)

    output_path = f"log/score_{env_name}_{model_name}_{pair_name_base}.png"
    plt.savefig(output_path, format="png")
    plt.close()
