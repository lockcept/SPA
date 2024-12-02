import torch
from data_generation import RNN
from data_loading import load_dataset, load_pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_score_model(env_name, model_path, pair_name):
    dataset = load_dataset(env_name)
    pair = load_pair(env_name, pair_name)

    obs_dim = dataset["observations"].shape[-1]
    act_dim = dataset["actions"].shape[-1]

    model, _ = RNN.initialize(
        config={"obs_dim": obs_dim, "act_dim": act_dim}, path=model_path
    )

    answer_count = 0
    count = 0

    for s0, s1, mu in pair["data"]:
        print(count, answer_count)
        s0_obs = dataset["observations"][s0[0] : s0[1]]
        s0_act = dataset["actions"][s0[0] : s0[1]]
        s1_obs = dataset["observations"][s1[0] : s1[1]]
        s1_act = dataset["actions"][s1[0] : s1[1]]

        s0_obs = torch.tensor(s0_obs, dtype=torch.float32).to(device)
        s0_act = torch.tensor(s0_act, dtype=torch.float32).to(device)
        s1_obs = torch.tensor(s1_obs, dtype=torch.float32).to(device)
        s1_act = torch.tensor(s1_act, dtype=torch.float32).to(device)

        s0_state = torch.cat((s0_obs, s0_act), dim=-1)
        s1_state = torch.cat((s1_obs, s1_act), dim=-1)

        s0_score = model(s0_state).detach().numpy()
        s1_score = model(s1_state).detach().numpy()

        if s0_score > s1_score and mu < 0.5:
            answer_count += 1
        elif s0_score < s1_score and mu > 0.5:
            answer_count += 1
        count += 1

    print(answer_count / len(pair["data"]))
