import torch
from data_generation import RNN
from data_loading import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_score_model(env_name, model_path, pair_name):
    data_loader, obs_dim, act_dim = get_dataloader(
        env_name=env_name, pair_name=pair_name, drop_last=False
    )
    print(model_path)

    model, _ = RNN.initialize(
        config={"obs_dim": obs_dim, "act_dim": act_dim},
        path=model_path,
        skip_if_exists=False,
    )

    model.eval()

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

            mu_batch = mu_batch.unsqueeze(1)

            condition = ((s0_score <= s1_score) & (0.5 <= mu_batch)) | (
                (s0_score >= s1_score) & (0.5 >= mu_batch)
            )
            answer_count += torch.sum(condition).item()

    print(answer_count / len(data_loader.dataset))
