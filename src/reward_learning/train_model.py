from data_loading import get_dataloader
from reward_learning.MR import MR


def train_reward_model(
    env_name, pair_name, pair_val_name, reward_model_algo, reward_model_path, num
):
    """
    train reward model
    """
    data_loader, obs_dim, act_dim = get_dataloader(
        env_name=env_name,
        pair_name=pair_name,
    )

    val_data_loader, _, _ = get_dataloader(
        env_name=env_name,
        pair_name=pair_val_name,
    )

    print("obs_dim:", obs_dim, "act_dim:", act_dim)

    if reward_model_algo == "MR":
        model, optimizer = MR.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=reward_model_path,
            skip_if_exists=True,
        )
    elif reward_model_algo == "MR-linear":
        model, optimizer = MR.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=reward_model_path,
            linear_loss=True,
            skip_if_exists=True,
        )
    else:
        model = None
        optimizer = None

    if model is not None:
        model.train_model(
            train_loader=data_loader,
            val_loader=val_data_loader,
            optimizer=optimizer,
            num_epochs=num,
        )
