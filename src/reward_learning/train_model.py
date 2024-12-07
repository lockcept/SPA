from data_loading import get_dataloader
from reward_learning.MR import MR
from helper import get_reward_model_path


def train_reward_model(
    env_name, exp_name, pair_algo, reward_model_algo, reward_model_tag, num_epoch
):
    """
    train reward model
    """
    train_data_loader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=pair_algo,
    )

    obs_dim, act_dim = train_data_loader.dataset.get_dimensions()

    val_data_loader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pair_algo=pair_algo,
    )

    print("obs_dim:", obs_dim, "act_dim:", act_dim)

    reward_model_path = get_reward_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag=reward_model_tag,
    )

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
            train_loader=train_data_loader,
            val_loader=val_data_loader,
            optimizer=optimizer,
            num_epochs=num_epoch,
        )
