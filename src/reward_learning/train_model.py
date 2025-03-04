from data_loading import get_dataloader
from reward_learning.get_reward_model import get_reward_model
from utils import get_reward_model_path


def train_reward_model(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
    reward_model_tag,
    num_epoch,
    train_from_existing=False,
    no_val_data=False,
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

    try:
        val_data_loader = get_dataloader(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="val",
            pair_algo=pair_algo,
        )
    except FileNotFoundError as e:
        if no_val_data:
            val_data_loader = None
        else:
            print(f"Error loading validation data: {e}")
            raise e

    print("obs_dim:", obs_dim, "act_dim:", act_dim)

    reward_model_path = get_reward_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag=reward_model_tag,
    )

    model, optimizer = get_reward_model(
        reward_model_algo=reward_model_algo,
        obs_dim=obs_dim,
        act_dim=act_dim,
        model_path=reward_model_path,
        allow_existing=train_from_existing,
    )

    if model is not None:
        model.train_model(
            train_loader=train_data_loader,
            val_loader=val_data_loader,
            optimizer=optimizer,
            num_epochs=num_epoch,
        )
