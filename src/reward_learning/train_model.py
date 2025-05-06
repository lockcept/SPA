from data_loading import get_dataloader
from reward_learning.MR_SURF import MR_SURF
from reward_learning.get_model import get_reward_model
from utils import get_reward_model_path


def train_reward_model(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
    reward_model_tag,
    num_epoch,
    unlabel_pair_algo=None,
    train_from_existing=False,
    no_val_data=True,
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

    if unlabel_pair_algo is not None:
        unlabel_data_loader = get_dataloader(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo=unlabel_pair_algo,
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
        if isinstance(model, MR_SURF):
            model.train_model_with_surf(
                labeled_loader=train_data_loader,
                unlabeled_loader=unlabel_data_loader,
                val_loader=val_data_loader,
                optimizer=optimizer,
                num_epochs=num_epoch,
            )
        else:
            model.train_model(
                train_loader=train_data_loader,
                val_loader=val_data_loader,
                optimizer=optimizer,
                num_epochs=num_epoch,
            )
