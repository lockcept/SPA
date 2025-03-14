from reward_learning.MR import MR


def get_reward_model(reward_model_algo, model_path, allow_existing, obs_dim, act_dim):
    """
    Return reward model
    """
    if reward_model_algo == "MR":
        model, optimizer = MR.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
            linear_loss=False,
            allow_existing=allow_existing,
        )
    elif reward_model_algo == "MR-linear":
        model, optimizer = MR.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
            linear_loss=True,
            allow_existing=allow_existing,
        )
    else:
        model = None
        optimizer = None

    return model, optimizer
