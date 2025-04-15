from reward_learning.MR import MR
from reward_learning.MR_mc_dropout import MRWithMCDropout


def get_reward_model(reward_model_algo, model_path, allow_existing, obs_dim, act_dim):
    """
    Return reward model
    """
    if reward_model_algo == "MR-exp":
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
    elif reward_model_algo == "MR-dropout":
        model, optimizer = MRWithMCDropout.initialize(
            config={"obs_dim": obs_dim, "act_dim": act_dim},
            path=model_path,
            allow_existing=allow_existing,
        )
    else:
        model = None
        optimizer = None

    return model, optimizer
