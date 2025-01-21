import random

import numpy as np
import torch
import wandb

from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, DiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger

from data_loading import get_env
from utils import get_new_dataset_path, get_policy_model_path


def get_configs():
    """
    suggested hypers
    expectile=0.7, temperature=3.0 for all D4RL-Gym tasks
    """
    hidden_dims = [256, 256]
    actor_lr = 3e-4
    critic_q_lr = 3e-4
    critic_v_lr = 3e-4
    dropout_rate = None
    lr_decay = True
    gamma = 0.99
    tau = 0.005
    expectile = 0.7
    temperature = 3.0
    epoch = 300
    step_per_epoch = 1000
    eval_episodes = 5
    batch_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "hidden_dims": hidden_dims,
        "actor_lr": actor_lr,
        "critic_q_lr": critic_q_lr,
        "critic_v_lr": critic_v_lr,
        "dropout_rate": dropout_rate,
        "lr_decay": lr_decay,
        "gamma": gamma,
        "tau": tau,
        "expectile": expectile,
        "temperature": temperature,
        "epoch": epoch,
        "step_per_epoch": step_per_epoch,
        "eval_episodes": eval_episodes,
        "batch_size": batch_size,
        "device": device,
    }


def normalize_rewards(dataset):
    terminals_float = np.zeros_like(dataset["rewards"])
    for i in range(len(terminals_float) - 1):
        if (
            np.linalg.norm(
                dataset["observations"][i + 1] - dataset["next_observations"][i]
            )
            > 1e-6
            or dataset["terminals"][i] == 1.0
        ):
            terminals_float[i] = 1
        else:
            terminals_float[i] = 0

    terminals_float[-1] = 1

    # split_into_trajectories
    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append(
            (
                dataset["observations"][i],
                dataset["actions"][i],
                dataset["rewards"][i],
                1.0 - dataset["terminals"][i],
                terminals_float[i],
                dataset["next_observations"][i],
            )
        )
        if terminals_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    # normalize rewards
    dataset["rewards"] /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset["rewards"] *= 1000.0

    return dataset


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
    )
    wandb.run.save()


def train(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
):
    """
    Train IQL policy on the given dataset
    """
    policy_dir = get_policy_model_path(env_name, exp_name, pair_algo, reward_model_algo)

    # import gym lazyly to reduce the overhead
    from offlinerlkit.policy_trainer import MFPolicyTrainer  # pylint: disable=C0415
    from offlinerlkit.policy import IQLPolicy  # pylint: disable=C0415

    configs = get_configs()
    # create env and dataset
    env = get_env(env_name, is_hidden=False)
    dataset_path = get_new_dataset_path(
        env_name, exp_name, pair_algo, reward_model_algo
    )
    dataset_npz = np.load(dataset_path)
    dataset = {key: dataset_npz[key] for key in dataset_npz}
    dataset = qlearning_dataset(env, dataset=dataset)
    if "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    if "halfcheetah" in env_name or "walker2d" in env_name or "hopper" in env_name:
        dataset = normalize_rewards(dataset)
        print("normalized rewards")
    configs.update(
        {
            "obs_shape": env.observation_space.shape,
            "action_dim": np.prod(env.action_space.shape),
            "max_action": env.action_space.high[0],
        }
    )

    print(configs)

    # seed

    seed = random.randint(0, 2**31 - 1)
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # wandb

    configs.update({"seed": seed})
    configs.update({"project": "iql"})
    group = f"{env_name}_{exp_name.split('-')[0]}_{pair_algo}_{reward_model_algo}"
    configs.update({"group": group})
    configs.update({"name": exp_name})
    wandb_init(config=configs)

    # create policy model
    actor_backbone = MLP(
        input_dim=np.prod(configs["obs_shape"]),
        hidden_dims=configs["hidden_dims"],
        dropout_rate=configs["dropout_rate"],
    )
    critic_q1_backbone = MLP(
        input_dim=np.prod(configs["obs_shape"]) + configs["action_dim"],
        hidden_dims=configs["hidden_dims"],
    )
    critic_q2_backbone = MLP(
        input_dim=np.prod(configs["obs_shape"]) + configs["action_dim"],
        hidden_dims=configs["hidden_dims"],
    )
    critic_v_backbone = MLP(
        input_dim=np.prod(configs["obs_shape"]), hidden_dims=configs["hidden_dims"]
    )
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=configs["action_dim"],
        unbounded=False,
        conditioned_sigma=False,
        max_mu=configs["max_action"],
    )
    actor = ActorProb(actor_backbone, dist, configs["device"])
    critic_q1 = Critic(critic_q1_backbone, configs["device"])
    critic_q2 = Critic(critic_q2_backbone, configs["device"])
    critic_v = Critic(critic_v_backbone, configs["device"])

    for m in (
        list(actor.modules())
        + list(critic_q1.modules())
        + list(critic_q2.modules())
        + list(critic_v.modules())
    ):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=configs["actor_lr"])
    critic_q1_optim = torch.optim.Adam(
        critic_q1.parameters(), lr=configs["critic_q_lr"]
    )
    critic_q2_optim = torch.optim.Adam(
        critic_q2.parameters(), lr=configs["critic_q_lr"]
    )
    critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=configs["critic_v_lr"])

    if configs["lr_decay"]:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            actor_optim, configs["epoch"]
        )
    else:
        lr_scheduler = None

    # create IQL policy
    policy = IQLPolicy(
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        actor_optim,
        critic_q1_optim,
        critic_q2_optim,
        critic_v_optim,
        action_space=env.action_space,
        tau=configs["tau"],
        gamma=configs["gamma"],
        expectile=configs["expectile"],
        temperature=configs["temperature"],
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=configs["obs_shape"],
        obs_dtype=np.float32,
        action_dim=configs["action_dim"],
        action_dtype=np.float32,
        device=configs["device"],
    )
    buffer.load_dataset(dataset)

    # log
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(policy_dir, output_config)
    logger.log_hyperparameters(configs)

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        logger=logger,
        epoch=configs["epoch"],
        step_per_epoch=configs["step_per_epoch"],
        batch_size=configs["batch_size"],
        eval_episodes=configs["eval_episodes"],
        lr_scheduler=lr_scheduler,
    )

    # train
    policy_trainer.train()
