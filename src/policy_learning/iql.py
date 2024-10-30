import random
import os

import gym
import d4rl

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, DiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MFPolicyTrainer
from offlinerlkit.policy import IQLPolicy

"""
suggested hypers
expectile=0.7, temperature=3.0 for all D4RL-Gym tasks
"""


def get_configs():
    algo_name = "iql"
    seed = 0
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
    epoch = 2000
    step_per_epoch = 1000
    eval_episodes = 10
    batch_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "algo_name": algo_name,
        "seed": seed,
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


def train(env_name, dataset_path, log_dir):
    configs = get_configs()
    # create env and dataset
    env = gym.make(env_name)
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
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    torch.backends.cudnn.deterministic = True
    env.seed(configs["seed"])

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
    # log_dirs = make_log_dirs(env_name, configs["algo_name"], configs["seed"], configs)
    log_dirs = log_dir
    if not os.path.exists(log_dirs):
        os.makedirs(log_dirs)
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard",
    }
    logger = Logger(log_dirs, output_config)
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
