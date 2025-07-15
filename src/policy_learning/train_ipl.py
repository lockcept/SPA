import random

import numpy as np
import torch
import wandb

from data_loading import get_env
from data_loading.load_data import load_dataset
from data_loading.preference_dataloader import get_dataloader
from policy_learning.actor_module import ActorProb
from policy_learning.critic_module import Critic
from policy_learning.dist_module import DiagGaussian
from policy_learning.load_dataset import qlearning_dataset
from policy_learning.logger import Logger
from policy_learning.mlp import MLP
from policy_learning.replay_buffer import ReplayBuffer
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
    preference_batch_size = 8
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
        "preference_batch_size": preference_batch_size,
        "device": device,
    }


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
    )
    wandb.run.save()


def train_ipl_policy(
    env_name,
    exp_name,
    pair_algo: str,
    reward_model_algo,
):

    """
    Train IPL policy on the given dataset
    """
    policy_dir = get_policy_model_path(
        env_name, exp_name, pair_algo, reward_model_algo
    )

    # import gym lazyly to reduce the overhead
    from policy_learning.policy_trainer import MFPolicyTrainer  # pylint: disable=C0415
    from policy_learning.ipl_policy import IPLIQLPolicy  # pylint: disable=C0415

    configs = get_configs()
    # create env and dataset
    env = get_env(env_name, is_hidden=False)

    dataset_npz = load_dataset(
        env_name=env_name,
    )
    dataset = {key: dataset_npz[key] for key in dataset_npz}
    dataset = qlearning_dataset(env, dataset=dataset)

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
    # env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # wandb
    configs.update({"seed": seed})
    configs.update({"project": "CUDA"})
    new_exp_name = "-".join(exp_name.split("-")[:-1])
    simple_pair_algo = pair_algo.replace("ternary-", "t-").replace("bucket-", "buc-")
    group = f"{simple_pair_algo}_{reward_model_algo}"
    configs.update({"group": group})
    configs.update({"name": exp_name})
    configs.update(
        {
            "env": env_name,
            "exp_name": new_exp_name,
            "pair_algo": pair_algo,
            "reward_model_algo": reward_model_algo,
        }
    )
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
        list(actor.modules()) + list(critic_q1.modules()) + list(critic_v.modules())
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
    policy = IPLIQLPolicy(
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

    preference_dataloader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=pair_algo,
        batch_size=configs["preference_batch_size"],
        shuffle=True,
        drop_last=True,
    )

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
        preference_dataloader=preference_dataloader,
        lr_scheduler=lr_scheduler,
    )

    # train
    policy_trainer.train()
