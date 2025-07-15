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
from policy_learning.dppo_policy import DPPOPolicy
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
    dropout_rate = None
    step_per_epoch = 1000
    batch_size = 256
    preference_batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M_epochs = 200  # predictor training epochs
    N_epochs = 300  # policy training epochs

    return {
        "hidden_dims": hidden_dims,
        "actor_lr": actor_lr,
        "dropout_rate": dropout_rate,
        "step_per_epoch": step_per_epoch,
        "batch_size": batch_size,
        "preference_batch_size": preference_batch_size,
        "device": device,
        "M_epochs": M_epochs,
        "N_epochs": N_epochs,
    }


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
    )
    wandb.run.save()


def evaluate_policy_detailed(policy_trainer, env, n_episodes=5):
    eval_info = {
        "eval/episode_reward": [],
        "eval/episode_length": [],
        "eval/episode_success": [],
    }
    obs = env.reset()
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length, episode_success = 0, 0, 0

    while num_episodes < n_episodes:
        action = policy_trainer.policy.select_action(
            obs.reshape(1, -1), deterministic=True
        )
        next_obs, reward, terminal, info = env.step(action.flatten())
        episode_reward += reward
        episode_length += 1
        if "success" in info:
            episode_success += info["success"]

        obs = next_obs
        if terminal:
            eval_ep_info_buffer.append(
                {
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode_success": episode_success,
                }
            )
            num_episodes += 1
            episode_reward, episode_length, episode_success = 0, 0, 0
            obs = env.reset()

    eval_info["eval/episode_reward"] = [
        ep["episode_reward"] for ep in eval_ep_info_buffer
    ]
    eval_info["eval/episode_length"] = [
        ep["episode_length"] for ep in eval_ep_info_buffer
    ]
    eval_info["eval/episode_success"] = [
        ep["episode_success"] for ep in eval_ep_info_buffer
    ]
    return eval_info


def train_dppo_policy(
    env_name,
    exp_name,
    pair_algo: str,
    reward_model_algo,
):
    """
    Train DPPO policy on the given dataset
    """
    policy_dir = get_policy_model_path(env_name, exp_name, pair_algo, reward_model_algo)

    # import gym lazyly to reduce the overhead
    from policy_learning.policy_trainer import MFPolicyTrainer  # pylint: disable=C0415

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
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=configs["action_dim"],
        unbounded=False,
        conditioned_sigma=False,
        max_mu=configs["max_action"],
    )
    actor = ActorProb(actor_backbone, dist, configs["device"])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=configs["actor_lr"])

    lr_scheduler = None

    # create DPPO policy
    policy = DPPOPolicy(
        actor,
        actor_optim,
        action_space=env.action_space,
    )

    # create Preference Predictor model
    pref_input_dim = np.prod(configs["obs_shape"]) + configs["action_dim"]
    from policy_learning.dppo_policy import (
        PreferencePredictor,
        train_preference_predictor,
    )

    pref_predictor = PreferencePredictor(pref_input_dim).to(configs["device"])
    pref_opt = torch.optim.Adam(pref_predictor.parameters(), lr=configs["actor_lr"])

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
        step_per_epoch=configs["step_per_epoch"],
        batch_size=configs["batch_size"],
        preference_dataloader=preference_dataloader,
        lr_scheduler=lr_scheduler,
    )

    # Phase 1: Train preference predictor only
    for epoch_idx in range(configs["M_epochs"]):
        for pref_batch in preference_dataloader:
            pref_loss_dict = train_preference_predictor(
                pref_predictor,
                pref_opt,
                pref_batch,
                buffer=buffer,
                smoothness_weight=1.0,
            )
            # logkv_mean for each batch metric
            for key, value in pref_loss_dict.items():
                logger.logkv_mean(key, value)
        # dump averaged metrics once per epoch
        logger.set_timestep(epoch_idx)
        logger.dumpkvs()
        wandb.log(pref_loss_dict, step=epoch_idx)

    # Phase 2: Fix predictor, train policy
    for epoch_idx in range(configs["N_epochs"]):
        for _ in range(configs["step_per_epoch"]):
            unlabeled_batch = buffer.sample_overlapping_segments(
                configs["preference_batch_size"]
            )
            policy_loss_dict = policy.learn_with_predictor(
                pref_predictor,
                unlabeled_batch,
                lambda_reg=0.5,
            )
            for key, value in policy_loss_dict.items():
                logger.logkv_mean(key, value)
        logger.set_timestep(epoch_idx)
        logger.dumpkvs()
        wandb.log(policy_loss_dict, step=epoch_idx)

        # detailed evaluation
        eval_info = evaluate_policy_detailed(policy_trainer, env, n_episodes=5)
        normalized_rewards = [
            env.get_normalized_score(r) for r in eval_info["eval/episode_reward"]
        ]
        norm_ep_rew_mean = np.mean(normalized_rewards)
        norm_ep_rew_std = np.std(normalized_rewards)
        ep_length_mean = np.mean(eval_info["eval/episode_length"])
        ep_length_std = np.std(eval_info["eval/episode_length"])
        ep_success_mean = np.mean(np.array(eval_info["eval/episode_success"]) > 0)

        wandb_log = {
            "eval/normalized_episode_reward": norm_ep_rew_mean,
            "eval/normalized_episode_reward_std": norm_ep_rew_std,
            "eval/episode_length": ep_length_mean,
            "eval/episode_length_std": ep_length_std,
            "eval/episode_reward": np.mean(eval_info["eval/episode_reward"]),
            "eval/episode_success": ep_success_mean,
        }
        # also log to logger
        for key, value in wandb_log.items():
            logger.logkv(key, value)
        logger.set_timestep(epoch_idx)
        logger.dumpkvs()
        wandb.log(wandb_log, step=epoch_idx)
