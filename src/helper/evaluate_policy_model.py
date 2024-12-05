import csv
import numpy as np
import torch
from tqdm import tqdm

from offlinerlkit.modules.actor_module import ActorProb
from offlinerlkit.modules.critic_module import Critic
from offlinerlkit.modules.dist_module import DiagGaussian
from offlinerlkit.nets.mlp import MLP


from data_loading import get_env
from policy_learning import get_configs


def evaluate_policy(env_name, model_path, pair_name, reward_model_algo):
    # import gym lazyly to reduce the overhead
    from offlinerlkit.policy.model_free.iql import IQLPolicy  # pylint: disable=C0415

    configs = get_configs()
    # create env and dataset
    env = get_env(env_name)
    configs.update(
        {
            "obs_shape": env.observation_space.shape,
            "action_dim": np.prod(env.action_space.shape),
            "max_action": env.action_space.high[0],
        }
    )
    torch.manual_seed(configs["seed"])
    torch.cuda.manual_seed_all(configs["seed"])
    torch.backends.cudnn.deterministic = True
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

    actor_optim = torch.optim.Adam(actor.parameters(), lr=configs["actor_lr"])
    critic_q1_optim = torch.optim.Adam(
        critic_q1.parameters(), lr=configs["critic_q_lr"]
    )
    critic_q2_optim = torch.optim.Adam(
        critic_q2.parameters(), lr=configs["critic_q_lr"]
    )
    critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=configs["critic_v_lr"])

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

    state_dict = torch.load(
        model_path,
        map_location=configs["device"],
        weights_only=True,
    )

    policy.load_state_dict(state_dict)

    policy.eval()
    obs = env.reset(seed=0)
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length, episode_success = 0, 0, 0

    eval_episodes = 1000

    with tqdm(total=eval_episodes, desc="Evaluating Episodes") as pbar:
        while num_episodes < eval_episodes:
            action = policy.select_action(obs.reshape(1, -1), deterministic=True)
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
                pbar.update(1)
                episode_reward, episode_length, episode_success = 0, 0, 0
                obs = env.reset(seed=num_episodes)

    reward_list = [ep["episode_reward"] for ep in eval_ep_info_buffer]
    length_list = [ep["episode_length"] for ep in eval_ep_info_buffer]
    success_list = [ep["episode_success"] for ep in eval_ep_info_buffer]

    reward_mean = np.mean(reward_list)
    length_mean = np.mean(length_list)
    success_mean = np.mean(np.array(success_list) > 0)

    log_path = "log/main_evaluate_policy.csv"

    if not model_path.endswith("best_policy.pth"):
        return

    with open(log_path, "a", encoding="utf-8", newline="") as log_file:
        writer = csv.writer(log_file)

        if log_file.tell() == 0:
            writer.writerow(
                [
                    "EnvName",
                    "PairName",
                    "RewardModelAlgo",
                    "RewardMean",
                    "LengthMean",
                    "SuccessMean",
                ]
            )

        writer.writerow(
            [
                env_name,
                pair_name,
                reward_model_algo,
                f"{reward_mean:.3f}",
                f"{length_mean:.2f}",
                f"{success_mean:.6f}",
            ]
        )


def evaluate_best_and_last_policy(
    env_name, pair_name, reward_model_algo, policy_model_dir
):
    """
    evaluate best and last policy
    """
    evaluate_policy(
        env_name=env_name,
        model_path=f"{policy_model_dir}/model/best_policy.pth",
        pair_name=pair_name,
        reward_model_algo=reward_model_algo,
    )

    # evaluate_policy(
    #     env_name=env_name,
    #     model_path=f"{policy_model_dir}/model/last_policy.pth",
    #     pair_name=pair_name,
    #     reward_model_algo=reward_model_algo,
    # )
