import os
import sys
import numpy as np
import torch
from offlinerlkit.modules.actor_module import ActorProb
from offlinerlkit.modules.critic_module import Critic
from offlinerlkit.modules.dist_module import DiagGaussian
from offlinerlkit.nets.mlp import MLP
from offlinerlkit.policy.model_free.iql import IQLPolicy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from data_loading.load_data import get_env
from policy_learning.iql import get_configs


def evaluate_policy(env_name, model_path):
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

    # 저장된 가중치 로드
    state_dict = torch.load(
        model_path, map_location=configs["device"], weights_only=True
    )

    # 모델에 가중치 적용
    policy.load_state_dict(state_dict)

    print("Model successfully loaded!")

    policy.eval()
    obs = env.reset()
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length = 0, 0

    while num_episodes < configs["eval_episodes"]:
        action = policy.select_action(obs.reshape(1, -1), deterministic=True)
        next_obs, reward, terminal, _ = env.step(action.flatten())
        episode_reward += reward
        episode_length += 1

        obs = next_obs

        if terminal:
            eval_ep_info_buffer.append(
                {"episode_reward": episode_reward, "episode_length": episode_length}
            )
            num_episodes += 1
            episode_reward, episode_length = 0, 0
            obs = env.reset()

    print(
        model_path,
        "\n",
        "Average episode reward: ",
        np.mean([ep["episode_reward"] for ep in eval_ep_info_buffer]) * 100,
        "Average episode length: ",
        np.mean([ep["episode_length"] for ep in eval_ep_info_buffer]),
    )
