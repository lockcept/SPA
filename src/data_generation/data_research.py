import glob
from math import comb
import os
from random import sample, shuffle
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import numpy as np
from tqdm import tqdm
from data_generation.utils import save_feedbacks_npz
from data_loading.load_data import load_dataset, load_pair
from policy_learning.change_reward_pt import change_reward_and_save_pt
from reward_learning.get_model import get_reward_model
from reward_learning.train_model import train_reward_model
from utils.path import get_pair_path, get_reward_model_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epoch = 200


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_total_reward(s, e, reward_cumsum):
    return reward_cumsum[e - 1] - (reward_cumsum[s - 1] if s > 0 else 0)


def predict_rewards(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo,
):
    dataset = load_dataset(env_name)
    obs_dim = dataset["observations"].shape[1]
    act_dim = dataset["actions"].shape[1]

    print("obs_dim:", obs_dim, "act_dim:", act_dim)
    model_path_pattern = get_reward_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo=pair_algo,
        reward_model_algo=reward_model_algo,
        reward_model_tag="*",
    )
    model_files = glob.glob(model_path_pattern)
    model_list = []

    for model_file in model_files:
        model, _ = get_reward_model(
            reward_model_algo=reward_model_algo,
            obs_dim=obs_dim,
            act_dim=act_dim,
            model_path=model_file,
            allow_existing=True,
        )
        model_list.append(model)

    def compute_rewards(model, dataset):
        num_samples = len(dataset["observations"])
        batch_size = num_samples // 20

        model_rewards = []
        model_logits = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            obs_batch = torch.tensor(
                dataset["observations"][start_idx:end_idx], dtype=torch.float32
            ).to(device)
            act_batch = torch.tensor(
                dataset["actions"][start_idx:end_idx], dtype=torch.float32
            ).to(device)

            rewards, logits = model(obs_batch, act_batch, return_logit=True)
            model_rewards.append(rewards.cpu().detach().numpy())
            model_logits.append(logits.cpu().detach().numpy())

        return np.concatenate(model_rewards, axis=0), np.concatenate(
            model_logits, axis=0
        )

    predicted_rewards = []
    predicted_logits = []

    for model in model_list:
        pred_rewards, pred_logits = compute_rewards(model, dataset)
        predicted_rewards.append(pred_rewards)
        predicted_logits.append(pred_logits)

    preds_array = np.stack(predicted_rewards, axis=0)  # [7, T]
    pred_mean = preds_array.mean(axis=0)  # [T]
    pred_std = preds_array.std(axis=0)  # [T]
    logit_std = np.array(predicted_logits).std(axis=0)  # [T]

    true_rewards = dataset["rewards"]
    result = list(zip(true_rewards, pred_mean, pred_std, logit_std))
    return result


def train_mr(
    env_name,
    exp_name,
    n=7,
    label_pair_algo="ternary-500",
    unlabel_pair_algo="ternary-10000",
    reward_model_algo="MR-exp",
    num_epoch=num_epoch,
):
    # -------------------------------
    # MR 모델 학습 (라벨: ternary-500)
    # -------------------------------
    for i in range(n):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=label_pair_algo,
            unlabel_pair_algo=unlabel_pair_algo,
            reward_model_algo=reward_model_algo,
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )


def train_mr_with_conf_filter(
    env_name,
    exp_name,
    result,
    pair_algo="ternary-500",
    unlabel_pair_algo="unlabel-100000",
    threshold=0.999,
):
    data = np.array(result)
    mean_rewards = data[:, 1].squeeze()
    mean_rewards_cum = np.cumsum(mean_rewards, dtype=np.float64)

    unlabel_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=unlabel_pair_algo,
    )

    feedbacks_to_augment = []

    for p in tqdm(unlabel_feedbacks, desc="Filtering pairs using uncertainty"):
        (s0, e0), (s1, e1), _ = p

        predicted_r0 = get_total_reward(s0, e0, mean_rewards_cum)
        predicted_r1 = get_total_reward(s1, e1, mean_rewards_cum)

        mu = sigmoid(predicted_r1 - predicted_r0)

        # mu가 threshold 이상인 경우만 confident_pairs에 추가
        if mu > threshold or mu < 1 - threshold:
            new_mu = 1 if mu > 0.5 else 0
            feedbacks_to_augment.append(((s0, e0), (s1, e1), new_mu))

    print(
        f"[{len(feedbacks_to_augment)} / {len(unlabel_feedbacks)}] confident pairs selected"
    )

    # 라벨 데이터 + 필터된 무라벨 데이터 합치기
    labeled_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=pair_algo,
    ).tolist()
    combined_pairs = labeled_feedbacks + feedbacks_to_augment

    new_pair_name = f"{pair_algo}-aug-conf"

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=new_pair_name,
        feedbacks=combined_pairs,
    )

    # MR 학습
    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=new_pair_name,
            reward_model_algo="MR-linear",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )


def divide_into_buckets(
    env_name,
    exp_name,
    result,
    unlabel_pair_algo="unlabel-100000",
    n=10000,
    min_k=10,
    max_k=20,
    use_knn=False,
):
    data = np.array(result)
    mean = data[:, 1]
    std = data[:, 2]
    var = std**2
    mean_cum = np.cumsum(mean, dtype=np.float64)
    var_cum = np.cumsum(var, dtype=np.float64)

    unlabel_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=unlabel_pair_algo,
    )

    trajectories = []

    for p in unlabel_feedbacks:
        trajectories.append(p[0])
        trajectories.append(p[1])

    trajectories = trajectories[:n]  # n개만 사용

    traj_data = []
    for s, e in trajectories:
        r = get_total_reward(s, e, mean_cum)
        v = get_total_reward(s, e, var_cum)
        std_ = np.sqrt(v)
        traj_data.append(((s, e), r, std_))

    # 클러스터링
    traj_data.sort(key=lambda x: x[1])  # reward 기준 정렬
    n = len(traj_data)

    if use_knn:
        rewards = np.array([r for _, r, _ in traj_data]).reshape(-1, 1)
        best_k = min_k
        best_score = -1
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(rewards)
            score = silhouette_score(rewards, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k

        print(f"[KMeans] Best k = {best_k} with silhouette score = {best_score:.4f}")

        # kmeans 클러스터링 적용
        final_kmeans = KMeans(n_clusters=best_k, random_state=0).fit(rewards)
        labels = final_kmeans.labels_

        # bucket 분리
        buckets = [[] for _ in range(best_k)]
        for traj, label in zip(traj_data, labels):
            buckets[label].append(traj)

        # ----------- 각 bucket을 평균 reward 기준으로 정렬 -----------
        bucket_mean_rewards = []
        for i in range(best_k):
            rewards_in_bucket = [r for (_, r, _) in buckets[i]]
            mean_r = np.mean(rewards_in_bucket) if rewards_in_bucket else float("inf")
            bucket_mean_rewards.append((i, mean_r))

        # reward 평균 기준으로 index 재정렬
        sorted_bucket_indices = [
            i for i, _ in sorted(bucket_mean_rewards, key=lambda x: x[1])
        ]
        sorted_buckets = [buckets[i] for i in sorted_bucket_indices]
        buckets = sorted_buckets  # overwrite

    else:
        # k-means 클러스터링
        k = max_k
        buckets = [traj_data[n * i // k : n * (i + 1) // k] for i in range(k)]

    return buckets

def extract_feedbacks_without_buckets(
    env_name,
    exp_name,
    result,
    label_pair_algo="ternary-500",
    unlabel_pair_algo="unlabel-100000",
    new_pair_name="aug-bucket",
    n=10000,
    m=10000,
    z=3.1,
    threshold=0.99,
    use_conf=False,
):
    pair_name = f"{label_pair_algo}-{new_pair_name}"

    save_path = get_pair_path(env_name, exp_name, pair_name, pair_name)

    # 이미 존재하는 경우 스킵
    if os.path.exists(save_path):
        print(f"Already exists: {save_path} — skipping generation.")
        return

    unlabeled_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=unlabel_pair_algo,
    )

    data = np.array(result)
    mean = data[:, 1]
    std = data[:, 2]
    var = std**2
    mean_cum = np.cumsum(mean, dtype=np.float64)
    var_cum = np.cumsum(var, dtype=np.float64)

    # trajectory list 생성
    trajectories = []
    for p in unlabeled_feedbacks:
        trajectories.append(p[0])
        trajectories.append(p[1])
    trajectories = trajectories[:n]

    traj_data = []
    for s, e in trajectories:
        r = get_total_reward(s, e, mean_cum)
        v = get_total_reward(s, e, var_cum)
        std_ = np.sqrt(v)
        traj_data.append(((s, e), r, std_))

    seen_pairs = set()
    feedbacks = []
    total = len(traj_data)

    pbar = tqdm(total=m, desc="Sampling confident feedbacks")
    while len(feedbacks) < m:
        i, j = random.sample(range(total), 2)
        if i == j:
            continue

        pair_key = (min(i, j), max(i, j))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        t0 = traj_data[i]
        t1 = traj_data[j]

        if t0[1] > t1[1]:
            t0, t1 = t1, t0
        
        (s0, r0, std0) = t0
        (s1, r1, std1) = t1

        if use_conf:
            mu = sigmoid(r1 - r0)
            if mu > threshold:
                feedbacks.append((s0, s1, 1.0))
                pbar.update(1)
            elif 1 - mu > threshold:
                feedbacks.append((s0, s1, 0.0))
                pbar.update(1)
        else:
            upper_0 = r0 + z * std0
            lower_1 = r1 - z * std1

            if upper_0 < lower_1:
                feedbacks.append((s0, s1, 1.0))
                pbar.update(1)

    pbar.close()

    labeled_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=label_pair_algo,
    ).tolist()

    feedbacks = labeled_feedbacks + feedbacks

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_name,
        feedbacks=feedbacks,
    )


def extract_feedbacks_from_buckets(
    env_name,
    exp_name,
    buckets,
    label_pair_algo="ternary-500",
    unlabel_pair_algo="unlabel-100000",
    new_pair_name="aug-bucket",
    n=10000,
    m=10000,
    z=3.1,
    threshold=0.99,
    use_conf=False,
    use_ratio=False,
):
    unlabel_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=unlabel_pair_algo,
    )
    trajectories = []
    for p in unlabel_feedbacks:
        trajectories.append(p[0])
        trajectories.append(p[1])

    trajectories = trajectories[:n]  # n개만 사용

    feedbacks_bucket = []
    k = len(buckets)

    # bucket 크기 정보
    bucket_sizes = [len(b) for b in buckets]
    pair_count = np.zeros((k, k), dtype=int)

    if use_ratio:
        # 비례 분배
        total_weight = 0
        weight_matrix = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                w = bucket_sizes[i] * bucket_sizes[j]
                weight_matrix[i, j] = w
                total_weight += w

        for i in range(k):
            for j in range(i + 1, k):
                weight = weight_matrix[i, j]
                pair_count[i, j] = int(round(m * weight / total_weight))

    else:
        # 균등 분배
        total_pairs = comb(k, 2)
        uniform_count = m // total_pairs
        for i in range(k):
            for j in range(i + 1, k):
                pair_count[i, j] = uniform_count

    feedbacks_bucket = []

    for i in range(k):
        for j in range(i + 1, k):
            trajs_i = buckets[i]
            trajs_j = buckets[j]

            local_pairs = []

            for traj_i in trajs_i:
                (s0, r0, std0) = traj_i
                upper_i = r0 + z * std0

                for traj_j in trajs_j:
                    (s1, r1, std1) = traj_j
                    lower_j = r1 - z * std1

                    if use_conf:
                        if sigmoid(r1 - r0) > threshold:
                            local_pairs.append((s0, s1, 1.0))
                        if sigmoid(r0 - r1) > threshold:
                            local_pairs.append((s0, s1, 0.0))
                    else:
                        if upper_i < lower_j:
                            local_pairs.append((s0, s1, 1.0))

            alloc = pair_count[i, j]
            if len(local_pairs) > alloc:
                feedbacks_bucket.extend(sample(local_pairs, alloc))
            elif len(local_pairs) > 0:
                feedbacks_bucket.extend(local_pairs)

    print(f"[feedbacks_bucket] Generated {len(feedbacks_bucket)} confident pairs")

    # ----------- 기존 라벨 피드백과 결합 및 저장 -----------
    label_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=label_pair_algo,
    ).tolist()
    feedbacks_bucket = label_feedbacks + feedbacks_bucket

    pair_name = f"{label_pair_algo}-{new_pair_name}"

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_name,
        feedbacks=feedbacks_bucket,
    )

    return pair_name


def train_aug_mr(
    env_name,
    exp_name,
    pair_algo="ternary-500",
    aug_pair_algo="aug-bucket",
    reward_model_algo="MR-linear",
):
    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=f"{pair_algo}-{aug_pair_algo}",
            reward_model_algo=reward_model_algo,
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )


def data_research(env_name, exp_name):
    """
    Research function for data generation and analysis.
    """

    label_pair_algo = "ternary-500"

    # train_mr(env_name=env_name, exp_name=exp_name, reward_model_algo="MR-exp")

    # train_mr(env_name=env_name, exp_name=exp_name, n=3, reward_model_algo="MR-linear")

    # train_mr(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     reward_model_algo="MR-SURF-exp"
    # )

    # train_mr(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     n=3,
    #     reward_model_algo="MR-SURF-linear",
    # )
    # train_mr(env_name=env_name, exp_name=exp_name, n=3, reward_model_algo="PT-exp", label_pair_algo=label_pair_algo, num_epoch=200)
    # change_reward_and_save_pt(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     pair_algo=label_pair_algo,
    #     is_linear=False,
    # )

    # train_mr(env_name=env_name, exp_name=exp_name, n=3, reward_model_algo="PT-exp", label_pair_algo=f"{label_pair_algo}-aug-10000-bucket-20-uncert-3.1", num_epoch=200)
    # change_reward_and_save_pt(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     pair_algo=f"{label_pair_algo}-aug-10000-bucket-20-uncert-3.1",
    #     is_linear=False,
    # )

    # train_mr(env_name=env_name, exp_name=exp_name, n=3, reward_model_algo="PT-linear", label_pair_algo=label_pair_algo, num_epoch=200)
    # change_reward_and_save_pt(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     pair_algo=label_pair_algo,
    #     is_linear=True,
    # )

    # train_mr(env_name=env_name, exp_name=exp_name, n=3, reward_model_algo="PT-linear", label_pair_algo=f"{label_pair_algo}-aug-10000-bucket-20-uncert-3.1", num_epoch=200)
    # change_reward_and_save_pt(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     pair_algo=f"{label_pair_algo}-aug-10000-bucket-20-uncert-3.1",
    #     is_linear=True,
    # )

    result = predict_rewards(
        env_name, exp_name, pair_algo=label_pair_algo, reward_model_algo="MR-exp"
    )

    use_knn = False
    use_conf = False
    use_ratio = False
    m = 10000
    mu = 0.999
    z = 3.1
    min_k = 10
    max_k = 20

    params = [  # (use_knn, use_conf, use_ratio, m, mu, z, min_k, max_k) ]
        # (False, False, False, m, mu, z, min_k, max_k),
        # (False, True, False, m, mu, z, min_k, max_k),
        # (True, False, False, m, mu, z, min_k, max_k),
        # (True, False, True, m, mu, z, min_k, max_k),
        (True, True, True, m, mu, z, min_k, max_k),
    ]

    for param in params:
        use_knn, use_conf, use_ratio, m, mu, z, min_k, max_k = param

        new_pair_name = "aug"
        new_pair_name = new_pair_name + f"-{m}"

        if use_knn:
            new_pair_name = new_pair_name + "-bucket-knn"

            if use_ratio:
                new_pair_name = new_pair_name + "-ratio"

            new_pair_name = new_pair_name + f"-{min_k}-{max_k}"
        else:
            new_pair_name = new_pair_name + f"-bucket-{max_k}"

        if use_conf:
            new_pair_name = new_pair_name + f"-conf-{mu}"
        else:
            new_pair_name = new_pair_name + f"-uncert-{z}"

        buckets = divide_into_buckets(
            env_name=env_name,
            exp_name=exp_name,
            result=result,
            unlabel_pair_algo="unlabel-100000",
            min_k=min_k,
            max_k=max_k,
            use_knn=use_knn,
        )

        extract_feedbacks_from_buckets(
            env_name=env_name,
            exp_name=exp_name,
            buckets=buckets,
            label_pair_algo=label_pair_algo,
            unlabel_pair_algo="unlabel-100000",
            new_pair_name=new_pair_name,
            m=m,
            z=z,
            threshold=mu,
            use_conf=False,
            use_ratio=use_ratio,
        )

        # train_aug_mr(
        #     env_name=env_name,
        #     exp_name=exp_name,
        #     pair_algo=label_pair_algo,
        #     aug_pair_algo=new_pair_name,
        #     reward_model_algo="MR-linear",
        # )

    params_no_bucket = [ # (use_conf, m, mu, z) ]
        # (False, m, mu, z),
        # (True, m, mu, z),
        # (True, 10000, 0.8, 3.1),
    ]

    for param in params_no_bucket:
        use_conf, m, mu, z = param

        new_pair_name = "aug"
        new_pair_name = new_pair_name + f"-{m}"

        if use_conf:
            new_pair_name = new_pair_name + f"-conf-{mu}"
        else:
            new_pair_name = new_pair_name + f"-uncert-{z}"

        extract_feedbacks_without_buckets(
            env_name=env_name,
            exp_name=exp_name,
            result=result,
            unlabel_pair_algo="unlabel-100000",
            new_pair_name=new_pair_name,
            m=m,
            z=z,
            threshold=mu,
            use_conf=use_conf,
        )

        train_aug_mr(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500",
            aug_pair_algo=new_pair_name,
            reward_model_algo="MR-linear",
        )

