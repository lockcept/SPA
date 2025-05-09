import csv
import itertools
import os
from random import sample, shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import numpy as np
from tqdm import tqdm
from data_generation.picker.mr_dropout import mr_dropout_test
from data_generation.utils import generate_pairs_from_indices, save_feedbacks_npz
from data_loading.load_data import load_dataset, load_pair
from reward_learning.get_model import get_reward_model
from reward_learning.train_model import train_reward_model
from utils.path import get_reward_model_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epoch = 200

def categorize_mu(mu_value):
    if mu_value < 1 / 3:
        return 0.0
    elif mu_value > 2 / 3:
        return 1.0
    else:
        return 0.5

def train_mr_and_surf(env_name, exp_name):
    # -------------------------------
    # 1. MR-exp 모델 7개 학습 (라벨: ternary-500)
    # -------------------------------
    for i in range(7):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500",
            reward_model_algo="MR-exp",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )
    
    for i in range(7):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500",
            reward_model_algo="MR-linear",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )

    # -------------------------------
    # 2. MR-SURF 모델 7개 학습 (라벨: ternary-500, 언라벨: ternary-10000)
    # -------------------------------
    for i in range(7):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500",
            unlabel_pair_algo="ternary-10000",
            reward_model_algo="MR-SURF-exp",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500",
            unlabel_pair_algo="ternary-10000",
            reward_model_algo="MR-SURF-linear",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )

def calculate_from_mr(env_name, exp_name, pair_algo="ternary-500"):
    # -------------------------------
    # MR-exp로 학습된 모델 7개 로드
    # -------------------------------
    models = []
    dataset = load_dataset(env_name)

    for i in range(7):
        model_path = get_reward_model_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_algo,
            reward_model_algo="MR-exp",
            reward_model_tag=f"{i:02d}",
        )
        model, _ = get_reward_model(
            reward_model_algo="MR-exp",
            model_path=model_path,
            allow_existing=True,
            obs_dim=dataset["observations"].shape[1],
            act_dim=dataset["actions"].shape[1],
        )
        models.append(model)

    # -------------------------------
    # 전체 dataset에서 예측 리워드 계산
    # -------------------------------
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

        return np.concatenate(model_rewards, axis=0), np.concatenate(model_logits, axis=0)

    predicted_rewards = []
    predicted_logits = []

    for model in models:
        pred_rewards, pred_logits = compute_rewards(model, dataset)
        predicted_rewards.append(pred_rewards)
        predicted_logits.append(pred_logits)

    preds_array = np.stack(predicted_rewards, axis=0)  # [7, T]
    pred_mean = preds_array.mean(axis=0)              # [T]
    pred_std = preds_array.std(axis=0)                # [T]
    logit_std = np.array(predicted_logits).std(axis=0)           # [T]

    # -------------------------------
    # 결과 반환: (true_reward, mean, std, logit_std)
    # -------------------------------
    true_rewards = dataset["rewards"]
    result = list(zip(true_rewards, pred_mean, pred_std, logit_std))
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_mr_with_conf_filter(
    env_name,
    exp_name,
    result,  # [(true, mean, std, logit_std)]
    pair_algo="ternary-500",
    unlabel_pair_algo="ternary-10000",
    threshold=0.999,
):
    # 1. 누적 보상 및 분산 계산
    data = np.array(result)
    mean_rewards = data[:, 1].squeeze()

    mean_rewards_cum = np.cumsum(mean_rewards, dtype=np.float64)

    # 2. 무라벨 pair 불러오기
    unlabel_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=unlabel_pair_algo,
    )

    feedbacks_to_augment = []


    for p in tqdm(unlabel_feedbacks, desc="Filtering pairs using uncertainty"):
        (s0, e0), (s1, e1), _ = p

        # 예측 reward 합
        r0 = mean_rewards_cum[e0 - 1] - (mean_rewards_cum[s0 - 1] if s0 > 0 else 0)
        r1 = mean_rewards_cum[e1 - 1] - (mean_rewards_cum[s1 - 1] if s1 > 0 else 0)

        mu = sigmoid(r1 - r0)

        # mu가 threshold 이상인 경우만 confident_pairs에 추가
        if mu > threshold or mu < 1 - threshold:
            feedbacks_to_augment.append(((s0,e0), (s1,e1), categorize_mu(mu)))

    print(f"[{len(feedbacks_to_augment)} / {len(unlabel_feedbacks)}] confident pairs selected")

    # 3. 라벨 데이터 + 필터된 무라벨 데이터 합치기
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


    # 4. MR 학습
    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=new_pair_name,
            reward_model_algo="MR-exp",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )

def augment_with_bucket_uniform(env_name, exp_name, result, k=20, select_ratio=0.005):
    """
    각 bucket에서 uncertainty 가장 낮은 top N% trajectory 선택 후,
             모든 bucket 쌍에 대해 pair 생성
    """

    # ----------- 1. 기본 정보 세팅 (trajectory 정리) -----------
    data = np.array(result)
    mean = data[:, 1]
    std = data[:, 2]
    var = std**2
    mean_cum = np.cumsum(mean, dtype=np.float64)
    var_cum = np.cumsum(var, dtype=np.float64)

    feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo="ternary-10000",
    )
    trajectories = []

    for p in feedbacks:
        trajectories.append(p[0])
        trajectories.append(p[1])

    trajs = []
    for (s, e) in trajectories:
        r = mean_cum[e - 1] - (mean_cum[s - 1] if s > 0 else 0)
        v = var_cum[e - 1] - (var_cum[s - 1] if s > 0 else 0)
        std_ = np.sqrt(v)
        trajs.append(((s, e), r, std_))

    # ----------- 2. 버킷 나누기 -----------
    trajs.sort(key=lambda x: x[1])  # reward 기준 정렬
    n = len(trajs)
    buckets = [trajs[n * i // k : n * (i + 1) // k] for i in range(k)]

    # ----------- 3. Step 3: std 기준 top N% 선택 후 전체 조합 -----------
    feedbacks_bucket_1 = []

    # 각 bucket에서 select_ratio 비율만큼 선택
    selected_per_bucket = []
    for bucket in buckets:
        num_select = max(1, int(len(bucket) * select_ratio))
        sorted_by_std = sorted(bucket, key=lambda x: x[2])
        selected_per_bucket.append(sorted_by_std[:num_select])

    # 모든 bucket 쌍 조합에 대해 pair 생성
    for i, j in itertools.combinations(range(k), 2):
        for traj_i, traj_j in itertools.product(selected_per_bucket[i], selected_per_bucket[j]):
            s0, r0, _ = traj_i
            s1, r1, _ = traj_j
            feedbacks_bucket_1.append((s0, s1, 1.0))

    print(f"[bucket_1] Generated {len(feedbacks_bucket_1)} confident pairs")

    # ----------- 5. 저장 -----------
    label_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo="ternary-500",
    ).tolist()

    feedbacks_bucket_1 = label_feedbacks + feedbacks_bucket_1

    pair_name_1 = f"ternary-500-aug-bucket-1"

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_name_1,
        feedbacks=feedbacks_bucket_1,
       )
    # 6. MR 학습
    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_name_1,
            reward_model_algo="MR-exp",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )

def augment_with_bucket(env_name, exp_name, result, pair_algo="ternary-500", n=10000, k=20, num_per_bucket_pair=100, z=10):
    """
    각 bucket 쌍에서 신뢰구간 겹치지 않는 pair를 top_k 개 만큼 생성
    """

    # ----------- 1. 기본 정보 세팅 (trajectory 정리) -----------
    data = np.array(result)
    mean = data[:, 1]
    std = data[:, 2]
    var = std**2
    mean_cum = np.cumsum(mean, dtype=np.float64)
    var_cum = np.cumsum(var, dtype=np.float64)

    if n == 10000:
        feedbacks = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="ternary-10000",
        )
    elif n == 1000:
        feedbacks = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="ternary-1000",
        )
    elif n == 50000:
        feedbacks = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="ternary-100000",
        )[:50000]

    trajectories = []

    for p in feedbacks:
        trajectories.append(p[0])
        trajectories.append(p[1])

    trajs = []
    for (s, e) in trajectories:
        r = mean_cum[e - 1] - (mean_cum[s - 1] if s > 0 else 0)
        v = var_cum[e - 1] - (var_cum[s - 1] if s > 0 else 0)
        std_ = np.sqrt(v)
        trajs.append(((s, e), r, std_))

    # ----------- 2. 버킷 나누기 -----------
    trajs.sort(key=lambda x: x[1])  # reward 기준 정렬
    n = len(trajs)
    buckets = [trajs[n * i // k : n * (i + 1) // k] for i in range(k)]

    # ----------- 3. Step 3 -----------
    feedbacks_bucket = []
    n = len(trajectories)

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

                    if upper_i < lower_j:
                        local_pairs.append((s0, s1, 1.0))  # 신뢰 충분히 확보된 pair

            # 무작위로 top_k_per_pair만 선택
            if len(local_pairs) > num_per_bucket_pair:
                feedbacks_bucket.extend(sample(local_pairs, num_per_bucket_pair))
            elif len(local_pairs) > 0:
                feedbacks_bucket.extend(local_pairs)

    print(f"[feedbacks_bucket] Generated {len(feedbacks_bucket)} confident pairs")

    # ----------- 4. 저장 -----------
    label_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=pair_algo,
    ).tolist()
    feedbacks_bucket = label_feedbacks + feedbacks_bucket

    pair_name = f"{pair_algo}-aug-bucket-4-{num_per_bucket_pair}"
    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_name,
        feedbacks=feedbacks_bucket,
    )

    # 5. MR 학습
    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_name,
            reward_model_algo="MR-linear",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )

def augment_with_bucket_conf(env_name, exp_name, result, pair_algo="ternary-500", n=10000, k=20, num_per_bucket_pair=100, mu=0.99):
    """
    각 bucket 쌍에서 신뢰구간 겹치지 않는 pair를 top_k 개 만큼 생성
    """

    # ----------- 1. 기본 정보 세팅 (trajectory 정리) -----------
    data = np.array(result)
    mean = data[:, 1]
    std = data[:, 2]
    var = std**2
    mean_cum = np.cumsum(mean, dtype=np.float64)
    var_cum = np.cumsum(var, dtype=np.float64)

    if n == 10000:
        feedbacks = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="ternary-10000",
        )
    elif n == 1000:
        feedbacks = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="ternary-1000",
        )
    elif n == 50000:
        feedbacks = load_pair(
            env_name=env_name,
            exp_name=exp_name,
            pair_type="train",
            pair_algo="ternary-100000",
        )[:50000]

    trajectories = []

    for p in feedbacks:
        trajectories.append(p[0])
        trajectories.append(p[1])

    trajs = []
    for (s, e) in trajectories:
        r = mean_cum[e - 1] - (mean_cum[s - 1] if s > 0 else 0)
        v = var_cum[e - 1] - (var_cum[s - 1] if s > 0 else 0)
        std_ = np.sqrt(v)
        trajs.append(((s, e), r, std_))

    # ----------- 2. 버킷 나누기 -----------
    trajs.sort(key=lambda x: x[1])  # reward 기준 정렬
    n = len(trajs)
    buckets = [trajs[n * i // k : n * (i + 1) // k] for i in range(k)]

    # ----------- 3. Step 3 -----------
    feedbacks_bucket = []
    n = len(trajectories)

    for i in range(k):
        for j in range(i + 1, k):
            trajs_i = buckets[i]
            trajs_j = buckets[j]

            local_pairs = []

            for traj_i in trajs_i:
                (s0, r0, std0) = traj_i
                for traj_j in trajs_j:
                    (s1, r1, std1) = traj_j

                    if sigmoid(r1 - r0) > mu:
                        local_pairs.append((s0, s1, 1.0))  
                    if sigmoid(r0 - r1) > mu:
                        local_pairs.append((s0, s1, 0.0))

            # 무작위로 top_k_per_pair만 선택
            if len(local_pairs) > num_per_bucket_pair:
                feedbacks_bucket.extend(sample(local_pairs, num_per_bucket_pair))
            elif len(local_pairs) > 0:
                feedbacks_bucket.extend(local_pairs)

    print(f"[feedbacks_bucket] Generated {len(feedbacks_bucket)} confident pairs")

    # ----------- 4. 저장 -----------
    label_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo=pair_algo,
    ).tolist()
    feedbacks_bucket = label_feedbacks + feedbacks_bucket

    pair_name = f"{pair_algo}-aug-bucket-4-conf"
    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_name,
        feedbacks=feedbacks_bucket,
    )

    # 5. MR 학습
    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_name,
            reward_model_algo="MR-linear",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )

def augment_with_bucket_knn(env_name, exp_name, result, n=10000, min_k=10, max_k=20, num_per_bucket_pair=100, z=10):
    """
    reward 기반 클러스터링을 통해 trajectory를 bucket으로 나누고,
    각 bucket 쌍에서 신뢰구간 겹치지 않는 pair를 top_k 개 만큼 생성
    """
    # ----------- 1. 기본 정보 세팅 (trajectory 정리) -----------
    data = np.array(result)
    mean = data[:, 1]
    std = data[:, 2]
    var = std**2
    mean_cum = np.cumsum(mean, dtype=np.float64)
    var_cum = np.cumsum(var, dtype=np.float64)

    if n == 10000:
        feedbacks = load_pair(env_name, exp_name, "train", "ternary-10000")
    elif n == 1000:
        feedbacks = load_pair(env_name, exp_name, "train", "ternary-1000")
    elif n == 50000:
        feedbacks = load_pair(env_name, exp_name, "train", "ternary-100000")[:50000]

    trajectories = []
    for p in feedbacks:
        trajectories.append(p[0])
        trajectories.append(p[1])

    trajs = []
    for (s, e) in trajectories:
        r = mean_cum[e - 1] - (mean_cum[s - 1] if s > 0 else 0)
        v = var_cum[e - 1] - (var_cum[s - 1] if s > 0 else 0)
        std_ = np.sqrt(v)
        trajs.append(((s, e), r, std_))

    # ----------- 2. 최적의 k 선택 및 클러스터링 ----------
    rewards = np.array([r for _, r, _ in trajs]).reshape(-1, 1)
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
    for traj, label in zip(trajs, labels):
        buckets[label].append(traj)

    # ----------- 각 bucket을 평균 reward 기준으로 정렬 -----------
    bucket_mean_rewards = []
    for i in range(best_k):
        rewards_in_bucket = [r for (_, r, _) in buckets[i]]
        mean_r = np.mean(rewards_in_bucket) if rewards_in_bucket else float('inf')
        bucket_mean_rewards.append((i, mean_r))

    # reward 평균 기준으로 index 재정렬
    sorted_bucket_indices = [i for i, _ in sorted(bucket_mean_rewards, key=lambda x: x[1])]
    sorted_buckets = [buckets[i] for i in sorted_bucket_indices]
    buckets = sorted_buckets  # overwrite

    # ----------- 3. 버킷 간 confident pair 생성 -----------
    feedbacks_bucket = []
    for i in range(best_k):
        for j in range(i + 1, best_k):
            trajs_i = buckets[i]
            trajs_j = buckets[j]
            local_pairs = []
            for traj_i in trajs_i:
                (s0, r0, std0) = traj_i
                upper_i = r0 + z * std0
                for traj_j in trajs_j:
                    (s1, r1, std1) = traj_j
                    lower_j = r1 - z * std1
                    if upper_i < lower_j:
                        local_pairs.append((s0, s1, 1.0))  # 신뢰 높은 pair
            if len(local_pairs) > num_per_bucket_pair:
                feedbacks_bucket.extend(sample(local_pairs, num_per_bucket_pair))
            elif len(local_pairs) > 0:
                feedbacks_bucket.extend(local_pairs)

    print(f"[feedbacks_bucket] Generated {len(feedbacks_bucket)} confident pairs")

    # ----------- 4. 기존 라벨 피드백과 결합 및 저장 -----------
    label_feedbacks = load_pair(env_name, exp_name, "train", "ternary-500").tolist()
    feedbacks_bucket = label_feedbacks + feedbacks_bucket

    pair_name = f"ternary-500-aug-bucket-knn"
    save_feedbacks_npz(env_name, exp_name, "train", pair_name, feedbacks_bucket)

    # ----------- 5. MR 모델 학습 -----------
    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_name,
            reward_model_algo="MR-linear",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )



    

def data_research(env_name, exp_name):
    """
    Research function for data generation and analysis.
    """
    # train_mr_and_surf(env_name, exp_name)
    result = calculate_from_mr(env_name, exp_name)
    # train_mr_with_conf_filter(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     result=result,
    #     pair_algo="ternary-500",
    #     unlabel_pair_algo="ternary-10000",
    #     threshold=0.999,
    # )
    # augment_with_bucket_uniform(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     result=result,
    #     k=20,
    #     select_ratio=0.01,
    # )
    # augment_with_bucket(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     result=result,
    #     n=1000,
    #     k=20,
    #     num_per_bucket_pair=20,
    #     z=3.1,
    # )
    # augment_with_bucket_knn(
    #     env_name=env_name,
    #     exp_name=exp_name,
    #     result=result,
    #     n=10000,
    #     min_k=10,
    #     max_k=20,
    #     num_per_bucket_pair=100,
    #     z=3.1,
    # )
    augment_with_bucket_conf(
        env_name=env_name,
        exp_name=exp_name,
        result=result,
        n=10000,
        k=20,
        num_per_bucket_pair=100,
        mu=0.99,
    )
