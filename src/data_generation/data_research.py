import csv
import itertools
import os
from random import sample, shuffle
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

def test1(env_name, exp_name):
    """
    Research function for data generation and analysis.
    """
    data = mr_dropout_test(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="ternary-500",
        device=device,
    )

    save_dir = "reward_eval_stats"
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "MCdropout.csv")

    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "env_name",
                    "exp_name",
                    "pair_algo",
                    "correct_ratio",
                    "incorrect_ratio",
                    "correct_reward_mean",
                    "incorrect_reward_mean",
                    "correct_uncertainty_mean",
                    "incorrect_uncertainty_mean",
                    "correct_uncertainty_std",
                    "incorrect_uncertainty_std",
                ]
            )

    correct = []
    incorrect = []

    correct_reward = []
    incorrect_reward = []

    for datum in data:
        (
            (s0, e0),
            (s1, e1),
            predicted_mu,
            mu,
            uncertainty_0,
            uncertainty_1,
            predicted_sum_of_rewards_0,
            predicted_sum_of_rewards_1,
        ) = datum
        
        if categorize_mu(predicted_mu) == 0.5:
            continue

        if categorize_mu(predicted_mu) == categorize_mu(mu):
            correct.append(uncertainty_0)
            correct.append(uncertainty_1)
            correct_reward.append(predicted_sum_of_rewards_0)
            correct_reward.append(predicted_sum_of_rewards_1)
        else:
            incorrect.append(uncertainty_0)
            incorrect.append(uncertainty_1)
            incorrect_reward.append(predicted_sum_of_rewards_0)
            incorrect_reward.append(predicted_sum_of_rewards_1)

    print("Correct predictions:", len(correct) / 2, np.mean(correct))
    print("Incorrect predictions:", len(incorrect) / 2, np.mean(incorrect))

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                env_name,
                exp_name,
                "ternary-500",
                round(len(correct) / (len(correct) + len(incorrect)), 4),
                round(len(incorrect) / (len(correct) + len(incorrect)), 4),
                round(np.mean(correct_reward), 4),
                round(np.mean(incorrect_reward), 4),
                round(np.mean(correct), 4),
                round(np.mean(incorrect), 4),
                round(np.std(correct), 4),
                round(np.std(incorrect), 4),
            ]
        )


def test2(env_name, exp_name):
    """
    Research function for uncertainty-based top-k accuracy evaluation.
    Selects top-K high uncertainty pairs and evaluates correctness ratio.
    """
    data = mr_dropout_test(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="ternary-500",
        device=device,
    )



    filtered_data = [d for d in data if categorize_mu(d[2]) != 0.5]

    print("data length:", len(data))
    print("Filtered data length:", len(filtered_data))


    # 필터 기준: uncertainty 합
    ranked_data = sorted(
        filtered_data,
        key=lambda d: (d[4] + d[5]),  # predicted_std_0 + predicted_std_1
        reverse=False
    )

    # Top-k 정확도 계산
    top_k_values = [100, 1000, 10000, 100000]
    # top_k_values = [100]
    results = []

    for k in top_k_values:
        top_k = ranked_data[:k]

        correct_count = 0
        total_count = 0

        for datum in top_k:
            (s0, e0), (s1, e1), predicted_mu, mu, _, _, predicted_sum_of_rewards_0, predicted_sum_of_rewards_1 = datum
            if categorize_mu(predicted_mu) == mu:
                correct_count += 1
            total_count += 1

            # print(predicted_sum_of_rewards_0, predicted_sum_of_rewards_1, np.sum(dataset["rewards"][s0:e0]), np.sum(dataset["rewards"][s1:e1]), predicted_mu, mu)

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"Top-{k} accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        results.append((k, accuracy, correct_count, total_count))

    # CSV 저장
    save_dir = "reward_eval_stats"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "MCdropout_topk.csv")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["env_name", "exp_name", "pair_algo", "top_k", "accuracy", "correct", "total"])

        for k, acc, correct, total in results:
            writer.writerow([env_name, exp_name, "ternary-500", k, acc, correct, total])

def test3(env_name, exp_name):
    data = mr_dropout_test(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="ternary-500",
        device=device,
    )

    trajectories = []
    for i in range(10000):
        traj = data[i][0]
        predicted_reward = data[i][6]
        uncertainty = data[i][4]
        trajectories.append((traj, predicted_reward, uncertainty))


    dataset = load_dataset(env_name)
    reward_cumsum = np.cumsum(dataset["rewards"], axis=0, dtype=np.float64)

    top_100_certainty = sorted(
        trajectories,
        key=lambda x: x[2],  # uncertainty
        reverse=True
    )[:100]

    # 100C2 쌍 만들기
    trajectory_pairs = list(itertools.combinations(top_100_certainty, 2))

    correct = 0
    total = 0

    for (traj0, pred_r0, _), (traj1, pred_r1, _) in trajectory_pairs:
        total += 1

        # 예측 label 계산
        pred_mu = (pred_r1 + 1e-6) / (pred_r0 + pred_r1 + 2e-6)
        pred_label = int(pred_mu > 0.5)

        # 실제 label 계산
        s0, e0 = traj0
        s1, e1 = traj1
        sum_r0 = reward_cumsum[e0 - 1] - (reward_cumsum[s0 - 1] if s0 > 0 else 0)
        sum_r1 = reward_cumsum[e1 - 1] - (reward_cumsum[s1 - 1] if s1 > 0 else 0)
        true_label = int(sum_r1 > sum_r0)

        if pred_label == true_label:
            correct += 1


    # CSV 저장
    save_dir = "reward_eval_stats"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "MCdropout_topk.csv")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["env_name", "exp_name", "pair_algo", "top_k", "accuracy", "correct", "total"])

        writer.writerow([env_name, exp_name, "ternary-500", 103, round(correct/total,4), correct, total])
    

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

def calculate_from_mr(env_name, exp_name):
    # -------------------------------
    # MR-exp로 학습된 모델 7개 로드
    # -------------------------------
    models = []
    dataset = load_dataset(env_name)

    for i in range(7):
        model_path = get_reward_model_path(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo="ternary-500",
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

    # -------------------------------
    # 예측값 정규화 및 통계 계산
    # -------------------------------
    def normalize_preds(preds, method="z"):
        preds = preds.astype(np.float64)
        if method == "z":
            mean = preds.mean()
            std = preds.std() + 1e-8
            return (preds - mean) / std
        elif method == "minmax":
            minv = preds.min()
            maxv = preds.max() + 1e-8
            print(minv, maxv)
            return (preds - minv) / (maxv - minv)
        elif method == "none":
            return preds
        else:
            raise ValueError("Unknown normalization method")

    normalized_preds = []
    for preds in predicted_rewards:
        norm_preds = normalize_preds(preds, method="none")
        normalized_preds.append(norm_preds)

    normalized_preds_array = np.stack(normalized_preds, axis=0)  # [7, T]
    pred_mean = normalized_preds_array.mean(axis=0)              # [T]
    pred_std = normalized_preds_array.std(axis=0)                # [T]
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

def augment_with_bucket(env_name, exp_name, result, k=20, select_ratio=0.005, z=10):
    """
    각 bucket에서 uncertainty 가장 낮은 top N% trajectory 선택 후,
             모든 bucket 쌍에 대해 pair 생성
    각 bucket 쌍에서 신뢰구간 겹치지 않는 pair를 top_k 개 만큼 생성
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

    # ----------- 4. 신뢰구간 미겹침 조건으로 top_k_per_pair 개씩 채우기 -----------
    feedbacks_bucket_2 = []
    n = len(trajectories)
    num_per_bucket_pair = int((n * select_ratio // k) ** 2)

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
                feedbacks_bucket_2.extend(sample(local_pairs, num_per_bucket_pair))
            elif len(local_pairs) > 0:
                feedbacks_bucket_2.extend(local_pairs)

    print(f"[feedbacks_bucket_2] Generated {len(feedbacks_bucket_2)} confident pairs")

    # ----------- 5. 저장 -----------
    label_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo="ternary-500",
    ).tolist()

    feedbacks_bucket_1 = label_feedbacks + feedbacks_bucket_1
    feedbacks_bucket_2 = label_feedbacks + feedbacks_bucket_2

    pair_name_1 = f"ternary-500-aug-bucket-1"
    pair_name_2 = f"ternary-500-aug-bucket-2"

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_name_1,
        feedbacks=feedbacks_bucket_1,
       )
    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_name_2,
        feedbacks=feedbacks_bucket_2,
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

    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_name_2,
            reward_model_algo="MR-exp",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )

def augment_with_bucket_2(env_name, exp_name, result, k=20, select_ratio=0.005, z=10):
    """
    각 bucket에서 uncertainty 가장 낮은 top N% trajectory 선택 후,
             모든 bucket 쌍에 대해 pair 생성
    각 bucket 쌍에서 신뢰구간 겹치지 않는 pair를 top_k 개 만큼 생성
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
    feedbacks_bucket_3 = []

    # 각 bucket에서 select_ratio 비율만큼 선택
    selected_per_bucket = []
    for bucket in buckets:
        num_select = max(1, int(len(bucket) * select_ratio))
        sorted_by_std = sorted(bucket, key=lambda x: x[2])
        selected_per_bucket.append(sorted_by_std[:num_select])

    # 모든 bucket 쌍 조합에 대해 pair 생성
    for i, j in itertools.combinations(range(k), 2):
        if np.abs(i - j) < 3:
            continue
        for traj_i, traj_j in itertools.product(selected_per_bucket[i], selected_per_bucket[j]):
            s0, r0, _ = traj_i
            s1, r1, _ = traj_j
            feedbacks_bucket_3.append((s0, s1, 1.0))

    print(f"[bucket_3] Generated {len(feedbacks_bucket_3)} confident pairs")

    # ----------- 4. 저장 -----------
    label_feedbacks = load_pair(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_algo="ternary-500",
    ).tolist()

    feedbacks_bucket_3 = label_feedbacks + feedbacks_bucket_3

    pair_name_3 = f"ternary-500-aug-bucket-3"

    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="train",
        pair_name=pair_name_3,
        feedbacks=feedbacks_bucket_3,
       )

    # 5. MR 학습
    for i in range(3):
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_name_3,
            reward_model_algo="MR-exp",
            reward_model_tag=f"{i:02d}",
            num_epoch=num_epoch,
        )
        train_reward_model(
            env_name=env_name,
            exp_name=exp_name,
            pair_algo=pair_name_3,
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
    train_mr_with_conf_filter(
        env_name=env_name,
        exp_name=exp_name,
        result=result,
        pair_algo="ternary-500",
        unlabel_pair_algo="ternary-10000",
        threshold=0.999,
    )
    augment_with_bucket(
        env_name=env_name,
        exp_name=exp_name,
        result=result,
        k=20,
        select_ratio=0.01,
        z=10,
    )
    augment_with_bucket_2(
        env_name=env_name,
        exp_name=exp_name,
        result=result,
        k=20,
        select_ratio=0.01,
        z=10,
    )

