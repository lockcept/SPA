from data_generation.utils import save_feedbacks_npz
import random

def generate_and_save_unlabel_pairs(
    env_name,
    exp_name,
    pair_type,
    label_pairs,
    all_trajs,
    n=100000,
    label_n=500,
    trajectory_length=25,
):
    # 1. trajectory set 생성
    traj_set = set()
    for s, e in all_trajs:
        if e - s < trajectory_length:
            continue
        for i in range(s, e - trajectory_length + 1):
            traj_set.add((i, i + trajectory_length))

    # 2. labeled pair에 포함된 trajectory들 추출
    labeled_traj_set = set()
    for (s0, e0), (s1, e1) in label_pairs[:label_n]:
        labeled_traj_set.add((s0, e0))
        labeled_traj_set.add((s1, e1))

    # 3. 후보 trajectory 리스트
    candidate_trajs = list(traj_set - labeled_traj_set)
    random.shuffle(candidate_trajs)

    print (
        f"Unlabel pairs: {len(candidate_trajs)} candidate trajectories, {len(labeled_traj_set)} labeled trajectories"
    )

    # 4. 순서대로 pair 만들기
    feedbacks = []
    max_pairs = min(n, len(candidate_trajs)) // 2
    for i in range(max_pairs):
        traj1 = candidate_trajs[2 * i]
        traj2 = candidate_trajs[2 * i + 1]
        feedbacks.append((traj1, traj2, 0.5))

    # 5. 저장
    save_feedbacks_npz(
        env_name=env_name,
        exp_name=exp_name,
        pair_type=pair_type,
        pair_name=f"unlabel-{n}",
        feedbacks=feedbacks,
    )