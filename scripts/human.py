import os
import ast
import numpy as np
import itertools

# === human-lire.txt 기반 ===
lire_path = "scripts/human-lire.txt"
bucket_dict = {}

with open(lire_path, "r") as file:
    for line in file:
        if not line.strip():
            continue
        b_idx_str, list_str = line.strip().split(" ", 1)
        b_idx = int(b_idx_str)
        traj_list = ast.literal_eval(list_str.strip())
        bucket_dict[b_idx] = traj_list

# 1. 모든 trajectory (start, end), 그리고 그에 대응하는 bucket index를 저장
all_trajs = []  # (start, end, bucket_index)

for b_idx, traj_list in bucket_dict.items():
    for start in traj_list:
        all_trajs.append(((start, start + 25), b_idx))

# 2. 모든 nC2 쌍에 대해 bucket index 비교로 mu 생성
lire_data = []

for (s0, b0), (s1, b1) in itertools.combinations(all_trajs, 2):
    if b0 == b1:
        mu = 0.5
    elif b0 < b1:
        mu = 0.0
    else:
        mu = 1.0

    lire_data.append((s0, s1, mu))
    lire_data.append((s1, s0, 1.0 - mu))  # 반대 쌍도 추가

# 3. 저장
lire_array = np.array(
    lire_data, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f4")]
)

save_dir = "pair/HUMAN/train"
os.makedirs(save_dir, exist_ok=True)
np.savez(os.path.join(save_dir, "human-lire.npz"), data=lire_array)

print(f"human-lire.npz pairs: {len(lire_array)}")
