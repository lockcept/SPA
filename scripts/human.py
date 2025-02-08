import numpy as np

file_path = "scripts/human.txt" 
data_list = []

with open(file_path, "r") as file:
    for line in file:
        s0_val, s1_val, mu_val = map(int, line.strip().split())

        s0 = (s0_val, s0_val + 25)
        s1 = (s1_val, s1_val + 25)

        mu_map = {1: 0.0, 2: 0.5, 3: 1.0}
        mu = mu_map.get(mu_val, 0.5)

        data_list.append((s0, s1, mu))

pair_data = np.array(data_list, dtype=[("s0", "i4", (2,)), ("s1", "i4", (2,)), ("mu", "f4")])

print(pair_data)


np.savez("scripts/human.npz", data=pair_data)