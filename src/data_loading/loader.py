import os
import gym
import d4rl #import 해야 gym.make()에서 d4rl 환경을 불러올 수 있음
import numpy as np

def save_dataset(env_name, dataset):
    save_dir = f"dataset/d4rl/{env_name}"
    os.makedirs(save_dir, exist_ok=True)
    print (dataset)
    
    file_path = os.path.join(save_dir, "dataset.npz")
    
    np.savez(file_path, observations=dataset["observations"], actions=dataset["actions"], rewards=dataset["rewards"], terminals=dataset["terminals"], timeouts=dataset["timeouts"]) 
    print(f"Dataset for {env_name} saved at {file_path}")

def load_antmaze():
    env_name = "antmaze-umaze-v0"
    antmaze_env = gym.make(env_name)
    antmaze_dataset = antmaze_env.get_dataset()
    
    save_dataset(env_name, antmaze_dataset)
    
    return antmaze_dataset


if __name__ == "__main__":
    antmaze_dataset = load_antmaze()