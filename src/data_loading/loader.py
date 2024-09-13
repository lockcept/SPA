import gymnasium as gym
import d4rl


def load_antmaze():
    antmaze_env = gym.make("antmaze-umaze-v0")
    antmaze_dataset = antmaze_env.get_dataset()
    return antmaze_dataset


def load_hopper():
    hopper_env = gym.make("hopper-medium-v0")
    hopper_dataset = hopper_env.get_dataset()
    return hopper_dataset


if __name__ == "__main__":
    antmaze_dataset = load_antmaze()
    print(antmaze_dataset["observations"].shape)
    print(antmaze_dataset["actions"].shape)

    hopper_dataset = load_hopper()
    print(hopper_dataset["observations"].shape)
    print(hopper_dataset["actions"].shape)
