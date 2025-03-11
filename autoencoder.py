import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# pylint: disable=C0413

from src.auto_encoder.autoencoder import train_autoencoder


if __name__ == "__main__":
    env_name = "box-close-v2"
    train_autoencoder(env_name)
