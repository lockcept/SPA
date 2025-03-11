import os
import sys

import torch

from src.utils.path import get_classifier_model_path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# pylint: disable=C0413

from data_loading.preference_dataloader import get_dataloader
from src.auto_encoder.autoencoder import AutoEncoder, train_autoencoder
from src.data_generation.classifier.classifier import (
    Classifier,
    evaluate_classifier,
    train_classifier,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    env_name = "box-close-v2"
    exp_name = "AESPA-0-00"
    # train_autoencoder(env_name)
    train_classifier(
        env_name=env_name, exp_name=exp_name, pair_algo="full-binary-with-0.5"
    )

    encoder = AutoEncoder(input_dim=43 * 25).encoder

    model = Classifier(encoder=encoder)
    classifier_path = get_classifier_model_path(
        env_name=env_name,
        exp_name=exp_name,
        pair_algo="full-binary-with-0.5",
    )
    model.load_state_dict(
        torch.load(classifier_path, weights_only=True, map_location=device)
    )

    dataloader = get_dataloader(
        env_name=env_name,
        exp_name=exp_name,
        pair_type="val",
        pair_algo="full-binary-with-0.5",
    )

    acc = evaluate_classifier(
        model=model,
        dataloader=dataloader,
        device=device,
    )
    print("Accuracy:", acc)
