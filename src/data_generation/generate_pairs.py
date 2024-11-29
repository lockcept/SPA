from data_generation.full_scripted_teacher import generate_full_pairs
from data_generation.list_scripted_teacher import generate_list_pairs
from data_generation.scored_pairs import generate_score_pairs
from data_loading.load_data import load_dataset


def generate_all_algo_pairs(env_name, pair_name_base, include_score_pairs=False):
    """
    generate all algo pairs with hard-coded values
    """
    TRAIN_TRAJECTORIES = 200
    VAL_TRAJECTORIES = 300
    TEST_TRAJECTORIES = 1500

    TRAIN_PAIRS = 1000
    VAL_PAIRS = 400
    TEST_PAIRS = 2000

    dataset = load_dataset(env_name=env_name)

    # generate_full_pairs(
    #     env_name=env_name,
    #     pair_name_base=pair_name_base,
    #     num_pairs=num,
    #     mu_types=[
    #         "binary",
    #         "sigmoid",
    #         "linear",
    #     ],
    # )
    # generate_list_pairs(
    #     env_name=env_name,
    #     pair_name_base=pair_name_base,
    #     num_trajectories=num,
    #     pair_algos=["list-2", "list-3", "list-5", "list-11"],
    # )
    # generate_score_pairs(
    #     env_name=env_name,
    #     pair_name_base=pair_name_base,
    #     num_pairs=num,
    #     pair_algos=["rnn"],
    # )
