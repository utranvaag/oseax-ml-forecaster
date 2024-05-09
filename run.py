"""
Initializes and runs the specified machine learning forecasing experiment.
It reads command-line arguments to set parameters for the experiment,
chooses the model based on the first argument, and initiates the model training and evaluation.

Arguments:
1: Forecasting model type (e.g., 'lstm')
2: Forecasting horizon by number of business days (e.g, '5' for a business-week)
3: Dataset (e.g., 'selected')
4: Run-mode (e.g., 'test' for testing)
5: Computing resource (e.g., 'GPU' for utilizing GPUs if available)
6: Number of threads/workers (e.g., '2')
7: Console output mode (e.g., 'silent')

Usage:
python run.py lstm 1 univariate test_light cpu 0 print

Raises:
ValueError: If an unknown argument is specified.
"""

import sys
import torch
from scripts.model_lstm_standard import run_lstm_standard_experiment
from scripts.initialization import (
    silence_irrelevant_messages,
    set_experiment_parameters,
    control_args,
)


N_ARGS_COMMAND_LINE = 8
DATASET_FOLDER_PATH = 'dataset/processed_data'
MODEL_MAPPINGS = {
    'lstm': run_lstm_standard_experiment,
}


def handle_command_line_arguments():
    """
    Processes command-line arguments to configure the experiment's settings.

    Returns:
        model (str): The model type as specified by the first argument.
        dataset (str): Dataset configuration as specified by the third argument.
        seeds (list): Seed values, either fixed or for randomness, derived from the configuration.
        prm (dict): A dictionary containing all other parameter settings.
    """

    control_args(N_ARGS_COMMAND_LINE, sys.argv)
    dataset, seeds, prm = set_experiment_parameters(sys.argv, DATASET_FOLDER_PATH)
    model = sys.argv[1].lower()

    return model, dataset, seeds, prm


def main():
    """Initiating experiment"""

    torch.multiprocessing.freeze_support()  # For multi GPU-support, if to be used
    silence_irrelevant_messages()           # Preventing console spam

    model, dataset, seeds, prm = handle_command_line_arguments()

    # Initializes model experiment by given parameters:
    if model in MODEL_MAPPINGS:
        MODEL_MAPPINGS[model](dataset=dataset, seeds=seeds, prm=prm)
    else:
        raise ValueError(
            f"Unknown model type passed: <{model}>. "
            f"Available options: {list(MODEL_MAPPINGS.keys())}"
        )


if __name__ == "__main__":
    main()
