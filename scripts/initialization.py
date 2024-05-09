"""
Contains functions for initializing the forecasting experiment.
"""

import os
import logging
import warnings
from datetime import datetime
import pandas as pd



def silence_irrelevant_messages():
    """
    Suppresses irrelevant warnings and logging messages that could clutter the console output.
    """

    # Silencing irrelevant warnings:
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.random")
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.ERROR)

    # Silencing PyTorch Lighning print spam:
    logging.getLogger('lightning').setLevel(0)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


def set_experiment_parameters(arguments, dataset_path):
    """
    Configures and returns the experimental parameters, dataset, and seeds based on the provided
    arguments.

    Parameters:
        arguments (List[str]): A list containing command-line arguments which include the model
                               type, horizon, dataset name, run mode, computing resource, number of
                               workers, and print option, in that order.
        dataset_path (str): The base path where the datasets are located. This path is used to
                            construct the full path to the specific dataset CSV file.

    Returns:
        tuple: A tuple containing the loaded dataset as a DataFrame, a list of seed values for the
               runs, and a dictionary of parameters that include settings like number of epochs,
               whether to use GPU, and logging preferences.
    """

    prm = {}  # Dict for holding parameters

    horizon = arguments[2]
    dataset_name = arguments[3]
    run_mode = arguments[4]
    resource = arguments[5]
    n_workers = arguments[6]
    print_opt = arguments[7]

    indataset = get_dataset(f'{dataset_path}/h_{horizon}/{dataset_name}.csv')

    # Setting print option:
    if print_opt == "print":
        prm['PRINT'] = True
    elif print_opt == "silent":
        prm['PRINT'] = False
    else:
        raise ValueError(
            f"Unknown 7th argument passed: <{print_opt}>. "
            "Pass <print> print all or <silent> to print minimal info"
            )

    # Setting number of workers:
    if n_workers == "all":
        prm['N_WORKERS'] = None
    else:
        try:
            prm['N_WORKERS'] = int(n_workers)  # Trying to convert str to int
        except ValueError as exc:
            raise ValueError(
                f"Unknows 6th argument passed: <{n_workers}>. "
                f"Pass an <<integer>> to allocate a defined number of "
                f"workers, <0> to only use main process, or <All> to "
                f"utilize all available workers"
            ) from exc

    # Setting computing resources (CPU/GPU):
    if resource == "cpu":
        prm['USE_GPU'] = False
    elif resource == "gpu":
        prm['USE_GPU'] = True
    else:
        raise ValueError(
            f"Unknown 5th argument passed: <{resource}>. "
            f"Pass <CPU> for running the program on the CPU, "
            f"or <GPU> to run the program on all available GPU(s)"
            )

    # Setting model forecasting horizon:
    try:
        prm['HORIZON'] = int(horizon)
    except ValueError as exc:
        raise ValueError(
            f"Unknows 2nd arg (forecasting_horizon) passed: <{horizon}>. "
            f"Must be passed as <class int> "
        ) from exc

    # Setting parameters and modifying datased according to run mode:
    indataset, seeds, prm = set_experiment_run_mode(run_mode, indataset, prm)

    prm['MODEL'] = arguments[1]                 # Setting model
    prm['DATASET_NAME'] = dataset_name          # Setting dataset name
    prm['TARGET_VAR'] = indataset.columns[0]    # Setting target variable
    prm['START_TIME'] = datetime.now()          # Recording start-time

    return indataset, seeds, prm


def set_experiment_run_mode(run_mode, indataset, prm):
    """
    Setting parameters and modifying datased according to run mode
    """

    if run_mode == "test":
        prm['MAX_EPOCHS'] = 5
        seeds = [42, None]
        indataset = indataset.tail(261*3)

    elif run_mode == "test_light":
        prm['MAX_EPOCHS'] = 1
        seeds = [42]
        indataset = indataset.tail(261*1)

    elif run_mode == "full":
        prm['MAX_EPOCHS'] = 1000
        seeds = [
            1284283653, 1436149779, 2040307284, 3946869200, 678601114,
            1071757855, 991116103, 271425751, 3787927488, 2050899463,
            2986567991, 2079996420, 1164449607, 786093039, 340742643,
            1356974857, 821158064, 3185019807, 241797956, 2092134574,
            294790851, 2899700789, 1713719506, 2658347021, 967060114,
            1430473511, 1909083991, 3187678702, 3516279659, 3662158617,
        ]

    else:
        raise ValueError(
            f"Unknows 4th argument passed: <{run_mode}>. Pass:\n"
            f"<'full'> to run full dataset, or\n"
            f"<'test'> to run in test-mode"
        )
    # Setting number of runs anc each runs seed (None = random seed):
    # E.g.: len(seeds) = no of runs. set None for runs w/random seed

    return indataset, seeds, prm


def get_dataset(path: str) -> pd.DataFrame:
    """
    Gets dataset as pandas DataFrame from the given path.
    
    Parameters:
        path (str): Path to the CSV file to be read.

    Returns:
        pd.DataFrame: The dataset loaded into a DataFrame.

    Raises:
        FileNotFoundError: If the specified file is not found.
        pd.errors.ParserError: If there is an error parsing the file.
    """
    try:
        dataset = pd.read_csv(
            path,
            parse_dates=['Date'],
            delimiter=';',
            encoding='utf-8-sig',
        )
        dataset.set_index(keys='Date', inplace=True)
        return dataset
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The file at {path} was not found.") from exc
    except pd.errors.ParserError as exc:
        raise pd.errors.ParserError("There was a problem parsing the file.") from exc


def control_args(n_args, arguments):
    """
    Ensures the required number of command-line arguments are passed.

    Parameters:
        n_args (int): The required number of arguments.
        arguments (List[str]): List of arguments passed to the script.

    Raises:
        ValueError: If the number of arguments does not match the expected count.
    """

    if len(arguments) != n_args:
        raise ValueError(
            f"\nRequired n args : {n_args}. "
            f"\nGiven           : {len(arguments)}.\n"
            f"Run file as <<\n{os.path.basename(__file__)} "
            f"model_type: <str>, "
            f"horizon: <int> ('1' or '5'), "
            f"dataset: <str> ('univariate' or 'selected'), "
            f"run-mode: <str> ('test', 'test_light', or 'full'), "
            f"resource: <str> ('cpu' or 'gpu'), "
            f"n-workers: <int> or <str> ('all'), "
            f"print-option: <str> ('print' or 'silent')\n>>\n"
            f"EXAMPLE: python run.py lstm 1 univariate test_light cpu 2 print"
        )
