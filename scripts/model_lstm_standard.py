"""
Training, testing and evaluation a Long Short-term Memory network model for the
given parameters, datasets and other variables set in this file.
"""

import pandas as pd
from darts.models import BlockRNNModel
from scripts.experiment import run_experiment


def run_lstm_standard_experiment(
        dataset: pd.DataFrame,
        prm: dict,
        seeds: list[int],
):
    """Initiates experiemnt for the model named in filename"""

    experiment_parameters = {
        'MODEL_NAME_ADDITION':  '',     # If to make own folder for this model
        'WINDOW_SIZE':          22,     # Standard - NOTE: to be hparamopted
        'STOPPER_PATIENCE':     50,     # Standard
        'STOPPER_MIN_DELTA':    0.00,   # Standard
        'TRAIN_SHARE':          0.70,
        'VAL_SHARE_OF_DS':      0.15,
        'SAVE_LOSS_PLOTS':       True,      # Save individual loss plots
        'PLOT_ALL':              True,      # Save individual forecast plots
        'TROUBLESHOOTING_PLOTS': False,     # Presents troubleshooting-plots
    }
    prm.update(experiment_parameters)
    prm['MODEL_NAME'] = prm['MODEL'] + prm['MODEL_NAME_ADDITION']

    run_experiment(
        model_cls=BlockRNNModel,
        dataset=dataset,
        prm=prm,
        seeds=seeds,
        # kwargs:
        model='LSTM',
        output_chunk_length=prm['HORIZON'],
        n_epochs=prm['MAX_EPOCHS'],
        model_name='',  # Prevent model from modifying dirs
        # Tunable parameters:
        input_chunk_length=prm['WINDOW_SIZE'],
        save_checkpoints=True,
        force_reset=True,       # If True: previously-existing models are reset
        log_tensorboard=False,  # If True: using Tensorboard to log paramters
    )
