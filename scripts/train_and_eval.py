"""Forecasting model training and evalutation script"""

import os
import random
from datetime import datetime
import numpy as np
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scripts.helpfuncs_models import LossLogger
from scripts.helpfuncs_plots import (
    plot_results,
    generate_and_save_loss_plots_and_loss_data,
)
from scripts.helpfuncs_models import (
    df_to_csv,
    make_output_dir,
    get_scores,
    oseax_yoy_to_idx,
    get_oseax_as_idx_values,
    to_latex_table,
)


def train_and_evaluate_model(
        model_cls,
        dataset: dict,
        scalers: dict,
        prm: dict,
        **kwargs,
):
    """
    Initializes, trains and evaluates a the provided Darts forecasting model
    """

    model_time_start = datetime.now()

    if prm['SEED'] is None:
        seed = random.randint(0, 2**32 - 1)  # Generating a random seed.
        print(seed, "(random)]")
    else:
        seed = prm['SEED']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = '0'
        print("]")

    # Checking for folder paths and generating those missing:
    model_name = (
        f"{prm['MODEL_NAME']}_model_{prm['MODEL_N']}/"
    )
    main_path = (
        f"{prm['SAVE_PATH']}h_{prm['HORIZON']}/{prm['DATASET_NAME']}/"
    )
    save_path = (
        f"{main_path}individual_runs/{model_name}/"
    )
    weights_save_path = save_path + "model_weights/"
    output_path = save_path + "output_data/"
    make_output_dir(save_path)
    make_output_dir(weights_save_path)
    make_output_dir(output_path)

    # Measuring time expenditure:
    time_start = datetime.now()

    if prm['PRINT']:
        print("["+model_name + "]", "starting  :", time_start)
        print("["+model_name + "]", "Saving to :", save_path)

    # Model initialization --------------------------------------------

    torch.manual_seed(seed)  # Setting seed

    # Callbacks management:
    my_stopper = EarlyStopping(
        monitor="val_loss",
        patience=prm['STOPPER_PATIENCE'],
        min_delta=prm['STOPPER_MIN_DELTA'],
        mode='min',
        verbose=prm['PRINT'],
    )
    loss_logger = LossLogger()
    callbacks = [
        my_stopper,
        loss_logger,
    ]

    # Setting pl_trainer_kwargs accordingly to GPU-availability:
    if (torch.cuda.is_available()) and (prm['USE_GPU'] is True):
        pl_trainer_kwargs = {
            "callbacks": callbacks,
            "accelerator": "gpu",
            "devices": -1,      # Use all available (GPU) devices
            "strategy": "ddp",   # Use data distributed parallelism
        }
    else:
        pl_trainer_kwargs = {
            "callbacks": callbacks
        }

    model = model_cls(
        work_dir=weights_save_path,
        random_state=seed,
        pl_trainer_kwargs=pl_trainer_kwargs,
        **kwargs
    )

    # Model training --------------------------------------------------

    if prm['TAKES_FUTC']:
        model.fit(
            series=dataset['train']['target variable'],
            future_covariates=dataset['train']['future covariates'],
            past_covariates=dataset['train']['past covariates'],
            val_series=dataset['validation']['target variable'],
            val_future_covariates=dataset['validation']['future covariates'],
            val_past_covariates=dataset['validation']['past covariates'],
            verbose=prm['PRINT'],
            num_loader_workers=prm['N_WORKERS'],
        )
    else:
        model.fit(
            series=dataset['train']['target variable'],
            past_covariates=dataset['train']['past covariates'],
            val_series=dataset['validation']['target variable'],
            val_past_covariates=dataset['validation']['past covariates'],
            verbose=prm['PRINT'],
            num_loader_workers=prm['N_WORKERS'],
        )

    # Removing first entry in val_loss (faulty from sanity check):
    loss_logger.val_loss = loss_logger.val_loss[1:]
    print()  # New line after no-newline loss prints.

    generate_and_save_loss_plots_and_loss_data(
        loss_logger=loss_logger,
        output_dir=output_path+'loss/',  # Own folder for loss info.
        save=prm['SAVE_LOSS_PLOTS'],
        info_print=prm['PRINT'],
        info_plot=prm['TROUBLESHOOTING_PLOTS'],
    )

    # Loading best instance of model weights (epoch with best results):
    best_model = model.load_from_checkpoint(
        model_name='',
        work_dir=weights_save_path,
        best=True  # Instructing function to load best instance.
    )

    # Getting info on numbers of epochs trained:
    n_epochs_trained = model.epochs_trained - 1
    n_epochs_trained_best = n_epochs_trained - prm['STOPPER_PATIENCE']
    if n_epochs_trained < prm['MAX_EPOCHS']:
        print(" | Early-stopped at epoch", n_epochs_trained, end=" ")
        print(f"(best ep: {n_epochs_trained_best})")
    else:
        print(f"[!] Max epochs reached ({prm['MAX_EPOCHS']})")
        # Set to max due to unknown if best is max or n<=patience earlier epoch
        # In this case, best ep must be read manually from filename.
        n_epochs_trained_best = prm['MAX_EPOCHS']

    # Model backtesting -----------------------------------------------

    # Historical forecast (backtesting):
    print(f" | [{datetime.now().strftime('%H:%M:%S')}] Backtesting")
    if prm['TAKES_FUTC']:
        forecast_yoy_scaled = best_model.historical_forecasts(
            series=dataset['test']['target variable'],
            past_covariates=dataset['test']['past covariates'],
            future_covariates=dataset['test']['future covariates'],
            forecast_horizon=prm['HORIZON'],
            stride=1,
            num_samples=1,
            retrain=False,
            verbose=False,  # no "prm['PRINT']" - verbose broken (spaming)
        )
    else:
        forecast_yoy_scaled = best_model.historical_forecasts(
            series=dataset['test']['target variable'],
            past_covariates=dataset['test']['past covariates'],
            forecast_horizon=prm['HORIZON'],
            stride=1,
            num_samples=1,
            retrain=False,
            verbose=False,  # no "prm['PRINT']" - verbose broken (spaming)
        )

    # To be saved as a time performance metric:
    time_end = datetime.now()
    time_expenditure = time_end - time_start

    # Model evaluation ------------------------------------------------

    # Inverse transforming using scaler fitted on target OSEAX series:
    forecast_yoy = scalers['scaler_target'].inverse_transform(
        forecast_yoy_scaled.copy()
    )

    # Converting forecasts from yoy to index valuations:
    forecast = oseax_yoy_to_idx(
        sr_oseax_yoy=forecast_yoy.copy(),
        target=prm['TARGET_VAR'],
        idx_dataset_path='dataset/dataset_full_half_preprocessed_info.csv',
    )

    # Saving scaled, yoy, and idx versions of forecast to csv:
    forecasts_dir = output_path+"forecasts/"

    make_output_dir(forecasts_dir)
    forecasts_yoy_scaled_df = forecast_yoy_scaled.pd_dataframe(copy=True)
    df_to_csv(forecasts_yoy_scaled_df, forecasts_dir, 'forecast_yoy_scaled')
    forecast_yoy_df = forecast_yoy.pd_dataframe(copy=True)
    df_to_csv(forecast_yoy_df, forecasts_dir, 'forecast_yoy')
    forecast_df = forecast.pd_dataframe(copy=True)
    df_to_csv(forecast_df, forecasts_dir, 'forecast_idx')

    # Retreiving target series for testing period:
    test_target_yoy_scaled = dataset['test']['target variable'].copy()
    test_target_yoy = scalers['scaler_target'].inverse_transform(
        test_target_yoy_scaled.copy()
    )
    test_target = oseax_yoy_to_idx(
        sr_oseax_yoy=test_target_yoy,
        target=prm['TARGET_VAR'],
        idx_dataset_path='dataset/dataset_full_half_preprocessed_info.csv',
    )

    # Retreiving target series for full time series data set duration:
    oseax_idx_full_sr = get_oseax_as_idx_values(
        raw_dataset_path='dataset/dataset_full_half_preprocessed_info.csv',
        oseax_col_name=prm['TARGET_VAR'],
    )

    test_target_df = test_target.pd_dataframe(copy=True)
    oseax_idx_full_df = test_target.pd_dataframe(copy=True)
    for i in range(0, len(test_target[prm['TARGET_VAR']])):
        i_value = test_target_df[prm['TARGET_VAR']][i]
        i_date = test_target_df.index[i]
        t_value = oseax_idx_full_df[prm['TARGET_VAR']][i_date]
        if i_value != t_value:
            print("   !] Non equal values: ", i_date, i_value, t_value)

    # Plotting results:
    plot_results(
        full_target=oseax_idx_full_sr,
        test_target=test_target,
        model_forecast=forecast,
        n_last_d=65,  # Because 64-66 BDays in a quarter on average
        directory=save_path+"plots/",
        show=prm['TROUBLESHOOTING_PLOTS'],
        save=prm['PLOT_ALL'],
        incl_c19_crash_focus=True
    )
    if prm['PRINT']:
        # Testing for overlap: If H & T = 1d, overlap = 3, due to how
        # the model works: removing one day from the datasets end (+1).
        print(
            "["+model_name + "]",
            "len(true)-len(forecast):",
            len(test_target)-len(forecast)
        )

    # TODO: Add scatterplot?

    # Calculating metrics ---------------------------------------------

    # Generating and saving model performance metrics:
    scores = get_scores(
        target_series=test_target,
        forecast_series=forecast,
        time_expenditure=time_expenditure,
        round_to=None,
    )

    # Adding number of eps' to scores (for information):
    scores.insert(
        loc=5,
        column='epochs',
        value=n_epochs_trained_best,
    )

    # Generating LaTeX table:
    to_latex_table(
        scores,
        filepath=save_path+'output_data/metrics',
        filename='scores_LaTeX',
    )

    df_to_csv(scores, path=save_path+'output_data/metrics', name='scores')

    # Calculating metrics done ----------------------------------------

    if prm['PRINT']:
        print(f" | Training COMPLETE [{time_end}]")
        print(f" | Time expenditure: {time_expenditure}")
        print("."*100)

    # Saving time expenditure:
    model_time_end = datetime.now()
    model_time_expenditure = model_time_end - model_time_start
    print(
        f" | [{datetime.now().strftime('%H:%M:%S')}] "
        f"Time_expenditure: {model_time_expenditure}\n"
        f" |______________________________________________________________"
    )

    return forecast_df, scores
