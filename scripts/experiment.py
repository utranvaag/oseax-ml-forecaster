"""
Training, testing and evaluation a the passed model with the
given parameters, datasets and other variables set in this file.
"""

import multiprocessing
from datetime import datetime
import pandas as pd
import torch
from scripts.train_and_eval import train_and_evaluate_model
from scripts.helpfuncs_models import df_to_csv, pprint_aligned
from scripts.helpfuncs_plots import (
    plot_forecast_summary,
    plot_forecast_summary_interesting_periods_focus,
)
from scripts.helpfuncs_data_preprocessing import (
    to_bday_series,
    split_and_normalize,
)


def run_experiment(
        model_cls,
        dataset: pd.DataFrame,
        prm: dict,
        seeds: list,
        **kwargs,
):
    """
    Runs n-seeds instances of the provided model and compares the results.
    """

    experiment_start = datetime.now()
    start_time = datetime.now()

    # Adding save path to experiment parameters:
    prm['SAVE_PATH'] = 'results/' + prm['MODEL_NAME'] + '/'
    prm['TAKES_FUTC'] = True if model_cls.supports_future_covariates else False

    # Initial information-printing:
    print("[i] Model                :", prm['MODEL_NAME'])
    print("[i] Dataset              :", prm['DATASET_NAME'])

    # Info: n-workers total and assigning desired amount:
    num_workers_available = multiprocessing.cpu_count()
    print(f'[i] Available workers    : {num_workers_available}', end=" ")
    if prm['N_WORKERS'] is None:
        print("utilizing all)")
    else:
        print(f"(utilizing {prm['N_WORKERS']})")

    # Info: if set to run with GPUs:
    if prm['USE_GPU']:
        print("[i] use_gpu set to TRUE: attempting to run on available GPU(s)")
        print("[i] CUDA devices available:", torch.cuda.is_available(),
              end=" "
              )
        if torch.cuda.is_available():
            print(f"- {torch.cuda.device_count()} GPU(s)")
            for device in range(torch.cuda.device_count()):
                print(f"    {device}: {torch.cuda.get_device_name(device)}")
            print("[i] Running on GPU")
        else:
            print()
    else:
        print("[i] Running on CPU")

    # Info: parameters
    print("[i] Experiment parameters:")
    pprint_aligned(prm, indent=5)
    print("[i] Model hyperparameters:")
    pprint_aligned(model_cls(work_dir='.', **kwargs).model_params, indent=5)

    # Running models:
    print("\n[i]_RUNNING_EXPERIMENT__________________________________________")

    all_scores_df = pd.DataFrame()
    all_forecasts_df = pd.DataFrame()
    model_path = f"{prm['SAVE_PATH']}h_{prm['HORIZON']}/{prm['DATASET_NAME']}/"

    # Final data preprocessing pipeline:

    # 1. DataFrames to Darts TimeSeries
    dataset_series = to_bday_series(dataset)

    # 2. Convert futcov compatibles to futcovs and remove them from pastcovs:
    dataset_container, scalers_container = split_and_normalize(
        series=dataset_series,
        prm=prm,
    )

    # Running n-experiments:

    for seed_n, seed in enumerate(seeds):

        prm['SEED'] = seed
        prm['MODEL_N'] = seed_n + 1  # Keeping track of which n model

        print(
            f" | [{datetime.now().strftime('%H:%M:%S')}] "
            f"Model {seed_n+1}/{len(seeds)} "
            f"[seed : {seed if seed else ''}",
            end="",
        )

        forecast, scores = train_and_evaluate_model(
            model_cls=model_cls,
            dataset=dataset_container,
            scalers=scalers_container,
            prm=prm,
            **kwargs,
        )

        # Appending models forecast/scores to the all-lists:
        model_name = f'model_{seed_n+1}'
        all_forecasts_df[model_name] = forecast.squeeze()
        all_scores_df[model_name] = scores.squeeze()

    print("[i] EXPERIMENT FINISHED")

    # Combinig all individual forecasts into one and plotting:
    average_forecast = all_forecasts_df.mean(axis=1)
    plot_forecast_summary(
        average_forecast=average_forecast,
        all_forecasts=all_forecasts_df,
        output_dir=model_path,
        filename='forecasts_summary_full_plot',
        true_path='dataset/dataset_full_half_preprocessed_info.csv',
        true_vals_col_name=prm['TARGET_VAR'],
    )
    plot_forecast_summary_interesting_periods_focus(
        average_forecast=average_forecast,
        all_forecasts=all_forecasts_df,
        output_dir=model_path,
        filename='forecasts_summary',
        true_path='dataset/dataset_full_half_preprocessed_info.csv',
        true_vals_col_name=prm['TARGET_VAR'],
        n_last_days=20,
    )

    # Mean & standard deviation calculation:

    # Convert time to total seconds:
    all_scores_df.loc['time'] = all_scores_df.loc['time'].apply(
        pd.to_timedelta,
    ).dt.total_seconds()

    # Ensuring that all other rows are numeric before calculating:
    all_scores_df = all_scores_df.apply(pd.to_numeric, errors='coerce')

    # Computing the mean and standard deviation of all_scores:
    mean_scores = all_scores_df.mean(axis=1)
    std_scores = all_scores_df.std(axis=1)
    combined_scores = pd.concat([mean_scores, std_scores], axis=1)
    combined_scores.columns = ['Mean', 'StdDev']
    combined_scores.rename(index={'time': 'time_model_avg'}, inplace=True)

    # Adding full model experiment time to combined scores:
    experiment_timex = datetime.now() - experiment_start
    experiment_timex_str = str(experiment_timex)
    new_row = {'Mean': experiment_timex_str, 'StdDev': 'None'}
    combined_scores.loc['time_full_scipt'] = new_row

    # Saving combined scores and forecasts:
    df_to_csv(all_forecasts_df, model_path, 'forecasts_individual')
    df_to_csv(average_forecast, model_path, 'forecast_mean')
    df_to_csv(all_scores_df, model_path, 'scores_individual')
    df_to_csv(combined_scores, model_path, 'scores_mean_stddev')

    # For measuring run time:
    end_time = datetime.now()
    end_time_f = end_time.strftime("%Y-%m-%d %H:%M:%S")

    time_expenditure = end_time - start_time
    time_expenditure_str = str(time_expenditure)

    # Write to file
    with open(
        file=prm['SAVE_PATH']+'time_expenditure.txt',
        mode='a',
        encoding='utf-8-sig',
    ) as file:
        file.write(
            f'Date: {end_time_f} | Time expenditure: {time_expenditure_str}\n'
        )

    # TODO I   : Impl manual features select
    # TODO II  : Impl feat select
    # TODO III : Impl hyperparamopt

    # Final info print (time expenditure)
    print(
        f"\n - DONE - [{end_time_f}] | time expenditure: {time_expenditure}"
    )
