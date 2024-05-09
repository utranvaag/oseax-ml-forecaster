"""Functions to be used by all machine learning models"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from pytorch_lightning.callbacks import Callback
from darts import TimeSeries
from darts.metrics import mape, rmse, r2_score, smape, mse


def make_output_dir(dir_path):
    """Checks if provided path exists. If not it creates it"""

    # Checking for outputs folder:
    check_folder = os.path.isdir(dir_path)
    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(dir_path)


def get_oseax_as_idx_values(
        raw_dataset_path: str,
        oseax_col_name: str,
        split_after=None
):
    """Retreives the OSEAX timeseries as points-denominated index valuations"""

    dataset = pd.read_csv(
        raw_dataset_path,
        parse_dates=['Date'],
        delimiter=';',
        index_col=['Date'],
    )
    series = (TimeSeries.from_dataframe(
        dataset,
        freq='B'
    )).astype(np.float32)
    oseax_series = series[oseax_col_name]

    if split_after is None:
        return oseax_series
    _, test = oseax_series.split_after(split_after)

    return test


def oseax_yoy_to_idx(
        sr_oseax_yoy: TimeSeries,
        target: str,
        idx_dataset_path: str
):
    """
    Coverts a year-onyear differentiated oseax series to true index values
    """

    # Converting to DataFrames:
    df_oseax_yoy = sr_oseax_yoy.pd_dataframe(copy=True)
    df_oseax_idx = get_oseax_as_idx_values(
        raw_dataset_path=idx_dataset_path,
        oseax_col_name=target
    ).pd_dataframe(copy=True)

    df_full_valued = pd.DataFrame(columns=[target], index=df_oseax_yoy.index)

    for i in range(0, len(df_oseax_yoy[target])):
        i_value = df_oseax_yoy[target][i]
        i_date = df_oseax_yoy.index[i]
        idx_date = i_date - BDay(261)
        df_full_valued[target][i_date] = df_oseax_idx[target][idx_date] * (
            1+i_value
        )

    series_full_valued = TimeSeries.from_dataframe(
        df=df_full_valued,
        freq='B',
    ).astype(np.float32)

    return series_full_valued


# Custom callback loss logger:
class LossLogger(Callback):
    """
    Custom loss logger. Automatically called at the end of each epoch.

    Note: The callback will give one more element in the loss_logger.val_loss
    as the model trainer performs a validationsanity check before the training
    begins.

    Example of use:\n
    loss_logger = LossLogger()

    model = SomeTorchForecastingModel(\n
        ...,\n
        nr_epochs_val_period=1,  # perform validation after every epoch\n
        pl_trainer_kwargs={"callbacks": [loss_logger]}\n
    )
    """

    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:

        self.train_loss.append(float(trainer.callback_metrics['train_loss']))

    def on_validation_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:

        val_loss = float(trainer.callback_metrics['val_loss'])
        self.val_loss.append(val_loss)

        print(
            f"\r | [{datetime.now().strftime('%H:%M:%S')}] "
            f"Epoch {len(self.val_loss)-1}, "
            # f"train_loss {round(self.train_loss[0], 6)} | "
            f"Validation loss: {round(val_loss, 6)} "
            f"(best {round(min(self.val_loss), 6)})",  # :<50}",
            end='',
        )


def get_scores(
        target_series: TimeSeries,
        forecast_series: TimeSeries,
        time_expenditure: datetime,
        round_to: int,
):
    """Calculates scores/metrics and returns the in a DataFrame"""

    # Cutting of parts not overlapping with forecast series:
    target_series = target_series.slice_intersect(forecast_series)

    # Calculating & saving scores in a dict:
    scores = {
        'RMSE':     rmse(target_series, forecast_series),
        'MAPE':     mape(target_series, forecast_series),
        'R2_score': r2_score(target_series, forecast_series),
        'sMAPE':    smape(target_series, forecast_series),
        'MSE':      mse(target_series, forecast_series),
    }

    if round_to is not None:
        for key, score in scores.items():
            scores[key] = round(score, round_to)
    scores['time'] = time_expenditure  # Also adding time expenditure stat

    # Converting dict to DataFrame:
    scores_df = pd.DataFrame([scores])

    return scores_df


def to_latex_table(
        dataframe: pd.DataFrame,
        filepath: str,
        filename: str,
) -> None:
    """
    Saves df as LaTeX code as .txt.
    Initiates a new directory if filepath is non-existent.
    """

    make_output_dir(filepath)
    path = filepath + '/' + filename + '.txt'

    # Convert the DataFrame to a LaTeX table:
    dataframe.style.to_latex(
        path,
    )


def pprint_aligned(dct: dict, indent: int = 0, start_with: str = ""):
    """Prints the given dict with ":"s aligned"""

    # Get the length of the longest key
    max_key_length = max(len(str(key)) for key in dct)

    for key, value in dct.items():
        line = start_with + " " * indent
        line += f"{key:<{max_key_length}} : {value}"
        print(line)


def df_to_csv(
        data: pd.DataFrame,
        path: str,
        name: str,
):
    """
    Saves the given dataframe as a .csv with standard settings
    * sep = ';'
    * encoding = 'utf-8-sig'

    and creates a directory if non-existent.
    """

    make_output_dir(path)
    path = path + '/' + name + '.csv'

    data.to_csv(
        path,
        sep=';',
        encoding='utf-8-sig',
    )


def remove_microseconds(delta):
    """Removes microseconds from timedelta object"""
    return delta - timedelta(microseconds=delta.microseconds)
