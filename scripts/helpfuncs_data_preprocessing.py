"""Helping fuctions and methods for data preprocessing"""

import io
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries as ts
from darts.dataprocessing.transformers import Scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from scripts.helpfuncs_models import make_output_dir


def shorten_feature_names(
        dataframe: pd.DataFrame,
):
    """
    Shortens feature names by removing everything after ":" within their name
    string.
    """

    col_names = dataframe.columns  # extracting column names
    new_col_names = []             # list for new short names

    for col_name in col_names:
        words = col_name.split()   # getting words
        new_name = ""
        for word in words:
            new_name += word
            if ":" in word:
                new_name = new_name.strip(":")
                break
            else:
                new_name += " "

        new_col_names.append(new_name)   # adding to short names-list

    dataframe = dataframe.rename(columns=dict(zip(dataframe, new_col_names)))

    return dataframe


def save_to_txt(
        dataframe: pd.DataFrame,
        filename: str = "dataset/dataset_preprocessed_info.txt"
):
    """Saving given pandas DataFrame as a .txt file"""
    # Create a buffer
    buf = io.StringIO()

    # Pass buffer to dataframe.info
    dataframe.info(buf=buf)

    # Get the string from the buffer
    info_str = buf.getvalue()

    # Write the string to a file
    with open(filename, 'w', encoding='utf-8-sig') as file:
        file.write(info_str)


def remove_non_bdays(
        dataframe: pd.DataFrame,
):
    """Removes all rows which are not Business Days"""

    # 1. get BD-range from min to max of in-dataframes date column:
    bd_range = pd.bdate_range(
        start=dataframe.index.min(),
        end=dataframe.index.max(),
    )

    # 2. filtering dataframe:
    dataframe = dataframe[dataframe.index.isin(bd_range)]

    return dataframe


def to_bday_series(
        dataset: pd.DataFrame(),
):
    """
    Converts a pandas DataFrame to Darts TimeSeries, and
    * checks for NaNs
    * ensures BDaily frequency
    * converts to np.float32
    """

    # Checking for NaNs in dataset:
    if dataset.isna().sum().sum() > 0:
        nan_cols = dataset.isna().sum()
        nan_cols = nan_cols[nan_cols > 0]
        error_message = f"Found NaN values in the dataset:\n{nan_cols}"
        raise ValueError(error_message)

    # Initiating Darts TimeSeries object and convertinf to np32:
    series = (ts.from_dataframe(
        df=dataset,
        freq='B',
    )).astype('float32')

    return series


def split_and_normalize(
        series: ts,
        prm: dict,
):
    """
    Splits and normalize a given TimeSeries object, returning both individual
    (now split) data sets and their respectively fitted scalers.
    """

    # Train-test splitting:
    trainval, test = series.split_after(
        prm['TRAIN_SHARE']+prm['VAL_SHARE_OF_DS']
    )
    if prm['VAL_SHARE_OF_DS'] > 0:
        train, val = trainval.split_after(
            1-(prm['VAL_SHARE_OF_DS']/(
                prm['TRAIN_SHARE']+prm['VAL_SHARE_OF_DS']
              ))
            )
    else:
        train = trainval.copy()
        val = []

    if prm['TROUBLESHOOTING_PLOTS']:
        train[prm['TARGET_VAR']].plot(
            label=prm['TARGET_VAR']+' train',
            color='black',
        )
        if len(val) > 0:
            val[prm['TARGET_VAR']].plot(
                label=prm['TARGET_VAR']+' val',
                color='green',
            )
        test[prm['TARGET_VAR']].plot(
            label=prm['TARGET_VAR']+' test',
            color='blue',
        )
        plt.show()

        train[prm['TARGET_VAR']].plot(
            label=prm['TARGET_VAR']+' train',
            color='black',
        )
        plt.show()

        if len(val) > 0:
            val[prm['TARGET_VAR']].plot(
                label=prm['TARGET_VAR']+' val',
                color='green'
            )
            plt.show()

        test[prm['TARGET_VAR']].plot(
            label=prm['TARGET_VAR']+' test',
            color='blue'
        )
        plt.show()

    # Normalize the time series
    transformer = Scaler()  # For whole data set
    transformer_target = Scaler()  # For target series only

    # Transforming ------------------------------------------------------------

    # Fit the target series transformer (no transform):
    transformer_target.fit(train[prm['TARGET_VAR']])

    # Train transformers
    # 1) Fitting transformer to train data set only:
    train_transformed = transformer.fit_transform(train)
    # 2) Extracting target (OSEAX close) series (a speciality of Darts):
    train_target_series = train_transformed[prm['TARGET_VAR']]
    # 3) Extracting other features (covariates) series for train:
    train_past_covariates = train_transformed.drop_columns(prm['TARGET_VAR'])
    # 5) Setting past covariates to None if the are none.
    if len(train_past_covariates.columns) < 1:
        train_past_covariates = None
        print("[i] (Message: No past covariates detected in this dataset)")

    # Validation transformers:
    # 1) Check if val set present:
    if prm['VAL_SHARE_OF_DS'] > 0:
        # 2) Transforming valdiation part of the series:
        val_transformed = transformer.transform(val)
        # 3) Extracting target (OSEAX close) and creating target series:
        val_target_series = val_transformed[prm['TARGET_VAR']]
        # 4) Creating validation past covariates series.
        val_past_covariates = val_transformed.drop_columns(prm['TARGET_VAR'])
        # 5) Setting past covariates to None if the are none.
        if len(val_past_covariates.columns) < 1:
            val_past_covariates = None
    else:
        val_past_covariates = None
        val_target_series = None

    # Test transformers:
    # 1) Transforming test part of series:
    test_transformed = transformer.transform(test)
    # 2) Extracting target (OSEAX close) and creating target series:
    test_target = test_transformed[prm['TARGET_VAR']]
    # 3) Creating test past covariates series:
    test_past_covariates = test_transformed.drop_columns(prm['TARGET_VAR'])
    # 4) Setting past covariates to None if the are none:
    if len(test_past_covariates.columns) < 1:
        test_past_covariates = None

    # Full dataset transformers (for plots and alike):
    # 1) Transforminf full series:
    series_transformed = transformer.transform(series)
    # 2) Transforming full OSEAX/target_variable series:
    target_series_transformed = series_transformed[prm['TARGET_VAR']]
    # 3) Transforming full covariates hist:
    past_covariates_transformed = series_transformed.drop_columns(
        prm['TARGET_VAR']
    )
    # 4) Setting past covariates to None if the are none:
    if len(past_covariates_transformed) < 1:
        past_covariates_transformed = None

    # Done transforming -------------------------------------------------------

    # Saving all data in a dictinary containers:
    data_container = {
        'full': {
            'full': series_transformed,
            'target variable': target_series_transformed,
            'past covariates': past_covariates_transformed,
        },
        'train': {
            'target variable': train_target_series,
            'past covariates': train_past_covariates,
        },
        'validation': {
            'target variable': val_target_series,
            'past covariates': val_past_covariates,
        },
        'test': {
            'target variable': test_target,
            'past covariates': test_past_covariates,
        },
    }
    scalers_container = {
        'scaler': transformer,
        'scaler_target': transformer_target,
    }

    # Setting compatible vars as futcovs and removes them from main dataset:
    outdata_container = convert_or_remove_future_covariates(
        data_container,
        prm=prm
    )

    if prm['PRINT']:
        print(f"[i] Train share      : {len(train)/len(series)} ",
              f"({len(train_target_series)})"
              )
        print(f"[i] Validation share : {len(val)/len(series)} ",
              f"({len(val_target_series)})"
              )
        print(f"[i] Test share       : {len(test)/len(series)} ",
              f"({len(test_target)})"
              )

    return outdata_container, scalers_container


def convert_or_remove_future_covariates(
        dataset: dict,
        prm: dict,
):
    """
    Sets all features capable of being used as future covariates as such.
    This is done by iterating through all features in the main data set,
    placing any future covariate compatible feature in a own series. If the
    model does not support future covariates, they will be removed.

    This function generates the future covatrates series using the passed
    series' DateTimeIndex. Therefore, the whole/full series should be provided
    as indataset.

    The follwoing features can be set to future covariates:
    * 'Year (Year)'
    * 'Month of Year (Month)'
    * 'Day of Month (Day)'
    * 'Week of Year (Week)'
    * 'Day of Week (Day/Week)'
    * 'Holiday Binary Indicator (Holidays)'
    * 'Datetime Value (Datetime)'

    Returns:
    * futcov_splits: dict with all future covariate splits.
    """

    futcov_compatible_features = [
        'Year (Year)',
        'Month of Year (Month)',
        'Day of Month (Day)',
        'Week of Year (Week)',
        'Day of Week (Day/Week)',
        'Holiday Binary Indicator (Holidays)',
        'Datetime Value (Datetime)',
    ]

    full_dataset_series = dataset['full']['full']

    # Iterating through the indatasets features & storing futcovs:
    futcovs_in_ds = []
    for feature in full_dataset_series.columns:
        if feature in futcov_compatible_features:
            futcovs_in_ds.append(feature)
    if prm['PRINT']:
        print("[i] Future covariates detected in dataset:", futcovs_in_ds)

    # Returning (the indataset) if no future covariates detected:
    if not futcovs_in_ds:
        return dataset

    # Initiating future and past covariate dataset series:
    pastcovs = full_dataset_series.copy().drop_columns(futcovs_in_ds)
    futucovs = full_dataset_series.copy().drop_columns(pastcovs.columns)

    # Splitting future covariates:
    trainval, test = futucovs.split_after(
        prm['TRAIN_SHARE']+prm['VAL_SHARE_OF_DS']
    )
    train, val = trainval.split_after(
        1-(prm['VAL_SHARE_OF_DS']/(prm['TRAIN_SHARE']+prm['VAL_SHARE_OF_DS']))
    )
    if prm['PRINT']:
        print("[i] Future covatiate splits stats:")
        print("train :", len(train), train.freq, train.dtype)
        print("val   :", len(val), val.freq, val.dtype)
        print("test  :", len(test), test.freq, test.dtype)
        print("full  :", len(futucovs), futucovs.freq, futucovs.dtype)

    if prm['TAKES_FUTC']:
        # Storing future covariate dataset splits in dictionary:
        dataset['full']['future covariates'] = futucovs
        dataset['train']['future covariates'] = train
        dataset['validation']['future covariates'] = val
        dataset['test']['future covariates'] = test

        # Removing future covariates form past covars:
        dataset = remove_futcovs_from_pastcovs(
            dataset=dataset,
            prm=prm,
            futcov_list=futcov_compatible_features,
        )

    if prm['TROUBLESHOOTING_PLOTS']:
        for dataset_split in dataset.values():
            for dataset_type in dataset_split.values():
                if dataset_type:
                    dataset_type.plot()
                    plt.show()

    return dataset


def remove_futcovs_from_pastcovs(
        dataset: dict,
        prm: dict,
        futcov_list: list,
):
    """
    Removes any duplicate covariates series which may have occured during the
    get_futcov process from the past covariates data set, (If the function
    have added DatetimeIndex features present in the standard data set).
    """

    # Defining all the keys that are not scalers or target series in the
    split_keys = ['full', 'train', 'validation', 'test']
    pcvs_key = 'past covariates'

    # Extracting futcov features from one of the subdatasets (all same):
    features_main = (dataset['full']['full'].columns).tolist()

    common_features = [feat for feat in features_main if feat in futcov_list]

    if any(common_features):
        if prm['PRINT']:
            print(
                "[i] Fut- and pastcov common features removed from pastcovs:",
                common_features,
            )

        # Dropping common features (columns by feature name):
        for split in split_keys:
            dataset[split][pcvs_key] = dataset[split][pcvs_key].drop_columns(
                common_features
            )
            if len(dataset[split][pcvs_key].columns) <= 0:
                dataset[split][pcvs_key] = None

            # print("After:\n", dataset[subds_key].columns)

    # For testing:
    # features_main_t = (dataset['full'].columns).tolist()
    # features_futc_t = (futur_cov_dict['full'].columns).tolist()
    # common_feat_test = [
    #     feat for feat in features_main_t if feat in features_futc_t
    # ]
    # print(common_feat_test)

    return dataset


def remove_highly_correlated_features(
        data: pd.DataFrame,
        log_file_path: str,
        extra_info: str = '',
        threshold: float = 0.9,
):
    """Removing Highly Correlated Features"""

    corr_matrix = data.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
    )

    to_drop = [
        column for column in upper_triangle.columns if any(
            upper_triangle[column] > threshold
        )
    ]

    # Logging:
    with open(log_file_path, "a", encoding='utf-8-sig',) as file:
        file.write(f"Removed features {extra_info}: (corr >= {threshold})\n")
        for feature in to_drop:
            file.write(f"   {feature}\n")
        file.write("\n")

    return pd.DataFrame(data).drop(columns=to_drop)


def random_forest_feature_selection(
        dataset: pd.DataFrame,
        horizon: int,
        prm: dict,
        mode: str = 'all model selected or 3 best',
):
    """Generates Random Forest datasets for each horizon instance"""

    # Setting and generating forlders for storing data & info:
    info_out_dir = 'results/.analytics/feature_selection/'
    data_output_dir = f'{prm["OUTPUT_DIR"]}h_{str(horizon)}/'
    make_output_dir(info_out_dir)
    make_output_dir(data_output_dir)
    info_out_file = f'{info_out_dir}feature_selection_stats_h_{horizon}.txt'

    # Saving info to file (& printing if desired):
    with open(info_out_file, "w", encoding='utf-8-sig') as file:
        file.write(f"Selecting features for the horizon={horizon} ")
        file.write("dataset:\n\n")

    # Preprocessing:

    # Split the dataset into train and test
    data_train, _ = train_test_split(
        dataset.copy(),
        train_size=prm['TRAIN_SHARE'],
        shuffle=False,
    )

    # Scaling the train dataset
    data_train_scaled_np = MinMaxScaler().fit_transform(
        data_train.copy(),
    )
    data_train_scaled = pd.DataFrame(  # np back to df
        data_train_scaled_np,
        index=data_train.index,
        columns=data_train.columns,
    )

    # Setting target variable, shifting, and handle data lengths and NaNs:
    target = data_train_scaled[prm['TARGET_VAR']].copy()
    target_shifted = target.shift(horizon)
    target_shifted.drop(target_shifted.head(horizon).index, inplace=True)

    # Building dataset versions:

    # Dataset 1.1: all features
    data_full = data_train_scaled.drop(
        data_train_scaled.copy().head(horizon).index,  # Cut to = len y
    )

    # Dataset 1.2: removing target series from X:
    data_excl_y = data_full.copy().drop(
        columns=[prm['TARGET_VAR']],
    )

    # Dataset 2.1: removing features correlating more than 0.9:
    data_nocorr = remove_highly_correlated_features(
        data=data_full.copy(),
        log_file_path=info_out_file,
        extra_info="for full dataset INCLUDING OSEAX-target",
        threshold=0.9,
    )

    # Dataset 2.2: same as 2.1, but without the target series:
    data_nocorr_excl_y = remove_highly_correlated_features(
        data_excl_y.copy(),
        log_file_path=info_out_file,
        extra_info="for full datasett EXCLUDING OSEAX-target",
        threshold=0.9,
    )

    # Build models and fit:

    # Training RF feature selection - including target:
    selected_features, feature_scores = select_features_rf(
        dataset=data_full,
        dataset_name='Full dataset',
        target_series=target_shifted,
        info_outdata_file=info_out_file,
    )

    # Training RF feature selection - without target:
    select_features_rf(
        dataset=data_excl_y,
        dataset_name='Dataset excluding target',
        target_series=target_shifted,
        info_outdata_file=info_out_file,
    )

    # Training RF feature selection - with correlated features removed:
    select_features_rf(
        dataset=data_nocorr,
        dataset_name='Dataset excluding high-correlated features',
        target_series=target_shifted,
        info_outdata_file=info_out_file,
    )

    # Training RF feature selection - correlated removed & without target:
    select_features_rf(
        dataset=data_nocorr_excl_y,
        dataset_name='Dataset excluding target & high-correlated features',
        target_series=target_shifted,
        info_outdata_file=info_out_file,
    )

    # Selecting features:

    final_selected_features = []

    # Dropping target & dt from selected features (will be re-included later):
    selected_features_uniques = selected_features.copy()
    if prm['TARGET_VAR'] in selected_features_uniques:            # Rm target
        selected_features_uniques.remove(prm['TARGET_VAR'])
    if 'Datetime Value (Datetime)' in selected_features_uniques:  # Rm dt
        selected_features_uniques.remove('Datetime Value (Datetime)')

    # Doing the same for feature importance rankings:    
    feature_scores_uniques = feature_scores.copy().drop(
        index=[
            prm['TARGET_VAR'],
            'Datetime Value (Datetime)',
        ]
    )
    top_3_uniques = list(feature_scores_uniques.head(3).index)
    top_7_uniques = list(feature_scores_uniques.head(7).index)

    # print("selected_features_uniques :", selected_features_uniques, type(selected_features_uniques))
    # print("selected_features         :", selected_features, type(selected_features))
    # exit()

    # Selecting features according to mode:
    if mode == 'all model selected or 3 best (excl target and dt)':
        if len(top_3_uniques) < len(selected_features_uniques):
            final_selected_features = selected_features_uniques
        else:
            final_selected_features = top_3_uniques
    elif mode == '7 best (excl target and dt)':
        final_selected_features = top_7_uniques
    else:
        raise ValueError(
            f"Unknown feature selection mode passed: {mode}"
        )

    # Adding final info to file:
    with open(info_out_file, "a", encoding='utf-8-sig') as file:
        file.write(f"Feature selection results (horizon={horizon}):\n")
        file.write("   Selected by model:\n")
        for feature in selected_features:
            file.write(f"      {feature}\n")
        file.write("   Top 3 ranked non-target, non-Datetime features\n")
        for feature in top_3_uniques:
            file.write(f"      {feature}\n")
        file.write("   Top 7 ranked non-target, non-Datetime features\n")
        for feature in top_7_uniques:
            file.write(f"      {feature}\n")
        file.write("\n")
        file.write("dataset:\n\n")

    # Adding target variable and date value (totaling 5 or 9 features):
    final_selected_features.insert(0, prm['TARGET_VAR'])
    final_selected_features.append('Datetime Value (Datetime)')

    return final_selected_features


def select_features_rf(
        dataset: pd.DataFrame,
        dataset_name: str,
        target_series: pd.Series,
        info_outdata_file: str,
):
    """
    Selects festures from the gives data set using a Random Forest Regressor
    within a SelectFromModel object.
    """

    # Building selector (meta transformer) utilizing RF feature importance:
    selector = SelectFromModel(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=6,
            verbose=0,
        )
    )
    # n_estimators is set to 100. 1000 was considered, but:
    # """
    # The official page of the algorithm states that random forest does not
    # overfit, and you can use as much trees as you want. But Mark R. Segal
    # (April 14 2004. "Machine Learning Benchmarks and Random Forest
    # Regression." Center for Bioinformatics & Molecular Biostatistics) has
    # found that it overfits for some noisy datasets.
    # """
    # https://stats.stackexchange.com/questions/36165

    # Fitting selector:
    selector = selector.fit(
        X=dataset,
        y=target_series,
    )

    # Getting results:
    seleted_features = selector.get_feature_names_out()
    n_selected = len(seleted_features)

    # Getting feature scores/importances:
    feature_scores = pd.Series(
        selector.estimator_.feature_importances_,
        index=dataset.columns
    ).sort_values(ascending=False)

    # Saving scores:
    max_index_length = max(
        [len(str(index)) for index in feature_scores.index]
    )
    with open(info_outdata_file, "a", encoding='utf-8-sig') as file:
        # file.write("Feature selection results:\n\n")
        file.write(f"{dataset_name} - features selected: ")
        file.write(f"{n_selected}\n")
        for feature in seleted_features:
            file.write(f"   {feature}\n")

        # Feature importances:
        file.write("Feature importances:\n")
        for index, value in feature_scores.items():
            formatted_index = str(index).ljust(max_index_length)
            file.write(f"   {formatted_index}: {round(value, 8)}\n")
        file.write("\n")

    seleted_features = seleted_features.tolist()

    return seleted_features, feature_scores
