import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scripts.helpfuncs_models import make_output_dir
from scripts.helpfuncs_plots import set_size

# Thesis standard plot syle:

style = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{newcent}',
    'font.family': 'serif',
    'font.serif': ['New Century Schoolbook'],
    'font.size': 12,

    'axes.titlesize': 12,   # OK
    "axes.labelsize": 12,   # OK
    "legend.fontsize": 12,  # Not tested
    "xtick.labelsize": 12,  # OK
    "ytick.labelsize": 11,  # OK
    # 'ytick.major.fontname': 'New Century Schoolbook',
    # 'ytick.minor.fontname': 'New Century Schoolbook',

    "xtick.bottom": True,
    "ytick.left": True,
}
sns.set(rc=style)
sns.set_style("whitegrid", {'axes.grid': True})  # ,{'axes.grid' : True})
plt.rcParams.update(style)


LATEX_DOC_TEXT_WIDTH = 418.25368
# FONT_FAMILY = 'Serif'
# FONT_SIZE = 11
# FONT_SIZE_WEIGHT = 'normal'
# AXES_LABEL_SIZE = 9
# AXES_LABEL_WEIGHT = 'normal'
# TICK_LABELS_SIZE = 9
# TICK_LABELS_WEIGHT = 'normal'
# TITLE_WEIGHT = 'normal'


def lineplot_forecasts(
        horizon: int,
        local_results_dir: str,
        # data,
        # save_path = str,
        # save_name = str,
        title: str = 'Average Forecast with Error Band',
        xlabel: str = 'Time',
        ylabel: str = 'Forecast Value',
        grid = True,
        # color = 'b',
):
    out_dir_name = 'forcast_lineplots/'
    rootdir = 'C:/Users/ulle_/OneDrive/Skolearbeid/IRISMaster/Masteroppgave/Kode/Hovedsystem/Mainsys_2/'
    save_path = 'results/.analytics/results_analytics/' + out_dir_name
    resultsdir = rootdir+local_results_dir
    make_output_dir(save_path)

    # Looping through every result dir for each experiment to get scores:
    for _, model_dirs, _ in os.walk(resultsdir):
        for model_name in model_dirs:
           
            models_dataset_dir = f"{resultsdir}{model_name}/h_{horizon}/"
            # print(models_dataset_dir)

            for _, dataset_dirs, _ in os.walk(models_dataset_dir):
                for dataset_name in dataset_dirs:
                
                    data_dir = models_dataset_dir + dataset_name

                    # # Getting mean forecasts:
                    # avg_forecast = pd.read_csv(
                    #     data_dir + '/forecast_mean.csv',
                    #     parse_dates=['time'],
                    #     delimiter=';',
                    #     encoding='utf-8-sig',
                    #     index_col='time'
                    # )
                    # avg_forecast.rename(columns={"0": "Average"}, inplace=True)
                    # avg_forecast.index.names = ['Time']

                    # Getting individual forecasts:
                    individual_fct = pd.read_csv(
                        data_dir + '/forecasts_individual.csv',
                        parse_dates=['time'],
                        delimiter=';',
                        encoding='utf-8-sig',
                        index_col='time'
                    )
                    individual_fct.index.names = ['Time']
                    # individual_fct.rename(columns={'time': 'Time'}, inplace=True)
                    # print(individual_fct.info())

                    # Getting true targer values:
                    true_oseax_full = pd.read_csv(
                        'dataset/dataset_full_half_preprocessed_info.csv',
                        usecols=['Date', 'Oslo Børs all-share index (OSEAX)'],
                        parse_dates=['Date'],
                        delimiter=';',
                        encoding='utf-8-sig',
                        index_col='Date'
                    )
                    true_oseax_full.rename(columns={'Oslo Børs all-share index (OSEAX)': 'True'}, inplace=True)
                    true_oseax_full.index.names = ['Time']
                    # individual_forecasts_timeidx = individual_forecasts.set_index('Time')
                    # print(individual_forecasts_timeidx.index)
                    # true_oseax_train = true_oseax_full.loc[individual_forecasts_timeidx.index]

                    # Focus on 2020 feb-april covid crash period:
                    start_c19crash = pd.Timestamp('2020-02-14')
                    end_c19crash = pd.Timestamp('2020-04-16')
                    individual_ccrash = individual_fct[start_c19crash:end_c19crash]
                    true_ccrash = true_oseax_full[start_c19crash:end_c19crash]

                    plt.figure(figsize=set_size(   #=(12, 6))
                        LATEX_DOC_TEXT_WIDTH, #replaced with 390.0 due do better fit.
                        subplots=(1, 1),
                        # height_mult=0.75,
                        # height_mult=0.50,
                        height_mult=.60,
                    ))
                    individual_ccrash_long = individual_ccrash.reset_index().melt(id_vars=['Time'], var_name='Model', value_name='Forecast')

                    # Adding the error band using standard deviation of individual forecasts
                    sns.lineplot(
                        data=individual_ccrash_long, x='Time', y='Forecast', estimator='mean', errorbar='sd', label='Forecast',
                        # color='blue',
                        alpha=1
                    )#, linewidth=.75)
                    sns.lineplot(
                        data=true_ccrash, x='Time', y='True', label='True',
                        # color='gold',
                        # alpha=.7, 
                    )#, linewidth=.75)

                    # Set date format on x-axis
                    ax = plt.gca()  # Get the current Axes instance on the current figure
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))  # Format the dates as 'month-day'
                    plt.xticks(rotation=45)  # Optional: Rotate the x-axis labels to improve spacing

                    # Setting plot labels and title
                    legend = plt.legend(loc='upper right', frameon=True) # customize legend.
                    plt.setp(legend.get_title(), fontsize=11)  # Set the fontsize of the legend title
                    plt.setp(legend.get_texts(), fontsize=11)   # Set the fontsize of the legend labels
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.grid(grid)
                    
                    # save_file = f"{save_path}{save_name}"
                    save_file = f"{save_path}{model_name}-{dataset_name}.svg"
                    plt.savefig(
                        save_file,
                        format='svg',
                        dpi=300,
                        bbox_inches='tight'
                    )
                    # print(save_file);sys.exit()
                    # sys.exit()

                    # if dataset_name == "selected":
                    #     dataset_name = "selected"
                    # elif dataset_name == "selected_rf_strict":
                    #     dataset_name = "RF strict"
                    # elif dataset_name == "selected_rf_top_7":
                    #     dataset_name = "RF top-7"
                    # elif dataset_name == "univariate":
                    #     dataset_name = "univariate"
                    # else:
                    #     raise ValueError(f"Unknown dataset naming: {dataset_name}")

                break
        break
        # Breaking to avoid other dirs than the most immediate to be looped.


def lineplot_figure_train_test_val(
        data = pd.DataFrame,
        save_path = str,
        save_name = str,
        xlabel = str,
        ylabel = str,
        color = 'b',
):
    make_output_dir(save_path)

    split_df = split_series_into_train_val_test(data_series=data)

     # Define plotting region
    _, axes = plt.subplots(1, 1, figsize=set_size(
        390.0,  # LATEX_DOC_TEXT_WIDTH replaced with 390.0 due do better fit.
        subplots=(1, 1),
        # height_mult=1.70,  # STANDARD
        # height_mult=1.50,
    ))
    sns.lineplot(
        data=split_df,
        y='value',
        x='Date',
        ax=axes,
        color=color,
        hue='split',
        linewidth=.75,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title('Bar lot of MAPE by Model-Tuning-Dataset')
    legend = plt.legend(title='Dataset splits', loc='upper left', frameon=True) # customize legend.
    plt.setp(legend.get_title(), fontsize=11)  # Set the fontsize of the legend title
    plt.setp(legend.get_texts(), fontsize=11)   # Set the fontsize of the legend labels

    # Disable the grid
    axes.grid(False)

    # Add vertical lines for train/validation and validation/test splits
    # Assuming that 'Date' is a datetime index and split_df has a 'split' column
    train_max_date = split_df[split_df['split'] == 'Train'].index.max()
    test_min_date = split_df[split_df['split'] == 'Test'].index.min()

    plt.axvline(x=train_max_date, color='grey', linestyle='--', lw=1)
    plt.axvline(x=test_min_date, color='grey', linestyle='--', lw=1)

    save_file = f"{save_path}{save_name}"
    plt.savefig(
        save_file,
        format='svg',
        dpi=300,
        bbox_inches='tight'
    )
    # print(split_df);sys.exit()


def split_series_into_train_val_test(data_series, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    assert train_frac + val_frac + test_frac == 1, "Fractions must add up to 1."

    # Calculate the indices for the end of each split
    train_end = int(len(data_series) * train_frac)
    val_end = train_end + int(len(data_series) * val_frac)

    # Create a new DataFrame to store the series with the split column
    split_df = pd.DataFrame({'value': data_series})
    split_df['split'] = 'Test'  # Default to 'test'
    split_df.iloc[:val_end, split_df.columns.get_loc('split')] = 'Validation'
    split_df.iloc[:train_end, split_df.columns.get_loc('split')] = 'Train'

    return split_df


def lineplot_figure_multi(
        data,
        save_path = str,
        save_name = str,
        xlabel = str,
        ylabel = str,
        color = 'b',
        grid = True,
):
    make_output_dir(save_path)

     # Define plotting region
    _, axes = plt.subplots(1, 1, figsize=set_size(
        390.0,  # LATEX_DOC_TEXT_WIDTH replaced with 390.0 due do better fit.
        subplots=(1, 1),
        # height_mult=1.70,  # STANDARD
        # height_mult=1.50,
    ))

    sns.lineplot(
        data=data,
        ax=axes,
        color=color,
    )
    # plt.xlabel('Year')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    axes.grid(grid)

    # TEMP: COMMENT INN WHEN NEEDED:
    legend = plt.legend(title='Data', loc='upper left', frameon=True) # customize legend.
    plt.setp(legend.get_title(), fontsize=11)  # Set the fontsize of the legend title
    plt.setp(legend.get_texts(), fontsize=11)   # Set the fontsize of the legend labels

    save_file = f"{save_path}{save_name}"
    plt.savefig(
        save_file,
        format='svg',
        dpi=300,
        bbox_inches='tight'
    )


def lineplot_figure(
        data = pd.DataFrame,
        save_path = str,
        save_name = str,
        xlabel = str,
        ylabel = str,
        color = 'b',
        grid = True,
):
    make_output_dir(save_path)

     # Define plotting region
    _, axes = plt.subplots(1, 1, figsize=set_size(
        390.0,  # LATEX_DOC_TEXT_WIDTH replaced with 390.0 due do better fit.
        subplots=(1, 1),
        # height_mult=1.70,  # STANDARD
        # height_mult=1.50,
    ))
    sns.lineplot(
        data=data,
        ax=axes,
        color=color,
        linewidth=.75
    )
    # plt.xlabel('Year')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    axes.grid(grid)

    save_file = f"{save_path}{save_name}"
    plt.savefig(
        save_file,
        format='svg',
        dpi=300,
        bbox_inches='tight'
    )


def barplot_all_models(
        models: str,
        horizon: int,
        local_results_dir: str,
        # desired_ticks: int = 6,
        # show_outliers: bool = False,
):
    out_dir_name = 'metrics_barplot_all_MAPEs/'
    rootdir = 'C:/Users/ulle_/OneDrive/Skolearbeid/IRISMaster/Masteroppgave/Kode/Hovedsystem/Mainsys_2/'
    save_path = 'results/.analytics/results_analytics/' + out_dir_name
    # model_dir = f"{model}/h_{horizon}/"
    resultsdir = rootdir+local_results_dir
    scores_list = []
    make_output_dir(save_path)

    # Looping through every result dir for each experiment to get scores:
    for _, model_dirs, _ in os.walk(resultsdir):
        for model_name in model_dirs:
           
            models_dataset_dir = f"{resultsdir}{model_name}/h_{horizon}/"
            # print(models_dataset_dir)

            for _, dataset_dirs, _ in os.walk(models_dataset_dir):
                for dataset_name in dataset_dirs:
                
                    data_dir = models_dataset_dir + dataset_name
                    data_file = data_dir + '/scores_individual.csv'
                    # print(data_file)

                    scores_df = pd.read_csv(
                        data_file,
                        delimiter=';',
                        encoding='utf-8-sig',
                    )

                    # Cleaning up naming and transposing:
                    scores_df.rename(columns={"Unnamed: 0": ""}, inplace=True)
                    scores_df = scores_df.T

                    # Setting upper row as column names:
                    scores_df.columns = scores_df.iloc[0]       # First row as colnames
                    scores_df.drop(scores_df.index[0], inplace=True)  # Rm above-said

                    # Resetting index and cleaning up naming:
                    scores_df.reset_index(inplace=True)
                    scores_df.rename(columns={"index": "Model"}, inplace=True)

                    if dataset_name == "selected":
                        dataset_name = "selected"
                    elif dataset_name == "selected_rf_strict":
                        dataset_name = "RF strict"
                    elif dataset_name == "selected_rf_top_7":
                        dataset_name = "RF top-7"
                    elif dataset_name == "univariate":
                        dataset_name = "univariate"
                    else:
                        raise ValueError(f"Unknown dataset naming: {dataset_name}")

                    model_tuning = model_name.split('-')
                    model = model_tuning[0].upper()
                    tuning = model_tuning[1]
                    model_tuning_dataset = f"{model} {tuning} {dataset_name}"
                    scores_df.insert(0, 'model-tuning-dataset', model_tuning_dataset)
                    scores_list.append(scores_df)
                    # print(scores_df); sys.exit()
                break
        break
        # Breaking to avoid other dirs than the most immediate to be looped.

    # Concatinating (adding vertically) every score df for each experiment:
    all_scores = pd.concat(scores_list, ignore_index=True)
    
    # Define the data for RWM and RWD models
    deterministic_models = pd.DataFrame({
        'model-tuning-dataset': ['RWM univariate', 'RWD univariate'],
        'Model': ['RWM', 'RWD'],
        'RMSE': [12.37569808959961, 12.373608589172363],
        'MAPE': [0.8071581833064556, 0.8073697797954082],
        'R2_score': [0.9953216034919024, 0.9953231834806502],
        'sMAPE': [0.8062105625867844, 0.8060645312070847],
        'MSE': [153.15789794921875, 153.10618591308594],
        'epochs': [1, 1],
        'time': ['0 days 00:00:20.293125', '0 days 00:00:20.171335'],
        'model_type': ['RW', 'RW']  # New color category for these models
    })

    # Append the deterministic model data to the all_scores DataFrame
    all_scores = pd.concat([all_scores, deterministic_models], ignore_index=True)


    # Calculate mean and standard deviation of MAPE for each 'model-tuning-dataset'
    mape_stats = all_scores.groupby('model-tuning-dataset')['MAPE'].agg(['mean', 'std']).reset_index()
    # Sort the 'model-tuning-dataset' by mean MAPE in ascending order
    mape_stats_sorted = mape_stats.sort_values('mean')
    # Use the sorted order for plotting
    sorted_order = mape_stats_sorted['model-tuning-dataset'].tolist()

    # # Add a new column to indicate the model type (LSTM or TFT)
    # all_scores['model_type'] = all_scores['model-tuning-dataset'].apply(lambda x: 'LSTM' if 'LSTM' in x else 'TFT')

    # Reassign 'model_type' to include "LSTM", "TFT", and "RW"
    all_scores['model_type'] = all_scores['model-tuning-dataset'].apply(
        lambda x: 'LSTM' if 'LSTM' in x else ('TFT' if 'TFT' in x else 'RW')
    )
    # print(all_scores);sys.exit()

    #  # Define plotting region
    # _, axes = plt.subplots(1, 1, figsize=set_size(
    #     390.0,  # LATEX_DOC_TEXT_WIDTH replaced with 390.0 due do better fit.
    #     subplots=(1, 1),
    #     # height_mult=1.70,  # STANDARD
    #     height_mult=1.50,
    # ))
    _, axes = plt.subplots(1, 1, figsize=(4.5, 5.75))
    # Now plot with the sorted order and include standard deviation
    # sns.set_palette('colorblind')
    sns.barplot(
        data=all_scores,
        x='MAPE',
        y='model-tuning-dataset',
        order=sorted_order,  # Order the bars by the sorted categories
        errorbar='sd',             # Show the standard deviation
        capsize=.2,          # Caps on the error bars
        ax=axes,

        hue='model_type',      # Differentiate by model type
        dodge=False,           # Do not separate the bars by hue
        alpha=.9,
        # palette=['#1f77b4', '#ff7f0e'],  # Set specific colors for LSTM and TFT
        # color="lb",
    )
    # Set labels and titles as needed
    plt.xlabel('MAPE')
    plt.ylabel('Model, tuning, and dataset')
    # plt.title('Bar lot of MAPE by Model-Tuning-Dataset')
    plt.legend(title='Model type', loc='upper right', frameon=True) # customize legend.

    plt.savefig(
        save_path+'bar_plot_mape_all_models.svg',
        format='svg',
        dpi=300,
        bbox_inches='tight'
    )


def boxplot_4_metrics(
        model: str,
        horizon: int,
        local_results_dir: str,
        desired_ticks: int = 6,
        show_outliers: bool = False,
):
    """
    Loop through the results dir. For each model found there; plots four box
    plots of MAPE, RMSE, R2 and Mean Runtime
    """

    out_dir_name = 'metrics_individual_4x_boxplot/'
    rootdir = 'C:/Users/ulle_/OneDrive/Skolearbeid/IRISMaster/Masteroppgave/Kode/Hovedsystem/Mainsys_2/'
    save_path = 'results/.analytics/results_analytics/' + out_dir_name

    model_dir = f"{model}/h_{horizon}/"
    scores_list = []
    make_output_dir(save_path)

    # Looping through every result dir for each experiment to get scores:
    resultsdir = rootdir+local_results_dir+model_dir
    for _, dataset_dirs, _ in os.walk(resultsdir):
        for dataset_dir_name in dataset_dirs:

            model_outdata_dir = local_results_dir + model_dir
            model_outdata_dir += dataset_dir_name + '/'
            # print(model_outdata_dir)

            scores_df = pd.read_csv(
                model_outdata_dir + 'scores_individual.csv',
                delimiter=';',
                encoding='utf-8-sig',
            )

            # Cleaning up naming and transposing:
            scores_df.rename(columns={"Unnamed: 0": ""}, inplace=True)
            scores_df = scores_df.T

            # Setting upper row as column names:
            scores_df.columns = scores_df.iloc[0]       # First row as colnames
            scores_df.drop(scores_df.index[0], inplace=True)  # Rm above-said

            # Resetting index and cleaning up naming:
            scores_df.reset_index(inplace=True)
            scores_df.rename(columns={"index": "Model"}, inplace=True)

            if dataset_dir_name == "selected":
                dataset_dir_name = "Selected"
            elif dataset_dir_name == "selected_rf_strict":
                dataset_dir_name = "RF Strict"
            elif dataset_dir_name == "selected_rf_top_7":
                dataset_dir_name = "RF Top-7"
            elif dataset_dir_name == "univariate":
                dataset_dir_name = "Univariate"
            else:
                raise ValueError(f"Unknown dataset naming: {dataset_dir_name}")

            scores_df.insert(0, 'Dataset', dataset_dir_name)
            scores_list.append(scores_df)

            # Modifying df to fit bar plotting and concating accumulation-df:
            # # scores_df = scores_df.set_index("Metric")
            # # scores_df.drop("Unnamed: 0", axis=1, inplace=True)
            # scores_df.insert(0, 'model', model_dir_name)
        break
        # Breaking to avoid other dirs than the most immediate to be looped.

    # Concatinating (adding vertically) every score df for each experiment:
    all_scores = pd.concat(scores_list, ignore_index=True)
    all_scores['Dataset'].replace('RF Strict', 'RF strict', inplace=True)
    all_scores['Dataset'].replace('RF Top-7', 'RF top-7', inplace=True)

    # Define plotting region (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=set_size(
        390.0,  # LATEX_DOC_TEXT_WIDTH replaced with 390.0 due do better fit.
        subplots=(2, 2),
        height_mult=1.70,  # STANDARD
        # height_mult=1.10,
    ))

    # Managing each individual sub-boxplot:

    # MAPE:
    mape = sns.boxplot(
        data=all_scores,
        x='Dataset', y='MAPE',
        order=['Univariate', 'RF strict', 'RF top-7', 'Selected'],
        showfliers=show_outliers,
        ax=axes[0, 0],
        boxprops=dict(alpha=.8),
    )
    mape.set_title(r'MAPE', fontname='New Century Schoolbook')  # , weight=TITLE_WEIGHT
    mape.yaxis.set_major_locator(MaxNLocator(nbins=desired_ticks))
    mape.set(xlabel=None)
    mape.set_ylabel(r"Percentage", fontname='New Century Schoolbook')  # , weight=AXES_LABEL_WEIGHT
    mape.set_xticklabels(
        mape.get_xticklabels(),
        # rotation=45,
        rotation=30,  # STD
        # rotation=22.5,
        horizontalalignment='right',
    )

    # RMSE:
    rmse = sns.boxplot(
        data=all_scores,
        x='Dataset', y='RMSE',
        order=['Univariate', 'RF strict', 'RF top-7', 'Selected'],
        showfliers=show_outliers,
        ax=axes[0, 1],
        boxprops=dict(alpha=.8),
    )
    rmse.set_title('RMSE', fontname='New Century Schoolbook')  # , weight=TITLE_WEIGHT
    rmse.yaxis.set_major_locator(MaxNLocator(nbins=desired_ticks))
    rmse.set(xlabel=None)
    rmse.set_ylabel("Index points")  # , weight=AXES_LABEL_WEIGHT
    rmse.set_xticklabels(
        rmse.get_xticklabels(),
        # rotation=45,
        rotation=30,  # STD
        # rotation=22.5,
        horizontalalignment='right',
        fontname='New Century Schoolbook',
    )

    # R2-value:
    r2sc = sns.boxplot(
        data=all_scores,
        x='Dataset', y='R2_score',
        order=['Univariate', 'RF strict', 'RF top-7', 'Selected'],
        showfliers=show_outliers,
        ax=axes[1, 0],
        boxprops=dict(alpha=.8),
    )
    r2sc.set_title('R$^2$')  # , weight=TITLE_WEIGHT
    r2sc.yaxis.set_major_locator(MaxNLocator(nbins=desired_ticks))
    r2sc.set(xlabel=None)
    r2sc.set_ylabel("R$^2$ value")  # , weight=AXES_LABEL_WEIGHT
    r2sc.set_xticklabels(
        r2sc.get_xticklabels(),
        # rotation=45,
        rotation=30,  # STD
        # rotation=22.5,
        horizontalalignment='right',
    )

    # Time expenditure per model:
    time = sns.boxplot(
        data=all_scores,
        x='Dataset', y='time',
        order=['Univariate', 'RF strict', 'RF top-7', 'Selected'],
        showfliers=show_outliers,
        ax=axes[1, 1],
        boxprops=dict(alpha=.8),
    )
    time.set_title('Mean Model Runtime')  # , weight=TITLE_WEIGHT
    time.yaxis.set_major_locator(MaxNLocator(nbins=desired_ticks))
    time.set(xlabel=None)
    time.set_ylabel("Seconds")  # , weight=AXES_LABEL_WEIGHT
    time.set_xticklabels(
        time.get_xticklabels(),
        # rotation=45,
        rotation=30,  # STD
        # rotation=22.5,
        horizontalalignment='right',
    )

    # Adjust the spines for all subplots to make them thinner
    for axis in axes.flatten():
        axis.spines['top'].set_linewidth(0.1)
        axis.spines['right'].set_linewidth(0.1)
        axis.spines['bottom'].set_linewidth(0.1)
        axis.spines['left'].set_linewidth(0.1)

    model_name = model_dir.split('/', 1)[0]
    fig.tight_layout()  # Preventing overlap

    plt.savefig(
        save_path+model_name+'.svg',
        format='svg',
        dpi=300,
        bbox_inches='tight'
    )

    return


def lineplot_forecast_vs_true_testset(
        model: str,
        horizon: int,
        local_results_dir: str,
        latex_doc_text_widt: float = LATEX_DOC_TEXT_WIDTH,
):
    """
    Loops through the results directory. For each model found there, it plots
    a lineplot of the models forecast (compared to true values) and saves it.
    """

    target = 'Oslo Børs all-share index (OSEAX)'
    out_dir_name = 'forecast_lineplots/'
    rootdir = 'C:/Users/ulle_/OneDrive/Skolearbeid/IRISMaster/Masteroppgave/Kode/Hovedsystem/Mainsys_2/'
    save_path = 'results/.analytics/results_analytics/' + out_dir_name

    model_dir = f"{model}/h_{horizon}/"
    make_output_dir(save_path)

    # Looping through every result dir for each experiment to get scores:
    resultsdir = rootdir+local_results_dir+model_dir
    for _, dataset_dirs, _ in os.walk(resultsdir):
        for dataset_dir_name in dataset_dirs:

            model_outdata_dir = local_results_dir + model_dir
            model_outdata_dir += dataset_dir_name + '/'

            # Getting mean forecasts:
            mean_forecast = pd.read_csv(
                model_outdata_dir + 'forecast_mean.csv',
                parse_dates=['time'],
                delimiter=';',
                encoding='utf-8-sig',
                index_col='time'
            )
            mean_forecast.rename(columns={"0": "Mean forecast"}, inplace=True)
            mean_forecast.index.names = ['Time']

            # Getting individual forecasts:
            individual_forecasts = pd.read_csv(
                model_outdata_dir + 'forecasts_individual.csv',
                parse_dates=['time'],
                delimiter=';',
                encoding='utf-8-sig',
                # index_col='time'
            )
            individual_forecasts.rename(columns={'time': 'Time'}, inplace=True)
            individual_forecasts = individual_forecasts
            # individual_forecasts.index.names = ['Time']
            # print(individual_forecasts.info())

            # Melting df so that it can be plotted with error bands
            # print(individual_forecasts)
            individual_forecasts_melted = pd.melt(
                individual_forecasts,
                id_vars=['Time'],
                var_name='Model',
                value_name='Value'
            )
            # individual_forecasts_melted.insert(1, 'Data', 'Forecast')

            print(individual_forecasts_melted)

            # Getting true targer values:
            true_oseax_full = pd.read_csv(
                'dataset/dataset_full_half_preprocessed_info.csv',
                usecols=['Date', target],
                parse_dates=['Date'],
                delimiter=';',
                encoding='utf-8-sig',
                index_col='Date'
            )
            true_oseax_full.rename(columns={target: "Value"}, inplace=True)
            true_oseax_full.index.names = ['Time']
            individual_forecasts_timeidx = individual_forecasts.set_index('Time')
            print(individual_forecasts_timeidx.index)
            true_oseax_train = true_oseax_full.loc[individual_forecasts_timeidx.index]
            # true_oseax_train = true_oseax_full.loc(individual_forecasts_timeidx.index)

            print(true_oseax_full)

            # Joining mean forecast data over the forecasts axis:
            data_mean_forecast = mean_forecast.join(true_oseax_full)
            # data_individual_forecasts_melted = individual_forecasts_melted.join(true_oseax_full, on='Time')
            # print(data_individual_forecasts_melted); return

            # Define plotting region (single figure - 1 row, 1 column)
            figure, axes = plt.subplots(figsize=set_size(
                latex_doc_text_widt,
                subplots=(1, 1),
                # height_mult=1.70,  # STANDARD
                # height_mult=1.10,
            ))

            # sns.set_style("whitegrid", {'axes.grid': True})  # ,{'axes.grid' : True})
            # sns.set_style("darkgrid", {'axes.grid': True})  # ,{'axes.grid' : True})

            # # Initiating lineplot (average forecast vs true):
            # sns.lineplot(
            #     data=data_mean_forecast,
            #     linewidth=1.0,
            #     ax=axes,
            # )
            # axes.set(ylabel="OSEAX valuation")

            # Initiating lineplot with errorbands (sd):
            sns.lineplot(
                data=individual_forecasts_melted,
                y='Value',
                x='Time',
                # hue='Data',
                linewidth=1.0,
                ax=axes,
            )
            axes.set(ylabel="OSEAX valuation")

            # Adding target value:
            axes.plot(true_oseax_full.index, true_oseax_full['Value'], label='Your Label', color='r')

            plt.savefig(
                save_path+'testset_mean_errorb_'+model+'.svg',
                format='svg',
                dpi=300,
                bbox_inches='tight',
            )
            # plt.clf() TODO: INCLUDE

            return  # TODO: REMOVE
        break
        # Breaking to avoid other dirs than the most immediate to be looped.


def horizontal_boxplot_4_metrics(
        model: str,
        horizon: int,
        local_results_dir: str,
        height_multiplication: float,
        # desired_ticks: int = 6,
        # show_outliers: bool = True,
):
    """
    Loop through the results dir. For each model found there; plots four
    horizontal box plots of MAPE, RMSE, R2 and Mean Runtime.
    """

    out_dir_name = 'metrics_individual_4x_horizontal_boxplot/'
    rootdir = 'C:/Users/ulle_/OneDrive/Skolearbeid/IRISMaster/Masteroppgave/Kode/Hovedsystem/Mainsys_2/'
    save_path = 'results/.analytics/results_analytics/' + out_dir_name

    model_dir = f"{model}/h_{horizon}/"
    scores_list = []
    make_output_dir(save_path)

    # Looping through every result dir for each experiment to get scores:
    resultsdir = rootdir+local_results_dir+model_dir
    for _, dataset_dirs, _ in os.walk(resultsdir):
        for dataset_dir_name in dataset_dirs:

            model_outdata_dir = local_results_dir + model_dir
            model_outdata_dir += dataset_dir_name + '/'
            # print(model_outdata_dir)

            scores_df = pd.read_csv(
                model_outdata_dir + 'scores_individual.csv',
                delimiter=';',
                encoding='utf-8-sig',
            )

            # Cleaning up naming and transposing:
            scores_df.rename(columns={"Unnamed: 0": ""}, inplace=True)
            scores_df = scores_df.T

            # Setting upper row as column names:
            scores_df.columns = scores_df.iloc[0]       # First row as colnames
            scores_df.drop(scores_df.index[0], inplace=True)  # Rm above-said

            # Resetting index and cleaning up naming:
            scores_df.reset_index(inplace=True)
            scores_df.rename(columns={"index": "Model"}, inplace=True)

            if dataset_dir_name == "selected":
                dataset_dir_name = "Selected"
            elif dataset_dir_name == "selected_rf_strict":
                dataset_dir_name = "RF Strict"
            elif dataset_dir_name == "selected_rf_top_7":
                dataset_dir_name = "RF Top-7"
            elif dataset_dir_name == "univariate":
                dataset_dir_name = "Univariate"
            else:
                raise ValueError(f"Unknown dataset naming: {dataset_dir_name}")

            scores_df.insert(0, 'Dataset', dataset_dir_name)
            scores_list.append(scores_df)

            # Modifying df to fit bar plotting and concating accumulation-df:
            # # scores_df = scores_df.set_index("Metric")
            # # scores_df.drop("Unnamed: 0", axis=1, inplace=True)
            # scores_df.insert(0, 'model', model_dir_name)
        break
        # Breaking to avoid other dirs than the most immediate to be looped.

    # Concatinating (adding vertically) every score df for each experiment:
    all_scores = pd.concat(scores_list, ignore_index=True)

    # with sns.set_theme(style="ticks"): # TODO: TESTS
    # sns.set_theme(style="ticks")

    # Define plotting region (2 rows, 2 columns)
    fig, axes = plt.subplots(4, 1, figsize=set_size(
        LATEX_DOC_TEXT_WIDTH,
        subplots=(4, 1),
        height_mult=height_multiplication,
    ))

    # Managing each individual sub-boxplot:

    # MAPE:
    mape = sns.boxplot(
        data=all_scores,
        y='Dataset', x='MAPE',
        order=['Univariate', 'RF Strict', 'RF Top-7', 'Selected'],
        whis=[0, 100],
        width=.6,
        showfliers=True,
        ax=axes[0],
    )
    mape.set_title('MAPE', fontname='New Century Schoolbook')  # , weight=TITLE_WEIGHT
    mape.set(ylabel=None)
    mape.set_xlabel("Percentage", fontname='New Century Schoolbook')  # , weight=AXES_LABEL_WEIGHT

    # RMSE:
    rmse = sns.boxplot(
        data=all_scores,
        y='Dataset', x='RMSE',
        order=['Univariate', 'RF Strict', 'RF Top-7', 'Selected'],
        whis=[0, 100],
        width=.6,
        showfliers=True,
        ax=axes[1],
    )
    rmse.set_title('RMSE', fontname='New Century Schoolbook')  # , weight=TITLE_WEIGHT
    rmse.set(ylabel=None)
    rmse.set_xlabel("Index points")  # , weight=AXES_LABEL_WEIGHT

    # R2-value:
    r2sc = sns.boxplot(
        data=all_scores,
        y='Dataset', x='R2_score',
        order=['Univariate', 'RF Strict', 'RF Top-7', 'Selected'],
        whis=[0, 100],
        width=.6,
        showfliers=True,
        ax=axes[2],
    )
    r2sc.set_title('R$^2$')  # , weight=TITLE_WEIGHT
    r2sc.set(ylabel=None)
    r2sc.set_xlabel("R$^2$ value")  # , weight=AXES_LABEL_WEIGHT

    # Time expenditure per model:
    time = sns.boxplot(
        data=all_scores,
        y='Dataset', x='time',
        order=['Univariate', 'RF Strict', 'RF Top-7', 'Selected'],
        whis=[0, 100],
        width=.6,
        showfliers=True,
        ax=axes[3],
    )
    time.set_title('Mean Model Runtime')  # , weight=TITLE_WEIGHT
    time.set(ylabel=None)
    time.set_xlabel("Seconds")  # , weight=AXES_LABEL_WEIGHT

    model_name = model_dir.split('/', 1)[0]
    fig.tight_layout()  # Preventing overlap

    plt.savefig(
        save_path+model_name+'.svg',
        format='svg',
        dpi=300,
        bbox_inches='tight'
    )

    return


def plot_forecast_summary_interesting_periods_focus(
        average_forecast: pd.DataFrame,
        all_forecasts: pd.DataFrame,
        output_dir: str,
        filename: str,
        true_path: str,
        true_vals_col_name: str,
        n_last_days: int,
):
    """
    Plots all individual models forecasts and their average for:
    * The covid crash year of 2020 as a whole
    * Focus on the covid crash period alone (feb-apr 2020)
    * The last n-days of the training period...................................

    The plots are saved to save_path as .svgs'.
    """

    # Last n given days focus plot:
    end_date = average_forecast.index[-1]
    start_date = end_date - BDay(n_last_days-1)
    avrg_nlast = average_forecast[start_date:end_date]
    fcts_nlast = all_forecasts[start_date:end_date]

    plot_forecast_summary(
        average_forecast=avrg_nlast,
        all_forecasts=fcts_nlast,
        output_dir=output_dir,
        filename=f"{filename}_last_{n_last_days}-BDays_plot",
        true_path=true_path,
        true_vals_col_name=true_vals_col_name,
    )

    # Full year 2020 focus plot:
    avrg_2020 = average_forecast[average_forecast.index.year == 2020]
    fcts_2020 = all_forecasts[all_forecasts.index.year == 2020]

    if (not fcts_2020.empty) and (not avrg_2020.empty):

        plot_forecast_summary(
            average_forecast=avrg_2020,
            all_forecasts=fcts_2020,
            output_dir=output_dir,
            filename=filename+'_year_2020_plot',
            true_path=true_path,
            true_vals_col_name=true_vals_col_name,
        )
    else:
        print(
            "[!] This series does not contain full 2020 data:",
            "no 2020 focus plots generated",
        )

    # Focus on 2020 feb-april covid crash period:
    start_c19crash = pd.Timestamp('2020-02-14')
    end_c19crash = pd.Timestamp('2020-04-16')
    avrg_ccrash = average_forecast[start_c19crash:end_c19crash]
    fcts_ccrash = all_forecasts[start_c19crash:end_c19crash]

    if (not fcts_ccrash.empty) and (not avrg_ccrash.empty):

        plot_forecast_summary(
            average_forecast=avrg_ccrash,
            all_forecasts=fcts_ccrash,
            output_dir=output_dir,
            filename=filename+"_c19-crash_plot",
            true_path=true_path,
            true_vals_col_name=true_vals_col_name,
        )
    else:
        print(
            "[!] Forecast do not contain full data for the c-19 crash:",
            "no covid focus plots generated",
        )


def plot_forecast_summary(
        average_forecast: pd.DataFrame,
        all_forecasts: pd.DataFrame,
        output_dir: str,
        filename: str,
        true_path: str,
        true_vals_col_name: str,
):
    """Plots a summary of all n-run models and saves to save_path"""

    make_output_dir(output_dir)

    # Retreiving true values:
    true_values = get_oseax_as_idx_values(
        raw_dataset_path=true_path,
        oseax_col_name=true_vals_col_name,
    ).pd_dataframe()

    true_values_sliced = true_values.loc[average_forecast.index]

    plt.figure(figsize=(6, (3/4)*6))

    # Plot all 30 predictions with weaker colors:
    plt.plot(
        all_forecasts,
        color='lightblue',
        linewidth=0.5,
    )

    # Plot the true values and average forecast with stronger colors:
    plt.plot(
        true_values_sliced,
        color='black',
        label='True Values',
        linewidth=2,
    )
    plt.plot(
        average_forecast,
        color='blue',
        label='Average Forecast',
        linewidth=2,
    )

    # Finalizing and saving:
    # plt.grid(linestyle=':')
    plt.legend()
    plt.savefig(
        output_dir + filename + '.svg',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()