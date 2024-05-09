"""Help function and methods for plotting data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from darts import TimeSeries
from pandas.tseries.offsets import BDay
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
)
from scripts.helpfuncs_models import (
    make_output_dir,
    get_oseax_as_idx_values,
    LossLogger,
)


# PARAMS:
plt.rcParams['font.size'] = 12
# plt.rcParams.update({'font.family': 'Helvetica'})
# plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['pdf.fonttype'] = 42  # Make sure to embed the font into the PDF
# plt.rcParams.update({'font.family':'sans-serif'})
# plt.rcParams['font.family'] = 'Times New Roman'

# try:
#     rc('font', **{'family': 'serif', 'serif': ['Times']})
#     # rc('text', usetex=True)
# except Exception as exc:
#     print(f"[!] Were not able to set font family to serif Times Error: {exc}")


def gen_time_crmtrx_and_save(
        data: pd.DataFrame,
        target_variable: str,
        shift_n_days: list[int],
        output_dir: str,
        fig_size: tuple = (8.27, 11.69),
):
    """
    Creates a n time shifted correlation matrix for given differet time
    periods.
    """

    make_output_dir(output_dir)

    # Renaming target variable:
    renamed_target_var = "same day"

    # Shifting target value:
    df_shifted = data.copy()         # Copying data set.
    df_shifted = df_shifted.rename(  # Renaming target vairable's column.
        columns={target_variable: renamed_target_var}
    )

    # For n desired time-shifts:
    for shift in reversed(shift_n_days):

        # Give name to shifted target feature's column:
        namestr = "+" + str(abs(shift)) + "d"

        # Insert shifted version of target variable in new df:
        df_shifted.insert(
            1,
            namestr,
            df_shifted[renamed_target_var].shift(shift)
        )

    df_shifted.ffill(inplace=True)
    df_shifted.dropna(inplace=True)

    # Creating the correlation matrix:

    crmtrx = df_shifted.corr()

    # Slicing away unnecessary data:
    crmtrx_sliced = (
        crmtrx.iloc[len(shift_n_days)+1:, :len(shift_n_days)+1]
    ).copy()

    n_features = len(crmtrx_sliced.index)
    print("Number of features: ", n_features)

    # TODO: rm DateTime/futcovs from plots (?)
    mtrxs = {
        "matrix_1": crmtrx_sliced.iloc[:14, :].copy(),
        "matrix_2": crmtrx_sliced.iloc[14:28, :].copy(),
        "matrix_3": crmtrx_sliced.iloc[28:42, :].copy(),
        "matrix_4": crmtrx_sliced.iloc[42:56, :].copy(),
        "matrix_5": crmtrx_sliced.iloc[56:71, :].copy(),
    }

    for key, mtrx in mtrxs.items():
        print(key, "size:", len(mtrx))

    # Plot params:

    cbar_kws = dict(
        use_gridspec=True,
        location="top",
        pad=0.02,
        # shrink=.765
        shrink=0.5
    )

    with PdfPages(output_dir+'corr_matrix.pdf') as pdf_pages:
        for mtrx_data in mtrxs.values():
            fig, axes = plt.subplots(
                figsize=fig_size,
                gridspec_kw={"left": 0.5}
            )
            sns.heatmap(
                mtrx_data,
                vmin=-1,   # Color range min.
                center=0,  # Center color value
                vmax=1,    # Color range max.
                square=True,
                linewidths=0.5,    # Width of lines between cells.
                cbar_kws=cbar_kws,
                annot=True,     # Show numbers
                fmt='.2f',    # Num. of decimal digits displayed.
                cmap='RdBu',   # Color.
                ax=axes,
            )
    # (si, si*1.414)) - A4 dimension is 1:1.414

            plt.subplots_adjust(top=0.97)
            plt.grid(visible=False)
            plt.xlabel('')  # Removing axis labels
            plt.ylabel('')

            pdf_pages.savefig(fig, dpi=300)
            plt.close()

    return


def gen_lineplots_and_save(
        indata: pd.DataFrame,
        file_dest: str,
        filename_categorization: str = None,
        incl_labels_title: bool = False,
):
    """
    Creates and saves plots of every individual feature in the given DataFrame
    """

    make_output_dir(file_dest)

    for feature in indata:
        plt.figure(figsize=(6, (3/4)*6))
        plt.plot(
            indata.index,
            indata[feature],
            # color='blue',
            linewidth=2,
        )
        # plt.grid(linestyle=':')

        if incl_labels_title:
            plt.title(feature)
        else:
            plt.xlabel('')
            plt.ylabel('')

        # Replace forbidden chars and spaces:
        savename = feature.replace("/", "-").replace(" ", "_")

        plt.savefig(
            file_dest + savename + filename_categorization + ".svg",
            dpi=300,
            bbox_inches='tight',
        )

        plt.close()  # Clearing figure to prevent it from opening


def plot_results(
        full_target: TimeSeries,
        test_target: TimeSeries,
        model_forecast: TimeSeries,
        n_last_d: int = 365,
        directory: str = '',
        show: bool = False,
        save: bool = False,
        incl_c19_crash_focus=False,
):
    """
    Plots the results of the model prediction against the target series.
    """

    make_output_dir(directory)

    # Plot of models OSEAX forecast vs. true values for full history:
    plt.figure(figsize=(6, (3/4)*6))
    full_target.plot(label="Actual")
    model_forecast.plot(label="Forecast")
    # plt.ylabel("OSEAX valuation")
    if save:
        plt.savefig(
            directory+"full.svg",
            dpi=300,
            bbox_inches='tight',
        )
    # plt.grid(linestyle=':')
    plt.xlabel('')
    plt.ylabel('')
    if show:
        plt.show()
    else:
        plt.close()

    # Plot of testing period only:
    plt.figure(figsize=(6, (3/4)*6))
    test_target.plot(label="Actual")
    model_forecast.plot(label="Forecast")
    if save:
        plt.savefig(
            directory+"test.svg",
            dpi=300,
            bbox_inches='tight',
        )
    # plt.grid(linestyle=':')
    plt.xlabel('')
    plt.ylabel('')
    if show:
        plt.show()
    else:
        plt.close()

    # Plot of the last year of training data (260 BDays):
    plt.figure(figsize=(6, (3/4)*6))
    full_target[-260:].plot(label="Actual")
    model_forecast[-260:].plot(label="Forecast")
    if save:
        plt.savefig(
            directory+"last_260Bd.svg",
            dpi=300,
            bbox_inches='tight',
        )
    # plt.grid(linestyle=':')
    plt.xlabel('')
    plt.ylabel('')
    if show:
        plt.show()
    else:
        plt.close()

    # Plot of forecast vs true for n-last business days:
    plt.figure(figsize=(6, (3/4)*6))
    full_target[-n_last_d:].plot(label="Actual")
    model_forecast[-n_last_d:].plot(label="Forecast")
    if save:
        plt.savefig(
            directory+"last_"+str(n_last_d)+"Bd.svg",
            dpi=300,
            bbox_inches='tight',
        )
    # plt.grid(linestyle=':')
    plt.xlabel('')
    plt.ylabel('')
    if show:
        plt.show()
    else:
        plt.close()

    # Plot of last month (20 BDays)
    plt.figure(figsize=(6, (3/4)*6))
    full_target[-20:].plot(label="Actual")
    model_forecast[-20:].plot(label="Forecast")
    if save:
        plt.savefig(
            directory+"last_20d.svg",
            dpi=300,
            bbox_inches='tight',
        )
    # plt.grid(linestyle=':')
    plt.xlabel('')
    plt.ylabel('')
    if show:
        plt.show()
    else:
        plt.close()

    # Plot focusing on the covid-19 recession in 2020:
    if incl_c19_crash_focus:

        # Convert the darts TimeSeries to pandas DataFrames
        df_true = full_target.pd_dataframe()
        df_forecast = model_forecast.pd_dataframe()

        # Filter these DataFrames for 2020
        true_2020_df = df_true[df_true.index.year == 2020]
        forecast_2020_df = df_forecast[df_forecast.index.year == 2020]

        if (not true_2020_df.empty) and (not forecast_2020_df.empty):

            # Full year 2020 focus plot:
            plt.figure(figsize=(6, (3/4)*6))
            true_2020 = TimeSeries.from_dataframe(true_2020_df)
            true_2020.plot(label="Actual")
            forecast_2020 = TimeSeries.from_dataframe(forecast_2020_df)
            forecast_2020.plot(label="Forecast")

            # plt.grid(linestyle=':')
            plt.xlabel('')
            plt.ylabel('')

            if save:
                plt.savefig(
                    directory+"2020.svg",
                    dpi=300,
                    bbox_inches='tight',
                )
            if show:
                plt.show()
            else:
                plt.close()

            # Focus on crash period exclusively:
            plt.figure(figsize=(6, (3/4)*6))

            start_c19crash = pd.Timestamp('2020-02-14')
            end_c19crash = pd.Timestamp('2020-04-16')

            data_2020 = full_target.slice(start_c19crash, end_c19crash)
            data_2020.plot(label="Actual")
            forecast_cut = model_forecast.slice(start_c19crash, end_c19crash)
            forecast_cut.plot(label="Forecast")

            # plt.grid(linestyle=':')
            plt.xlabel('')
            plt.ylabel('')

            if save:
                plt.savefig(
                    directory+"c19-crash.svg",
                    dpi=300,
                    bbox_inches='tight',
                )
            if show:
                plt.show()
            else:
                plt.close()

        else:
            print(
                "[!] This series does not contain full 2020 data:",
                "no covid focus plots generated",
            )


def generate_and_save_loss_plots_and_loss_data(
        loss_logger: LossLogger(),
        output_dir: str,
        save: bool,
        info_print: bool,
        info_plot: bool,
):
    """
    Plotting and Saving Loss and Val Loss history to output_dir

    Regarding valid_loss: The callback will give one more element in the
    loss_logger.val_loss as the model trainer performs a validation sanity
    check before the training begins. To prevent the plots being offset,
    remove the first entry in the loss_logger.val_loss list.
    """

    make_output_dir(output_dir)  # Init own folder for loss data if desired

    train_loss = loss_logger.train_loss
    valid_loss = loss_logger.val_loss

    # Saving loss as txt:
    np.savetxt(output_dir+'train_loss.txt', train_loss)
    np.savetxt(output_dir+'validation_loss.txt', valid_loss)
    if info_print:
        print("train_loss:", train_loss)
        print("valid_loss:", valid_loss)

    # Standard scaled axis plot:
    plt.figure(figsize=(6, (3/4)*6))
    plt.plot(train_loss, label="Training loss", linewidth=2)
    plt.plot(valid_loss, label="Talidation loss", linewidth=2)
    plt.legend(loc='best')
    # plt.grid(linestyle=':')
    if save:
        plt.savefig(
            output_dir+"linear_loss_plot.svg",
            dpi=300,
            bbox_inches='tight',
        )
    if info_plot:
        plt.show()
    else:
        plt.close()

    # Logarithmic scaled plot:
    plt.figure(figsize=(6, (3/4)*6))
    plt.yscale('log')
    plt.plot(train_loss, label="Training loss", linewidth=2)
    plt.plot(valid_loss, label="Talidation loss", linewidth=2)
    plt.legend(loc='best')
    # plt.grid(linestyle=':')
    if save:
        plt.savefig(
            output_dir+"logarithmic_loss_plot.svg",
            dpi=300,
            bbox_inches='tight',
        )
    if info_plot:
        plt.show()
    else:
        plt.close()


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


def make_boxplots(
        all_scores: pd.DataFrame,
        prm: dict,
        save: bool,
        output_dir: str = '',
        plot_name: str = 'boxplot.svg',
        **kwargs,
):
    """Generates Seaborn Boxplots for the given metrics data"""

    # Setting stype (scientific paper/publishing grade):
    sns.set_style(
        "whitegrid",
        {
            'font.family': 'Times New Roman',
            'font.size':    12,
        },
    )
    # sns.set(font="Times New Roman")
    # sns.set_context(
    #     "paper",
    #     # font_scale=prm['font_scale'],
    #     rc={
    #         # "font.family": "Times New Roman",
    #         "axes.labelweight": "normal",
    #         },
    # )
    sns.set_palette("bright")
    font_properties = {
        'family':   'Times New Roman',
        'weight':   'normal',
        'size':     12,
    }

    # Generating boxplot:
    plt.figure(figsize=(prm['fig_dims'], prm['fig_dims']*1.15))
    axs = sns.boxplot(
        data=all_scores,
        x='model',
        y=prm['ylabel'],
        hue='dataset',
        **kwargs,
    )
    axs.set_xlabel('Model', fontdict=font_properties)
    axs.set_ylabel(prm['ylabel'], fontdict=font_properties)
    # ax.set_title('MAPE by Model and Dataset', fontdict=font_properties)
    # plt.title('MAPE by Model and Dataset')
    plt.xticks(rotation=prm['rotation'])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()

    # Saving:
    if save:
        make_output_dir(output_dir)
        plt.savefig(
            output_dir + plot_name,
            format='svg',
            dpi=300,
            bbox_inches='tight',
        )

    return axs


def make_triple_boxplot(
        all_scores: pd.DataFrame,
        prm: dict,
        save: bool,
        output_dir: str = '',
        plot_name: str = 'triple_boxplot.svg'
):
    """
    Generates a triple box plot of MAPE, RMSE, R2-score and time-expenditure.
    """

    # Setting stype (scientific paper/publishing grade):
    sns.set_style(
        "whitegrid",
        {
            'font.family': 'Times New Roman',
            'font.size':    12,
        },
    )
    sns.set_palette("bright")
    font_properties = {
        'family':   'Times New Roman',
        'weight':   'normal',
        'size':     12,
    }

    # Initiatin main figure:
    fig, axes = plt.subplots(
        2, 2,
        figsize=(prm['subfig_dims']*2+prm['hrzn_pad'], prm['subfig_dims']*2+prm['vrtc_pad']),
    )

    # MAPE:
    label = 'MAPE'
    ax_v = 0
    ax_h = 0
    sns.boxplot(
        ax=axes[ax_v, ax_h],
        data=all_scores,
        x='model',
        y=label,
        hue='dataset',
    )
    axes[ax_v, ax_h].set_xlabel('Model', fontdict=font_properties, fontsize=12)
    axes[ax_v, ax_h].set_ylabel(label, fontdict=font_properties, fontsize=12)
    axes[ax_v, ax_h].tick_params(axis='x', which='major', labelsize=12, rotation=45)
    axes[ax_v, ax_h].tick_params(axis='y', which='major', labelsize=12)
    axes[ax_v, ax_h].get_legend().remove()
    # plt.tight_layout()

    # RMSE:
    label = 'RMSE'
    ax_v = 1
    ax_h = 0
    sns.boxplot(
        ax=axes[ax_v, ax_h],
        data=all_scores,
        x='model',
        y=label,
        hue='dataset',
    )
    axes[ax_v, ax_h].set_xlabel('Model', fontdict=font_properties, fontsize=12)
    axes[ax_v, ax_h].set_ylabel(label, fontdict=font_properties, fontsize=12)
    axes[ax_v, ax_h].tick_params(axis='x', which='major', labelsize=12, rotation=45)
    axes[ax_v, ax_h].tick_params(axis='y', which='major', labelsize=12)
    axes[ax_v, ax_h].get_legend().remove()
    # axes[ax_v, ax_h].legend(
    #     title='Dataset',
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 1.30),
    #     # fancybox=True, shadow=True, 
    #     ncol=3,
    # )
    # plt.tight_layout()

    # R2-score:
    label = 'R2_score'
    ax_v = 0
    ax_h = 1
    sns.boxplot(
        ax=axes[ax_v, ax_h],
        data=all_scores,
        x='model',
        y=label,
        hue='dataset',
    )
    axes[ax_v, ax_h].set_xlabel('Model', fontdict=font_properties, fontsize=12)
    axes[ax_v, ax_h].set_ylabel(label, fontdict=font_properties, fontsize=12)
    axes[ax_v, ax_h].tick_params(axis='x', which='major', labelsize=12, rotation=45)
    axes[ax_v, ax_h].tick_params(axis='y', which='major', labelsize=12)
    axes[ax_v, ax_h].get_legend().remove()

    plt.tight_layout()
    plt.legend(
        title='Dataset',
        # loc='upper center',
        bbox_to_anchor=(-0.10, 1.325),
        ncol=3,
    )

    # Saving:
    if save:
        make_output_dir(output_dir)
        plt.savefig(
            output_dir + plot_name,
            format='svg',
            dpi=300,
            bbox_inches='tight',
        )

    return axes


def plot_and_save_optimization_history(
        study,
        path: str,
        target_name: str,
):
    """Optuna optimization history plot-and-save method"""

    plt.figure(figsize=(6, (3/4)*6))
    plot_optimization_history(study, target_name=target_name, error_bar=True)
    plt.savefig(path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_and_save_param_importances(
        study,
        path: str,
        target_name: str,
):
    """Optuna parameters importances plot-and-save method"""

    plt.figure(figsize=(6, (3/4)*6))
    plot_param_importances(study, target_name=target_name)
    plt.savefig(path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()


def set_size(width, fraction=1, subplots=(1, 1), height_mult=1.0):

    """
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    SORUCE:
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """

    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    fig_height_in = fig_height_in * height_mult

    return (fig_width_in, fig_height_in)
