# OSEAX Deep Learning LSTM Forecaster

An LSTM-based machine learning forecasting model for predicting the future valuation of the Oslo All-share Index (OSEAX) using (macro)economic data.

## 🔹 Dependencies

Required libraries include NumPy, Pandas, Darts, PyTorch, and other necessary packages listed in `environment.yml`. To install these, use the provided Conda environment setup file.

## 🔹 Installation

1) Unzip this folder to your desired location.
2) Install Conda from [Conda's official site](https://www.conda.io/projects/conda/en/latest/user-guide/install/index.html).
3) Create a new environment with the command: `conda env create -f environment.yml`.
4) Activate the new environment using `conda activate <env_name>` and navigate to the folder where you unpacked this zip in step 1.
5) Test the setup with the command provided in the installation guide.

## 🔹 Running the application

The script must be run from the folder where this zip was unpacked - the same location as the `run.py` file. Below are the instructions for running different intensities of the model.

### Test run - simple (recommended)

Run a light testing-version of the code with the following command:
```bash
python run.py lstm 1 univariate test_light cpu 0 print
```

### Test run - thorough

For a more thorough test run this command:
```bash
python run.py lstm 1 univariate test cpu 0 print
```

This will run the ***lstm*** model, forecasting ***1*** business day ahead, using a ***univariate*** dataset (historical prices only) which has been reduced to a ***test*** size of 5 years. It will run on the ***cpu*** with ***0*** indicating that no more than one core will be utilized. ***print*** makes all possible information be printed in the console (for debugging/learning).

### Full forecasting experiment run

[!] (CUDA-enabled GPUs are highly recommended)

For running the full experiment, the following command is recommended:
```bash
python run.py lstm 1 selected full gpu 2 silent
```

* The number after `lstm` can be either ***1***, ***5***, ***22***, or ***261***, for forecasting one business day, one business week, one business month, or one business year, respectively.

* The argument ***selected*** tells the model to train on the main dataset composed of 9 Norwegian macroeconomic indicators.

* The ***full*** argument tells the model to utilize the entire length of the dataset.

The ***gpu*** argument allows the model to utilize CUDA resources, if available. Following, the number ***2*** instructs the script to use up to two CPU-cores when running. Finally, the ***silent*** argument reduces the information printed to the console to the bare minimum.

## 🔹 Viewing the Results

After completing the forecasting, the results are stored in a `/results` folder. To view them, follow these steps:

1) Navigate to the folder corresponding to the forecasting horizon used, such as `/h_1` for a 1 business day horizon.

2) Within this folder, navigate to the subfolder that matches the dataset used in the forecast, either `/univariate` or `/selected`. In this folder, you will find:

    * Plots of the model's average forecast on the test dataset.

    * Accuracy measurement metrics. Averages over multiple model runs are found in the `scores_mean_stddev.csv` file.
