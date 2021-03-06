# addi-cc-default

## UCI Default of Credit Card Clients Dataset

The goal of this project is to automate in batch the prediction of the probability of default for the users that currently have a loan using the [UCI Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). The current repository uses a XGBoost model with a grid search approach to find optimal hyperparameters.

## How to reproduce the experiment?

Start by deleting all the csv files stored in the data folder.

Then, create a virtual environment using python 3.8 and install the requirements following the next steps:

```console
virtualenv venv --python=python3.8
```
```console
source venv/bin/activate
```
```console
pip install -r requirements.txt
```
### Option 1:

Pull the data, preprocess it and train the XGBoost model using Data Version Control (DVC) running the following command:

```console
dvc repro
```

The previous command will run the pipeline located in the dvc.yaml file.

### Option 2:

Run manually the following scripts in order:

```console
python pull.py
```
```console
python preprocess.py
```
```console
python train.py
```

The outputs will be located in the data folder, and those are:
- `data`:
    - `raw.csv`: csv downloaded from UCI website.
    - `train.csv`: train dataset from raw.
    - `test.csv`: test dataset from raw.
    - `classes.csv`: actual and predicted classes from test set.
    - `input_to_test.csv`: sample from raw dataset to test the batch_run.py.
    - `output.csv`: output from batch_run.py
    - `data.json`: file with the categorical and numerical variables used in the model.
- `img`: plots of the confusion matrix from the test data set and the shap values of all the variables used to predict default probability.
- `metrics`: threshold, roc_auc, precision, recall and f1_score.
- `models`: joblib files for imputer, model and shap explainer.

## How to run the model on batch?

### Option 1:

Follow the commands of the previous step to setup the virtual environment and have defined the input data to test.

Then run:

```console
python src/batch_run.py
```

This command will run the `src/batch_run.py` file using the `data/input_to_test.csv` as input.

### Option 2:

Build and deploy the docker container to run a cron schedule task, and change the source and output path of the data in the `src/batch_run.py` file.
