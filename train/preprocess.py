import json

import joblib
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing, impute


# Variables
with open("./data/data.json", "r") as file:
    data = json.load(file)

CATEGORICAL_FEATURES = data.get("CATEGORICAL_FEATURES")
NUMERICAL_FEATURES = data.get("NUMERICAL_FEATURES")
CATEGORIES = data.get("CATEGORIES")

# Options
dataset_path = "./data/raw.csv"
train_dataset_path = "./data/train.csv"
test_dataset_path = "./data/test.csv"
imputer_path = "./data/models/missing_imputer.joblib.dat"
scaler_path = "./data/models/scaler.joblib.dat"
metric_name = "DEFAULT PAYMENT NEXT MONTH"
test_size = 0.15
random_state = 42

# Read Dataset
print("Reading dataset...")
initial_dataset = pd.read_csv(dataset_path).drop_duplicates(subset="ID")
initial_dataset.columns = [x.upper() for x in initial_dataset.columns]

# To test
to_test = initial_dataset.sample(100)
to_test.to_csv('./data/input_to_test.csv', index=False)

initial_dataset = initial_dataset.set_index("ID")
dataset = initial_dataset[CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [metric_name]]

# Preprocess categorical
if CATEGORICAL_FEATURES:
    for feature in CATEGORICAL_FEATURES:
        dataset[feature] = dataset[feature].astype(object)
    ## Imput Missing Values
    dataset[CATEGORICAL_FEATURES] = dataset[CATEGORICAL_FEATURES].fillna("MISSING")

    for feature in CATEGORICAL_FEATURES:
        dataset[feature] = pd.Categorical(values=dataset[feature], categories=CATEGORIES[feature])
        
    ## Dummies
    categorical = pd.get_dummies(dataset[CATEGORICAL_FEATURES])
    dataset = dataset.join(categorical)
    dataset = dataset.drop(CATEGORICAL_FEATURES, axis=1)

# Split and save
print("Splitting...")
train, test = model_selection.train_test_split(dataset, test_size=test_size, random_state=random_state)

# Preprocess numerical
if NUMERICAL_FEATURES:
    for feature in NUMERICAL_FEATURES:
        train[feature] = train[feature].astype(float)
    ## Imput Missing values
    imputer = impute.SimpleImputer(missing_values=np.nan, strategy="mean")
    train[NUMERICAL_FEATURES] = imputer.fit_transform(train[NUMERICAL_FEATURES])
    test[NUMERICAL_FEATURES] = imputer.transform(test[NUMERICAL_FEATURES])
    joblib.dump(imputer, imputer_path)

# Rebalance
print("Rebalancing...")
train = pd.concat([train[train[metric_name] == 0].sample(train[metric_name].value_counts().loc[1]),
                   train[train[metric_name] == 1]])

train.to_csv(train_dataset_path, index=False)
test.to_csv(test_dataset_path, index=False)
