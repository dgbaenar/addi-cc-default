import json
import pandas as pd
import joblib


class DefaultModel:
    with open('./data/data.json', 'r') as file:
        data = json.load(file)
    with open('./data/metrics/threshold.json', 'r') as t_file:
        threshold = json.load(t_file)['threshold']

    model_defaulters = joblib.load('./data/models/model.joblib.dat')
    imputer = joblib.load('./data/models/missing_imputer.joblib.dat')
    explainer = joblib.load('./data/models/shap_explainer.joblib.dat')

    CATEGORICAL_FEATURES = data.get("CATEGORICAL_FEATURES")
    NUMERICAL_FEATURES = data.get("NUMERICAL_FEATURES")
    CATEGORIES = data.get("CATEGORIES")

    batch_input_vars = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

    @staticmethod
    def preprocess_categories_batch(data: pd.DataFrame) -> pd.DataFrame:
        input_vars = set(data.columns)

        for feature in DefaultModel.data.get("CATEGORICAL_FEATURES"):
            data[feature].fillna("MISSING", inplace=True)
        for key in DefaultModel.data.get("CATEGORIES").keys():
            if key in input_vars:
                for category in DefaultModel.data.get("CATEGORIES")[key]:
                    data[key + '_' + str(category)] = data[key].map(
                        lambda x: 1 if x == category else 0)

        for key in DefaultModel.data.get("CATEGORIES").keys():
            del data[key]

        return data

    @staticmethod
    def preprocess_numeric(dataset: pd.DataFrame) -> pd.DataFrame:
        for var in DefaultModel.data.get("NUMERICAL_FEATURES"):
            dataset[var] = dataset[var].fillna(0)
        dataset[DefaultModel.data.get("NUMERICAL_FEATURES")] = DefaultModel.imputer.transform(
            dataset[DefaultModel.data.get("NUMERICAL_FEATURES")])

        return dataset

    @staticmethod
    def get_feature_vector_batch(data: pd.DataFrame) -> pd.DataFrame:
        data = DefaultModel.preprocess_categories_batch(data)
        X = data[DefaultModel.data.get("FINAL_DATA")]
        X = DefaultModel.preprocess_numeric(X)
        return X

    @staticmethod
    def explain_score(X: pd.DataFrame) -> dict:
        individual_shaps = [round(float(x), 2)
                            for x in DefaultModel.explainer.shap_values(X)[0]]
        dictionary = dict(zip(X.columns.tolist(), individual_shaps))

        return dictionary

    @staticmethod
    def calculate_score(X: pd.DataFrame) -> float:
        proba = DefaultModel.model_defaulters.predict_proba(X)[0][1]
        proba = round(float(proba), 6)

        return proba

    @staticmethod
    def get_batch_model_response(data: pd.DataFrame) -> pd.DataFrame:
        input_vars = data[DefaultModel.batch_input_vars]
        vars = input_vars.copy()
        X = DefaultModel.get_feature_vector_batch(input_vars)
        proba = DefaultModel.model_defaulters.predict_proba(X)[:, 1]
        shaps = pd.DataFrame(DefaultModel.explainer.shap_values(X),
                             columns=input_vars.columns).round(2)
        shaps_ = pd.DataFrame({"explainability": shaps.to_dict("records")})
        X_ = pd.DataFrame({"variables": vars.to_dict("records"),
                          "threshold": DefaultModel.threshold})

        return proba, shaps_, X_
