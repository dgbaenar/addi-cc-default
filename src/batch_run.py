import os
import pandas as pd

from models.default_model import DefaultModel


input_file_path = "./data/input_to_test.csv"


def calculate_probability():
    df = pd.read_csv(input_file_path)
    proba, shaps, X = DefaultModel.get_batch_model_response(df)
    output = pd.DataFrame({"id": df["ID"],
                       "default_probability": proba,
                       "explainability": shaps["explainability"],
                       "variables": X["variables"],
                       "threshold": X["threshold"]})
    output["looks_defaulter"] = output["default_probability"] > output["threshold"]
    output.to_csv("./data/output.csv", index=False)


if __name__ == '__main__':
    calculate_probability()
