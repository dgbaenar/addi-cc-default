import os
import pandas as pd

from exceptions import ModelParamException
from models.default_model import DefaultModel
import request


input_file_path = "./data/input_to_test.csv"


def calculate_probability():
    try:
        df = pd.read_csv(input_file_path)
        proba, shaps, X = DefaultModel.get_batch_model_response(df)
        output = pd.DataFrame({"id": df["ID"],
                        "default_probability": proba,
                        "explainability": shaps["explainability"],
                        "variables": X["variables"],
                        "threshold": X["threshold"]})
        output["looks_defaulter"] = output["default_probability"] > output["threshold"]
        output.to_csv("./data/output.csv", index=False)
    except ModelParamException as e:
        return {'error': str(e)}


if __name__ == '__main__':
    calculate_probability()
