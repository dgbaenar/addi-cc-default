import json
import pandas as pd
from pandera import Column, DataFrameSchema

from exceptions import ModelParamException


def validate_input_columns(data: pd.DataFrame):
    with open('./data/data.json', 'r') as file:
        data = json.load(file)
    try:
        MODEL_INPUT = data.get("MODEL_INPUT")
        data = data.loc[:, MODEL_INPUT]
    except ModelParamException as e:
        return {'error': str(e)}

    return data


def validate_input(data: pd.DataFrame):
    try:
        data = validate_input_columns(data)
        schema = DataFrameSchema({"ID": Column(int, coerce=True),
                                "LIMIT_BAL": Column(int, coerce=True),
                                "AGE": Column(int, coerce=True),
                                "PAY_0": Column(int, coerce=True),
                                "PAY_2": Column(int, coerce=True),
                                "PAY_3": Column(int, coerce=True),
                                "PAY_4": Column(int, coerce=True),
                                "PAY_5": Column(int, coerce=True),
                                "PAY_6": Column(int, coerce=True),
                                "BILL_AMT1": Column(int, coerce=True),
                                "BILL_AMT2": Column(int, coerce=True),
                                "BILL_AMT3": Column(int, coerce=True),
                                "BILL_AMT4": Column(int, coerce=True),
                                "BILL_AMT5": Column(int, coerce=True),
                                "BILL_AMT6": Column(int, coerce=True),
                                "PAY_AMT1": Column(int, coerce=True),
                                "PAY_AMT2": Column(int, coerce=True),
                                "PAY_AMT3": Column(int, coerce=True),
                                "PAY_AMT4": Column(int, coerce=True),
                                "PAY_AMT5": Column(int, coerce=True),
                                "PAY_AMT6": Column(int, coerce=True)
                                })
        schema.validate(data)
    except ModelParamException as e:
        return {'error': str(e)}

    return data
