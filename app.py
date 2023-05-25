import onnxruntime as rt
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import nltk
import json
import numpy as np
import joblib
import pandas as pd

def select_features_prepared(df):
    """
    Select and prepare specific features from the input dataframe.

    Args:
        df (pandas.DataFrame): Input dataframe containing the data.

    Returns:
        pandas.DataFrame: DataFrame with the selected and prepared features.

    """

    seleted_cols = ['fallnr', 'log_euroscore_perc', 'height_cm', 'eks',
                    'nora_max_mcgperml', 'icu_max_postop_plusbilanz_ml',
                    'gesamtbilanz_l', 'bilanz_pro_kg_lkg', 'attest_ja1',
                    'haematokrit_preop_percent', 'tropi_preop_ngperml',
                    'tropi_max_postop_ngperml', 'tropi_min_postop_ngperml',
                    'tropt_min_postop_ngperl', 'ckmb_preop_mcgperl',
                    'ck_max_postop_uperl', 'ck_min_postop_uperl', 'age_at_surg',
                    'delir_dauer_tage', 'therapie_relevantes_delir_ja1']
    
    columns_boxcox = ['log_euroscore_perc', 'eks', 'nora_max_mcgperml',
                      'icu_max_postop_plusbilanz_ml', 'gesamtbilanz_l',
                      'bilanz_pro_kg_lkg', 'tropi_preop_ngperml',
                      'tropi_min_postop_ngperml', 'tropt_min_postop_ngperl',
                      'ckmb_preop_mcgperl', 'ck_max_postop_uperl',
                      'ck_min_postop_uperl']
    
    df[columns_boxcox] = df[columns_boxcox] + 1

    return df[seleted_cols]


def simulation_function(df, path_model):
    """
    Perform a simulation using a pre-trained pipeline to make predictions on new data.

    Args:
        df (pandas.DataFrame): Input dataframe containing the data for simulation.
        path_model (str): File path to the saved pipeline model.

    Returns:
        numpy.ndarray: Predicted labels for the input data.

    """

    # Prepare the data by selecting the desired features
    data = select_features_prepared(df)

    # Load the saved pipeline from the specified file path
    pipeline = joblib.load(path_model)

    # Make predictions using the loaded pipeline
    y_pred = pipeline.predict(data)

    return y_pred

def validate_port_number(input):
    """
    Validate the port number entered by the user.

    Args:
        input (str): The port number entered by the user.

    Returns:
        int: The validated port number.

    """
    while True:
        try:
            input = int(input)
        except:
            input = input("Enter valid Port Number: ")
        else:
            return input

app = FastAPI()

class Item(BaseModel):
    data: pd.DataFrame

@app.post("/predict")
async def predict(text: Item):
    """
    Perform prediction on the input data.

    Args:
        text (Item): Input data containing the dataframe.

    Returns:
        Item: Object containing the predicted dataframe.

    """
    df = text.data
    result = simulation_function(df, 'pipeline.pkl')
    df['sentence'] = result
    print(result)
    return {'data': df}

port_number = validate_port_number(input("Enter Port Number: "))

uvicorn.run(app=app, host='127.0.0.1', port=port_number)

