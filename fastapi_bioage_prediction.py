from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pandas as pd
from joblib import load
from enum import Enum
import numpy as np
from pydantic import BaseModel, validator, ValidationError

app = FastAPI()


class Gender(Enum):
    Male = "Male"
    Female = "Female"


class InputFeatures(BaseModel):
    Age: float = Field(default=40, ge=10, le=120, description="Age in ranges 10 - 100")
    Weight_kg: float = Field(default=70.0, ge=20, le=200, description="Weight in kilograms within the range 20-200")
    Standing_Height_cm: float = Field(default=170.0, ge=50, le=250, description="Standing height in centimeters within the range 50-250")
    Waist_Circumference_cm: float = Field(default=80.0, ge=30, le=150, description="Waist circumference in centimeters within the range 30-150")
    Systolic_blood_pressure_average: float = Field(default=120.0, ge=70, le=250, description="Systolic blood pressure average in mmHg within the range 70-250")
    Gender: str = Field(default=Gender.Male, description="Gender of the individual")


# Define the output model
class PredictionResult(BaseModel):
    prediction: float
    leaderbord_text: str
    leaderbord_value: float
    age_group: str


# Load models
model_to_use = 'catboost'
model_paths = [
    f'models/{model_to_use}/model_fold_1.joblib',
    f'models/{model_to_use}/model_fold_2.joblib',
    f'models/{model_to_use}/model_fold_3.joblib',
    f'models/{model_to_use}/model_fold_4.joblib',
    f'models/{model_to_use}/model_fold_5.joblib'
] 

models = [load(model_path) for model_path in model_paths]
health_stock_percentile_dff = pd.read_csv('models/health_stock_percentile_dff.csv', index_col=0)

def get_preds(models, data_row):
    predictions = 0
    for model in models:
        pred = model.predict(data_row[model.feature_names_])
        predictions += pred[0]
    prediction = predictions / len(models)
    return round(prediction, 2)

def feature_engineering(df):
    df = df.rename(columns = {
        'Weight_kg': 'Weight (kg)',
        'Standing_Height_cm': 'Standing Height (cm)',
        'Waist_Circumference_cm': 'Waist Circumference (cm)',
        'Systolic_blood_pressure_average': 'Systolic blood pressure average',
        'Gender': 'gender',
                         })

    df['Body Mass Index (kg/m**2)'] = (df['Weight (kg)'] / ((df['Standing Height (cm)']*0.01)**2)).round(2)
    df['Waist to Height ratio'] = df['Waist Circumference (cm)'] / df['Standing Height (cm)']
    df['gender'] = df['gender'].replace({'Male':1, 'Female':0})

    df['Systolic blood pressure average^2'] = df['Systolic blood pressure average'] ** 2
    df['Body Mass Index (kg/m**2)^2'] = df['Body Mass Index (kg/m**2)'] ** 2
    df['Waist to Height ratio^2'] = df['Waist to Height ratio'] ** 2

    df['Systolic blood pressure average_log'] = df['Systolic blood pressure average'].apply(np.log)
    df['Body Mass Index (kg/m**2)_log'] = df['Body Mass Index (kg/m**2)'].apply(np.log)
    df['Waist to Height ratio_log'] = df['Waist to Height ratio'].apply(np.log)

    return df

def data_validation(df):
        
        ranges = {
            'Weight_kg': (20, 200),
            'Standing_Height_cm': (50, 250),
            'Waist_Circumference_cm': (40, 150),
            'Systolic_blood_pressure_average': (70, 200),
        }
        for key in ranges.keys():
            value = df[key].values.tolist()[0]
            min_val, max_val = ranges[key]
            if not (min_val <= value <= max_val):
                raise ValueError(f'{key} must be between {min_val} and {max_val}.')
            elif Gender not in ['Male', 'Female']:
                raise ValueError(f'{key} must be Male or Female')
                

def find_bioage_percentile(age: int, health_stock: float, percentile_df: pd.DataFrame) -> str:
    """
    Find the percentile rank of an individual's health_stock within their age group.
    
    Parameters:
    - age: The age of the individual.
    - health_stock: The health_stock value of the individual.
    - dff: The DataFrame containing health_stock values across different ages.
    
    Returns:
    - A string indicating how the individual's health_stock compares to others in their age group.
    """
    percentile_df = percentile_df.reset_index().melt(id_vars='index')
    percentile_df.rename(columns={'index': 'Percentile', 'variable': 'Age', 'value': 'health_stock'}, inplace=True)

    # Determine the age group
    age = int(age)
    health_stock = float(health_stock)
    age_group = age // 5 * 5  # Group by every 5 years, aligning with the provided logic
    if age_group < 20:
        age_group = 20
    elif age_group > 85:
        age_group = 85
    
    # Filter the DataFrame for the specific age group
    age_group_dff = percentile_df[percentile_df['Age'].astype(int) == int(age_group)]

    # Compute the percentile of the individual's health_stock within the age group
    percentile_rank = round(np.mean(health_stock > age_group_dff['health_stock']) * 100, 2)
    
    return {"leaderbord_text": f"Ваше значение биовозраста лучше чем у {percentile_rank}% людей вашей возрастной группы",
            "leaderbord_value": percentile_rank,
            "age_group": age_group
            }


@app.post("/predict/", response_model=PredictionResult)
def predict(features: InputFeatures):
    try:
        data_dict = features.__dict__
        for key, value in data_dict.items():
            if isinstance(value, Enum):
                data_dict[key] = value.value

        input_df = pd.DataFrame([data_dict.values()], columns=data_dict.keys())
        final_df = feature_engineering(input_df)

        prediction = get_preds(models, final_df)
        final_df['y_pred'] = prediction
        Age = input_df['Age'].values.tolist()[0]
        health_stock = Age - prediction
        leaderbord = find_bioage_percentile(Age, health_stock, health_stock_percentile_dff)

        return {"prediction": prediction,
                "leaderbord_text": leaderbord['leaderbord_text'],
                "leaderbord_value": leaderbord['leaderbord_value'],
                "age_group": f"От {int(leaderbord['age_group'])-5} до {int(leaderbord['age_group'])}"
                }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
