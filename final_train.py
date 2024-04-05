import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm.auto import tqdm
from catboost import CatBoostRegressor
import os

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import dump, load
import sys, os



# LOAD DATA
demographic_path = 'data_inner/nhanes_1994-2018/demographics_clean.csv'
chemical_path = 'data_inner/nhanes_1994-2018/chemicals_clean.csv'
response_path = 'data_inner/nhanes_1994-2018/response_clean.csv'
dictionary_path = 'data_inner/nhanes_1994-2018/dictionary_nhanes.csv'
translations_path = 'data/columns_translations.csv'

# Load dataframes with specified columns
demographics = pd.read_csv(demographic_path, index_col=0, usecols=['RIDAGEYR', 'RIAGENDR', 'SEQN', 'SEQN_new'])
chemicals = pd.read_csv(chemical_path)
response = pd.read_csv(response_path)
translations = pd.read_csv(translations_path, index_col=0)

# Merge dataframes
df = demographics.merge(chemicals, how='inner', on=['SEQN', 'SEQN_new'])
df = df.merge(response, how='inner', on=['SEQN', 'SEQN_new'])

# Remove data with too many NaNs
# df = df[df.columns[df.isna().mean() < 0.5]]

# Rename columns
dictionary_nhanes = pd.read_csv(dictionary_path)
mapping_dict = dict(zip(dictionary_nhanes['variable_codename_use'], dictionary_nhanes['variable_description_use']))
mapping_dict.update({'RIDAGEYR': 'age', 'RIAGENDR': 'gender'})
df = df.rename(columns=mapping_dict)

# Fix gender data
df['gender'] = df['gender'].astype(str).replace({'2': 'Female', '1': 'Male'})
# df.to_csv('data_inner/raw_nhanes_data.csv', index=False)
not_replicated_columns = df.columns[~df.columns.str.contains('replicate')]
df = df[not_replicated_columns]
df = df[(df['age']>18) & (df['age']<70)]



# SELECT FINAL FEATURES
# features = [
#     'gender', 
#     'Systolic blood pressure average', 

#     # 'alpha-tocopherol (µg/dL)', # 2620 рублей
#     # 'Serum creatinine (mg/dL)', # 1,610.00
#     # 'Serum homocysteine: SI (umol/L)', # 2270 рублей 
#     # 'Serum ferritin (ng/mL)', # 630

#     # ~ 3000 рублей
#     # 'Serum C-reactive protein (mg/dL)', # 690
#     # 'Estimated Glomerular Filtration Rate (mL/min/1.73 m2)', # 350 рублей
#     # 'Serum blood urea nitrogen (mg/dL)', # 385 
#     # 'Serum albumin:  SI (g/L)', # 440
#     # 'Forced expiratory vol(FEV),.5 sec,max-ml', 
# ]

# df['Body Mass Index (kg/m**2)'] = (df['Weight (kg)'] / ((df['Standing Height (cm)']*0.01)**2)).round(2)
# df['Waist to Height ratio'] = df['Waist Circumference (cm)'] / df['Standing Height (cm)']
# df['gender'] = df['gender'].replace({'Male':1, 'Female':0})

# df['Systolic blood pressure average^2'] = df['Systolic blood pressure average'] ** 2
# df['Body Mass Index (kg/m**2)^2'] = df['Body Mass Index (kg/m**2)'] ** 2
# df['Waist to Height ratio^2'] = df['Waist to Height ratio'] ** 2

# df['Systolic blood pressure average_log'] = df['Systolic blood pressure average'].apply(np.log)
# df['Body Mass Index (kg/m**2)_log'] = df['Body Mass Index (kg/m**2)'].apply(np.log)
# df['Waist to Height ratio_log'] = df['Waist to Height ratio'].apply(np.log)

# features.extend(['Body Mass Index (kg/m**2)', 'Waist to Height ratio'])
# features.extend(['Systolic blood pressure average^2', 'Body Mass Index (kg/m**2)^2', 'Waist to Height ratio^2'])
# features.extend(['Systolic blood pressure average_log', 'Body Mass Index (kg/m**2)_log', 'Waist to Height ratio_log'])
# features.extend(['Weight (kg)', 'Standing Height (cm)', 'Waist Circumference (cm)'])
# df[features] = df[features].astype(float)
# df['gender'] = df['gender'].astype(int)

# df[features + meta_features].to_csv('train_data.csv')

meta_features = ['age', 'Respondent sequence number', 'Respondent sequence number that includes an identifier for NHANES III and NHANES continuous']

base_cols = ['Systolic blood pressure average', 
              'Hip Circumference (cm)', 
              'Weight (kg)', 'Standing Height (cm)', 'Waist Circumference (cm)',
              'gender']
df = df[df[base_cols].isna().sum(axis=1) < 3]
print('1', df.shape)

df['Body Mass Index (kg/m**2)'] = (df['Weight (kg)'] / ((df['Standing Height (cm)']*0.01)**2)).round(2)
df['Waist to Height ratio'] = df['Waist Circumference (cm)'] / df['Standing Height (cm)']
df['Waist to Hip ratio'] = df['Waist Circumference (cm)'] / df['Hip Circumference (cm)']

features = base_cols + ['Body Mass Index (kg/m**2)', 'Waist to Height ratio']#, 'Waist to Hip ratio']


# Polinomial features

# df['Systolic blood pressure average^2'] = df['Systolic blood pressure average'] ** 2
# df['Body Mass Index (kg/m**2)^2'] = df['Body Mass Index (kg/m**2)'] ** 2
# df['Waist to Height ratio^2'] = df['Waist to Height ratio'] ** 2

# df['Systolic blood pressure average_log'] = df['Systolic blood pressure average'].apply(np.log)
# df['Body Mass Index (kg/m**2)_log'] = df['Body Mass Index (kg/m**2)'].apply(np.log)
# df['Waist to Height ratio_log'] = df['Waist to Height ratio'].apply(np.log)
# features.extend(['Systolic blood pressure average^2', 'Body Mass Index (kg/m**2)^2', 'Waist to Height ratio^2'])
# features.extend(['Systolic blood pressure average_log', 'Body Mass Index (kg/m**2)_log', 'Waist to Height ratio_log'])


# Filter overweight people
def filter_extreme_weights(df, lower_age, upper_age, lower_weight_quantile, upper_weight_quantile):
    """
    Filters out individuals with extreme weights based on the specified quantiles for each age group.

    :param df: DataFrame containing age and weight columns.
    :param lower_age: The age to start applying the filter.
    :param upper_age: The age to end applying the filter.
    :param lower_weight_quantile: The lower quantile threshold of weight.
    :param upper_weight_quantile: The upper quantile threshold of weight.
    :return: A filtered DataFrame.
    """
    # Apply filters for each age to smooth out the transitions
    for age in range(lower_age, upper_age + 1):
        age_df = df[df['age'] == age]
        lower_threshold = age_df['Weight (kg)'].quantile(lower_weight_quantile)
        upper_threshold = age_df['Weight (kg)'].quantile(upper_weight_quantile)
        
        # Filter the dataframe for the current age
        df = df[~((df['age'] == age) & ((df['Weight (kg)'] < lower_threshold) | (df['Weight (kg)'] > upper_threshold)))]
    
    return df


train_df = df[features + meta_features].copy()
train_df = train_df[train_df['age'].notna()]
print('train_df.shape', train_df.shape)

selected_ages            = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
selected_upper_quantiles = [82, 83, 84, 85, 86, 87, 90, 92, 94, 96, 100, 100, 100, 100]
filtered_train_df = train_df
for age, upper_quantile in zip(selected_ages, selected_upper_quantiles):
    filtered_train_df = filter_extreme_weights(filtered_train_df, age-5, age, 0.0, upper_quantile/100)
train_df = filtered_train_df.copy()
print('Filtered overweights', train_df.shape[0])

train_df.to_csv('train_df.csv')

def train_and_save_models(model, df, features, save_to='models/', n_splits=5):
    if not os.path.exists(save_to): os.mkdir(save_to)
    models_saved = []
    X = df[features]
    y = df["age"]

    kf = KFold(n_splits=n_splits)
    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        df.loc[df.iloc[test_index].index, 'y_pred'] = y_pred.tolist()

        # Save the model for this fold
        model_filename = f'{save_to}/model_fold_{fold}.joblib'
        dump(model, model_filename)
        models_saved.append(model_filename)
        print(f'Model saved for fold {fold}: {model_filename}')

    return df, models_saved


train_df, models_saved = train_and_save_models(
    CatBoostRegressor(silent=True, random_state=42, max_depth=5, iterations=600, l2_leaf_reg=5, cat_features=['gender']), 
    train_df, features, save_to='models/cat_waist_weight_height_hip_ad_filtred')
    # LinearRegression(), 
    # train_df, features, save_to='models/linreg')
mae = mean_absolute_error(train_df['age'], train_df['y_pred'])
r2 = r2_score(train_df['age'], train_df['y_pred'])


def create_and_save_health_stock_percentile_dff(train_df):
    train_df['health_stock'] = train_df['age'] - train_df['y_pred']

    dff = train_df[['health_stock', 'gender', 'age']].dropna()
    selected_ages = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    percentile_df = pd.DataFrame(index=range(1, 101))  # 1-100 percentiles

    for age in selected_ages:
        # Filter data for the current age, allowing a 5-year margin since exact ages may not be present
        age_group_data = dff[(dff['age'] >= age) & (dff['age'] < age + 5)]['health_stock']
        if not age_group_data.empty:
            percentiles = np.percentile(age_group_data, range(1, 101))
            percentile_df[age] = percentiles
        else:
            # If there's no data for the age group, fill with NaN or a placeholder
            percentile_df[age] = [np.nan] * 100

    # Now percentile_df contains percentile values for health_stock within the selected age groups
    percentile_df.to_csv('models/health_stock_percentile_dff.csv')

create_and_save_health_stock_percentile_dff(train_df)

print('MAE', mae, 'R2', r2)
# Plot y_pred against age
plt.figure(figsize=(10, 6))
plt.scatter(train_df['age'], train_df['y_pred'], alpha=0.5)
plt.title('Predicted Age vs. Actual Age')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.plot([train_df['age'].min(), train_df['age'].max()], [train_df['age'].min(), train_df['age'].max()], 'k--') # Diagonal line
plt.show()