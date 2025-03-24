"""Data Cleaning"""

import pandas as pd

house_data = pd.read_csv("real-estate-data.csv")
"""print(house_data.head())
house_data.info()
#house_data.isna().sum()"""

print(house_data.describe())
print(house_data.shape)
house_data.replace("N/A", pd.NA, inplace=True)
house_data_cleaned = house_data.dropna()
house_data_cleaned = house_data_cleaned.copy()
print(house_data_cleaned.shape)

house_data_cleaned['DEN_new'] = house_data_cleaned['DEN'].map({'YES': True, 'no': False})
house_data_cleaned['parking_new'] = house_data_cleaned['parking'].map({'Yes': True, 'N': False})
house_data_cleaned['location'] = list(zip(house_data_cleaned['lt'], house_data_cleaned['lg']))
house_data_cleaned['size_range'] = house_data_cleaned['size'].apply(
    lambda x: tuple(map(int, x.replace(" sqft", "").split('-'))) if isinstance(x, str) and '-' in x else None
)
del house_data_cleaned['size']
del house_data_cleaned['parking']
del house_data_cleaned['lt']
del house_data_cleaned['lg']
del house_data_cleaned['DEN']
del house_data_cleaned['exposure']
del house_data_cleaned['D_mkt']
del house_data_cleaned['ward']

print(house_data_cleaned.head())
