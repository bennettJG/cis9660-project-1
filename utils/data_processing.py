import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
import json
import io, os
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from census import Census
from us import states
import streamlit as st

random_state = 1701

@st.cache_data
def load_data():
    rents = pd.DataFrame()
    for f in glob.glob("**HomeHarvest*.csv", recursive=True):
      data = pd.read_csv(f)
      if(len(rents) == 0):
        rents = data
      else:
        rents = pd.concat([rents, data], axis=0)
        
    c = Census("aa512886c5449a582d837da8d3a07af66a043fe5")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 9000)
    census_vars = {'B08303': 'Travel Time to Work',
                   'B25004_001E':	'Vacant housing units for rent',
                   'B01003_001E':	'Total Population',
                   'B19013_001E':	'Median Household Income in the Past 12 Months (in 2023 Inflation-Adjusted Dollars)',
                   'B25035_001E':	'Median Year Structure Built',
                   'B01002_001E': 'Median Age'}
    census_data = pd.DataFrame(c.acs5.get(['B08303_002E','B08303_003E', 'B08303_004E', 'B08303_005E', 'B08303_006E', 'B08303_007E', 'B08303_008E', 'B08303_009E', 'B08303_010E', 'B08303_011E', 'B08303_012E',
                             'B08303_013E', 'B25004_001E', 'B01003_001E', 'B19013_001E', 'B25035_001E', 'B01002_001E'], {'for':'zip code tabulation area:*'}))
    census_data.rename({'B25004_001E': 'area_vacancies', 'B01003_001E': 'area_pop', 'B19013_001E': 'med_income', 'B25035_001E': 'med_yearbuilt', 'B01002_001E': 'med_age'}, axis=1, inplace=True)

    census_data['avg_commute'] = (2*census_data['B08303_002E'] + 7*census_data['B08303_003E'] + 12*census_data['B08303_004E'] + 17*census_data['B08303_005E'] +
                                  22*census_data['B08303_006E'] + 27*census_data['B08303_007E'] + 32*census_data['B08303_008E'] + 37*census_data['B08303_009E'] +
                                  42*census_data['B08303_010E'] + 52*census_data['B08303_011E'] + 74.5*census_data['B08303_012E'] + 100*census_data['B08303_013E']) / census_data.loc[:,'B08303_002E':'B08303_013E'].sum(axis=1)
    census_data = census_data.drop(list(census_data.filter(regex='B08303')), axis=1)

    boroughs = pd.read_csv("Boroughs.csv", dtype={'zip':object})
    print(boroughs.columns)
    census_data = pd.merge(census_data, boroughs, right_on='zip', left_on='zip code tabulation area')

    rents = rents[rents['status'] == "FOR_RENT"].dropna(subset=['beds', 'zip_code', 'sqft'])
    rents['zip_code'] = rents['zip_code'].astype('string').str.split(".").str.get(0)
    rents = pd.merge(rents, census_data, left_on='zip_code', right_on='zip')
    rents.fillna({'full_baths': 0, 'half_baths': 0, 'year_built': rents['med_yearbuilt'].astype(int), 'list_price': (rents['list_price_max']-rents['list_price_min'])/2,
                  'parking_garage': 0}, inplace=True)
    rents['baths'] = rents['full_baths'] + 0.5*rents['half_baths']
    rents = rents[(rents['baths'] < 10) & (rents['list_price'] < 1000000)]
    rents['building_age'] = 2025 - rents['year_built']

    rents.dropna(subset=['list_price'], inplace=True)
    return([rents, census_data])

@st.cache_data
def process_data(rents):
    rents_processed = rents.copy()
    rents_processed = rents_processed[['beds', 'baths', 'building_age', 'borough', 'area_vacancies', 'area_pop', 'med_income', 'avg_commute', 'list_price', 'sqft', 'style', 'med_age']]
    rents_processed = pd.get_dummies(rents_processed, columns=['borough', 'style'], drop_first=True)
    numeric_columns = ['beds', 'baths', 'building_age', 'area_vacancies', 'area_pop',	'med_income', 'avg_commute', 'sqft', 'med_age']
    categorical_columns = list(set(rents_processed.columns) - set(numeric_columns) - set(['list_price']))
    X = rents_processed.drop('list_price', axis=1)
    y = np.log(rents_processed['list_price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = pd.concat(
        [pd.DataFrame(scaler.fit_transform(X_train[numeric_columns]), columns=numeric_columns, index=X_train.index),
        X_train[categorical_columns]],
        axis=1
    )
    X_test_scaled = pd.concat(
        [pd.DataFrame(scaler.transform(X_test[numeric_columns]), columns=numeric_columns, index=X_test.index),
        X_test[categorical_columns]],
        axis=1
    )
    return [scaler, X_train_scaled, X_test_scaled, y_train, y_test]
    
@st.cache_data
def train_model(X_train, y_train):
    rf_model = RandomForestRegressor(random_state = random_state)
    rf_model.fit(X_train, y_train)
    return rf_model