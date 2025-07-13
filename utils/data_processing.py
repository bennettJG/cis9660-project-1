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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from census import Census
from us import states
import streamlit as st

random_state = 1701

@st.cache_data
def load_data():
    # Read in data files
    rents = pd.DataFrame()
    for f in glob.glob("**data/HomeHarvest*.csv", recursive=True):
      data = pd.read_csv(f)
      if(len(rents) == 0):
        rents = data
      else:
        rents = pd.concat([rents, data], axis=0) 
    # Get census data for ZIP codes
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
    # Calculate average commute time based on ranges in survey options,
    # drop original columns once average is computed
    census_data['avg_commute'] = (2*census_data['B08303_002E'] + 7*census_data['B08303_003E'] + 12*census_data['B08303_004E'] + 17*census_data['B08303_005E'] +
                                  22*census_data['B08303_006E'] + 27*census_data['B08303_007E'] + 32*census_data['B08303_008E'] + 37*census_data['B08303_009E'] +
                                  42*census_data['B08303_010E'] + 52*census_data['B08303_011E'] + 74.5*census_data['B08303_012E'] + 100*census_data['B08303_013E']) / census_data.loc[:,'B08303_002E':'B08303_013E'].sum(axis=1)
    census_data = census_data.drop(list(census_data.filter(regex='B08303')), axis=1)

    # Read in boroughs dataset and join on ZIP code as an additional discrete feature
    boroughs = pd.read_csv("data/Boroughs.csv", dtype={'zip':object})
    census_data = pd.merge(census_data, boroughs, right_on='zip', left_on='zip code tabulation area')

    # Clean and preprocess rents data. Drop any rows for which list price cannot be computed
    rents = rents[rents['status'] == "FOR_RENT"].dropna(subset=['beds', 'zip_code', 'sqft'])
    rents['zip_code'] = rents['zip_code'].astype('string').str.split(".").str.get(0)
    rents = pd.merge(rents, census_data, left_on='zip_code', right_on='zip')
    rents.fillna({'full_baths': 0, 'half_baths': 0, 'year_built': rents['med_yearbuilt'].astype(int), 'list_price': (rents['list_price_max']-rents['list_price_min'])/2,
                  'parking_garage': 0}, inplace=True)
    rents['baths'] = rents['full_baths'] + 0.5*rents['half_baths']
    rents = rents[(rents['baths'] < 10) & (rents['list_price'] < 1000000)]
    rents['building_age'] = 2025 - rents['year_built']

    rents.dropna(subset=['list_price'], inplace=True)
    rents.drop_duplicates(subset=['property_id'], inplace=True)
    return([rents, census_data])
    
@st.cache_data
def process_data(rents):
    # Limit to features chosen for model. Specify numeric and categorical columns
    # for proper scaling (of numeric) and encoding (of discrete) values.
    rents_processed = rents.copy()
    rents_processed = rents_processed[['beds', 'baths', 'building_age', 'borough', 'area_vacancies', 'area_pop', 'med_income', 'avg_commute', 'list_price', 'sqft', 'style', 'med_age']]
    numeric_columns = ['beds', 'baths', 'building_age', 'area_vacancies', 'area_pop',	'med_income', 'avg_commute', 'sqft', 'med_age']
    categorical_columns = list(set(rents_processed.columns) - set(numeric_columns) - set(['list_price']))
    # Separate target (list price) from features, split into train and test sets, preprocess
    X = rents_processed.drop('list_price', axis=1)
    y = np.log(rents_processed['list_price'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    ct = ColumnTransformer(
    [("text_preprocess", OneHotEncoder(drop='first', sparse_output=False), categorical_columns),
    ("num_preprocess", StandardScaler(), numeric_columns)],
    verbose_feature_names_out = False
    )
    # PCA to better capture the relationship between bedrooms, bathrooms, and area
    ct_pca = ColumnTransformer(
    [("size", PCA(n_components=2), ['beds', 'baths', 'sqft'])],
    remainder="passthrough",
    verbose_feature_names_out = False
    )
    ct.set_output(transform='pandas')
    ct_pca.set_output(transform='pandas')
    pipe = Pipeline(steps=[('basic_transform', ct), ('pca', ct_pca)])
    X_train_scaled = pipe.fit_transform(X_train)
    X_test_scaled = pipe.transform(X_test)
    # Return pipeline along with training and test sets so that
    # it can be used on user-input data.
    return [pipe, X_train_scaled, X_test_scaled, y_train, y_test]
    
@st.cache_data
def train_model(X_train, y_train):
    # Including only the model I ended up choosing here.
    # Other models tested are in the Jupyter notebook.
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model
    
@st.cache_data
def get_metrics(_model, X_test, y_test):
    y_pred = _model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2