import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
import json
import io, os
import glob
import streamlit as st
import plotly.express as px
import utils.data_processing
import weighted # from wquantiles
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import math, re

random_state = 1701

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    padding-right: 20px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
# NYC geojson from https://www.kaggle.com/datasets/saidakbarp/nyc-zipcode-geodata

st.set_page_config(page_title="NYC Rental Listings Dashboard", layout="wide")

raw_data, census_data = utils.data_processing.load_data()
scaler, X_train, X_test, y_train, y_test = utils.data_processing.process_data(raw_data)
rf_model = utils.data_processing.train_model(X_train, y_train)

with open('data/nyc_zcta.geojson') as fp:
    zipcodes = json.load(fp)

tab1, tab2, tab3 = st.tabs(["Predict apartment rent", "Explore by ZIP code", "Explore by apartment characteristics"])
with tab1:
    st.markdown("# Predict apartment rent")
    st.markdown("#### Apartment attributes:")
    #st.dataframe(raw_data)
    col1, col2, col3 = st.columns(3)
    with col1:
        beds_input = st.slider(
            "Bedrooms",
            min_value=0,
            max_value=int(max(raw_data['beds'])),
            value=2,
            step=1
        )
        baths_input = st.slider(
            "Bathrooms",
            min_value=0.0,
            max_value=max(raw_data['baths']),
            value=2.5,
            step=0.5, format="%0.1f"
        )
    with col2:
        zip_input = st.selectbox("ZIP code", 
            census_data['zip code tabulation area'].dropna().sort_values().unique(),
            index=65
            )
        style_input = st.selectbox("Style", 
            raw_data['style'].dropna().sort_values().unique()
            )
    with col3:
        sqft_input = st.slider(
            "Area (square feet)",
            min_value=int(min(raw_data['sqft'])),
            max_value=int(max(raw_data['sqft'])),
            value=int(raw_data['sqft'].median()),
            step=1
        )
        bldgage_input = st.slider(
            "Building age (years)",
            min_value=int(min(raw_data['building_age'])),
            max_value=int(max(raw_data['building_age'])),
            value=int(raw_data['building_age'].median()),
            step=1
        )
    st.markdown("---")
    area_data = census_data[census_data['zip code tabulation area'] == zip_input]
    input_data = pd.concat([pd.DataFrame({'beds':beds_input, 'baths':baths_input,
        'style':style_input, 'sqft':sqft_input,
        'building_age':bldgage_input}, index=[0]), area_data.reset_index()], axis=1).drop(
        ['index', 'zip code tabulation area', 'zip'], axis=1)

    numeric_columns = ['beds', 'baths', 'building_age', 'area_vacancies', 'area_pop',	'med_income', 'avg_commute', 'sqft', 'med_age']
    input_processed = scaler.transform(input_data)
    #st.dataframe(input_processed)
    predicted_rent = math.exp(rf_model.predict(input_processed)[0])
    st.markdown("# Predicted Rent:")
    st.success(f"${predicted_rent:,.2f} / month"+"\n===", width = 800)
    
    st.markdown("---")
    st.header("About the model")
    col1, col2 = st.columns(2)
    with col1:
        mse, rmse, mae, r2 = utils.data_processing.get_metrics(rf_model, X_test, y_test)
        st.markdown("### Performance on test set:")
        st.markdown(f"**Root Mean Squared Error: {rmse:,.4f}**")
        st.markdown(f"**Mean Squared Error: {mse:,.4f}**")
        st.markdown(f"**Mean Absolute Error: {mae:,.4f}**")
        st.markdown(f"**R-squared: {r2:,.4f}**")
    with col2:
        st.markdown("### Top Features")
        topfeatures = pd.Series(abs(rf_model.feature_importances_), index=X_train.columns)
        topfeatures = topfeatures.sort_values(ascending=False).head(10).reset_index()
        topfeatures.replace({'index':{'pca0':'Overall size (size component 0)',
            'pca1':'Rooms other than beds/baths (size component 1)', 'avg_commute':'Average commute time (ZIP code)', 
            'med_income':'Median household income (ZIP code)', 'building_age':'Building age',
            'med_age':'Median resident age (ZIP code)', 
            'area_vacancies':'Vacant housing units (ZIP code)', 'area_pop':'Population (ZIP code)',
            'style_CONDOS':'Unit is a condo', 'style_SINGLE_FAMILY':'Unit is a single-family dwelling'}},
            inplace=True)
        topfeatures.columns = ["Feature", "Importance Score"]
        st.dataframe(topfeatures)
        
with tab2:
    st.markdown("# Explore by ZIP code")
    st.markdown("#### Plot controls")
    st.markdown("* Click to select a ZIP code\n* Shift + click to select multiple ZIP codes\n* Double-click to reset selection and view information for all ZIP codes")
    zip_df = raw_data.copy()
    zip_codes = zip_df['zip_code']

    col1, col2 = st.columns(2)
    with col1:
        fig = px.choropleth_map(zip_df, geojson=zipcodes, 
        locations='zip_code',
            featureidkey="properties.postalCode",
            #color='med_age', 
            #color_continuous_scale="Viridis",
            #range_color=(0, 80),
            map_style="carto-positron",
            zoom=9.5, center = {"lat": 40.730610, "lon": -73.935242},
            opacity=1,
            labels={}
        )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width = 600, showlegend=False)
        nyc_map = st.plotly_chart(fig, on_select="rerun", selection_mode=["points", "box", "lasso"])
        points = nyc_map["selection"].get("points", [])
        if len(points) > 0:
            zip_codes=[]
            for i in range(len(points)):
                zip_codes.append(
                    points[i]["properties"].get("postal_code", None)
                )
            points = []
            zip_df = zip_df[zip_df['zip_code'].isin(zip_codes)]

    with col2:
        st.markdown("### Area data")
        zip_census_data = zip_df[['zip_code', 'area_pop', 'med_income', 'avg_commute', 'area_vacancies','med_age']].groupby('zip_code').first().reset_index()
        zip_census_data['weighted_commute'] = (zip_census_data['avg_commute'] *
            zip_census_data['area_pop']
        )/(sum(zip_census_data['area_pop']))
        col2a, col2b = st.columns(2)
        with col2a:
            # For averages, weight metrics like commute time etc. by population
            st.metric("Median Income", f'${weighted.median(zip_census_data['med_income'], 
                zip_census_data['area_pop']):,.0f}')
            st.metric("Median Age", f'{weighted.median(zip_census_data['med_age'], 
                zip_census_data['area_pop']):,.1f}')
            st.metric("Population", f'{sum(zip_census_data['area_pop']):,.0f}')
        with col2b:
            st.metric("Vacant housing units", f'{sum(zip_census_data['area_vacancies']):,.0f}')
            st.metric("Average Commute Time (min)", f'{sum(zip_census_data['weighted_commute']):,.1f}')
        st.markdown("Source: 2023 American Community Survey 5-year estimates")
    st.markdown("---")    
    st.markdown("### Listings data")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("# Listings", len(zip_df))
        bedroom_counts = (zip_df.groupby('beds').count()['zip_code'].reset_index().
        rename(columns={'zip_code':'Count'}))
        fig = px.bar(bedroom_counts, 
            x="beds", y="Count", title="# Bedrooms")
        fig.update_xaxes(tickmode='linear')
        st.plotly_chart(fig, use_container_width=True)
    with col4: 
        st.metric("Average List Price", 
        f'${zip_df['list_price'].mean():,.2f} / month')
        bathroom_counts = (zip_df.groupby('baths').count()['zip_code'].reset_index().
        rename(columns={'zip_code':'Count'}))
        fig = px.bar(bathroom_counts, 
            x="baths", y="Count", title="# Bathrooms")
        fig.update_xaxes(tickmode='linear')
        st.plotly_chart(fig, use_container_width=True)  
    with col5:
        st.metric("Average Size", 
        f'{zip_df['sqft'].mean():,.0f} square feet')
        fig = px.box(zip_df, y='building_age', labels=dict(building_age='Age (years)'),
        title = "Building Age")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("(Building age imputed using ZIP code median when not listed.)")
    st.markdown("---")
    fig = px.scatter(zip_df, y='list_price', x='sqft', title="List price by Area",
        labels=dict(list_price="List Price ($/month)", sqft="Area (square feet)"))
    st.plotly_chart(fig, use_container_width=False)
    
    with tab3:
        filtered_df = raw_data.copy()
        st.markdown("# Explore by apartment characteristics")
        st.markdown("#### Filters:")
        col1, col2, col3 = st.columns(3)
        with col1:
            beds_filter = st.slider(
                "Bedrooms",
                min_value=0,
                max_value=int(max(raw_data['beds'])),
                value=(0, int(max(raw_data['beds']))),
                step=1
            )
            baths_filter = st.slider(
                "Bathrooms",
                min_value=0.0,
                max_value=max(raw_data['baths']),
                value=(0.0, max(raw_data['baths'])),
                step=0.5, format="%0.1f"
            )
        with col2:
            borough_filter = st.multiselect("Borough", 
                raw_data['borough'].dropna().sort_values().unique(),
                default=raw_data['borough'].dropna().sort_values().unique()
                )
            style_filter = st.multiselect("Style", 
                raw_data['style'].dropna().sort_values().unique(),
                default=raw_data['style'].dropna().sort_values().unique()
                )
        with col3:
            sqft_filter = st.slider(
                "Area (square feet)",
                min_value=int(min(raw_data['sqft'])),
                max_value=int(max(raw_data['sqft'])),
                value=(int(min(raw_data['sqft'])), int(max(raw_data['sqft']))),
                step=1
            )
            bldgage_filter = st.slider(
                "Building age (years)",
                min_value=int(min(raw_data['building_age'])),
                max_value=int(max(raw_data['building_age'])),
                value=(int(min(raw_data['building_age'])), int(max(raw_data['building_age']))),
                step=1
            )
        filtered_df = filtered_df[
            (filtered_df['beds'] >= beds_filter[0]) &
            (filtered_df['beds'] <= beds_filter[1]) &
            (filtered_df['baths'] >= baths_filter[0]) &
            (filtered_df['baths'] <= baths_filter[1]) &
            (filtered_df['sqft'] >= sqft_filter[0]) &
            (filtered_df['sqft'] <= sqft_filter[1]) &
            (filtered_df['building_age'] >= bldgage_filter[0]) &
            (filtered_df['building_age'] <= bldgage_filter[1]) &
            filtered_df['borough'].isin(borough_filter) &
            filtered_df['style'].isin(style_filter)
        ]
        st.markdown("---")
        if (len(filtered_df) == 0):
            st.markdown("# No listings found for specified filters!")
            st.markdown("## Adjust your selections to view pricing metrics")
        else:
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Listings", len(filtered_df))
                fig = px.box(filtered_df,
                    title = "Price by # Bedrooms",
                    x='beds',
                    y='list_price'
                )
                fig.update_xaxes(tickmode='linear')
                st.plotly_chart(fig)
                fig2 = px.box(filtered_df,
                    title = "Price by # Bathrooms",
                    x='baths',
                    y='list_price'
                )
                fig2.update_xaxes(tickmode='linear')
                st.plotly_chart(fig2)
            with col5:
                st.metric("Average List Price", 
                f'${filtered_df['list_price'].mean():,.2f} / month')
                fig = px.box(filtered_df,
                    title = "Price by Borough",
                    x='borough',
                    y='list_price'
                )
                st.plotly_chart(fig)
                fig2 = px.box(filtered_df,
                    title = "Price by Style",
                    x='style',
                    y='list_price'
                )
                st.plotly_chart(fig2)
            with col6:
                st.metric("Average Size", 
                f'{zip_df['sqft'].mean():,.0f} square feet')
                fig = px.scatter(filtered_df,
                    title = "Price by Area",
                    x='sqft',
                    y='list_price'
                )
                st.plotly_chart(fig)
                fig2 = px.scatter(filtered_df,
                    title = "Price by Building Age",
                    x='building_age',
                    y='list_price'
                )
                st.plotly_chart(fig2)