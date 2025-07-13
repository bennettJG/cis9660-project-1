# CIS 9660 Project 1

Streamlit app URL: https://bennettjg-cis9660-project-1-data-exploration-pzvn5v.streamlit.app/
There are three tabs - the first allows a user to predict an apartment rental price based on their input, the other two allow exploring the data by ZIP code or apartment attributes.

To run the Streamlit app locally, clone the repository, navigate to the folder in a terminal, and use the command `streamlit run data_exploration.py`.

The Streamlit app implements only the best-performing model. Fitting and comparison code for all models tested can be found in the RentalDataExploration.ipynb Jupyter notebook in the root directory. Data is in the data folder, and the script used for data scraping is in utils/home_harvest.py.

Required libraries are listed in requirements.txt and can be installed through `pip`.

## Resources used:
- https://github.com/ZacharyHampton/HomeHarvest for data scraping
- https://github.com/erikgregorywebb/nyc-housing/blob/master/Data/nyc-zip-codes.csv for NYC zip code - borough correspondence
- https://www.kaggle.com/datasets/saidakbarp/nyc-zipcode-geodata for NYC GeoJSON data (used to display the map on the "Explore by ZIP code" tab of the Streamlit app)
- Streamlit, Plotly, and Scikit-learn documentation