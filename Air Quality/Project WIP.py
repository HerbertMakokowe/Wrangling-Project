import pandas as pd
import statsmodels.api as sm

data1 = pd.read_csv("C:/Users/herbe/OneDrive/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2020.csv")
data2 = pd.read_csv("C:/Users/herbe/OneDrive/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2021.csv")
data3 = pd.read_csv("C:/Users/herbe/OneDrive/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2022.csv")
data4 = pd.read_csv("C:/Users/herbe/OneDrive/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2023.csv")
data5 = pd.read_csv("C:/Users/herbe/OneDrive/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2024.csv")
data6 = pd.read_csv("C:/Users/herbe/OneDrive/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2025.csv")


complete_data = pd.concat([data1, data2, data3, data4, data5, data6], ignore_index=True)

complete_data.head()
complete_data.tail()

complete_data.describe(include='all')

complete_data.duplicated().sum()


#1) National Trend: How has air quality in the U.S. changed from 2020–2025?

#2) State Comparison: Which states have the best and worst air quality?

#3) State Trend: Which states improved or deteriorated the most over time?

#4) Pollution Drivers: Which pollutants most strongly predict poor AQI outcomes?

#5) Policy Targeting: Which counties should be prioritized for PM2.5 reduction efforts?

