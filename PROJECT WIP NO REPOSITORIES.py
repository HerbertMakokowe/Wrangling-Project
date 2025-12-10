import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

data1 = pd.read_csv("C:/Users/herbe/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2020.csv")
data2 = pd.read_csv("C:/Users/herbe/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2021.csv")
data3 = pd.read_csv("C:/Users/herbe/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2022.csv")
data4 = pd.read_csv("C:/Users/herbe/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2023.csv")
data5 = pd.read_csv("C:/Users/herbe/Documents/Wrangling-Project/Air Quality/Final Data/annual_aqi_by_county_2024.csv")


complete_data = pd.concat([data1, data2, data3, data4, data5], ignore_index=True)

complete_data.head()
complete_data.tail()

complete_data.describe(include='all')

complete_data.duplicated().sum()


#1) National Trend: How has air quality in the U.S. changed from 2020–2025?

complete_data.describe()
complete_data.shape

complete_data.columns = (complete_data.columns.str.strip().str.lower().str.replace(' ', '_'))

complete_data['year'] = complete_data['year'].astype(int)

complete_data.columns

cols =  ['good_days', 'moderate_days', 'unhealthy_days','very_unhealthy_days', 'hazardous_days', 'median_aqi']

national_trends = complete_data.groupby('year')[cols].mean().reset_index()

plt.plot('year', 'good_days', data=national_trends)
plt.show()


#2) State Comparison: Which states have the best and worst air quality?
# columns we will use are as follows state, good_days, unhealthy_days, days_with_aqi, hazardous_days, very_unhealthy_days, median_aqi

complete_data = complete_data[complete_data['days_with_aqi'] > 0]

complete_data['good_share'] = complete_data['good_days'] / complete_data['days_with_aqi']

complete_data['bad_share'] = (complete_data['unhealthy_days'] + complete_data['hazardous_days'] + complete_data['very_unhealthy_days']) / complete_data['days_with_aqi']

state_summary = (complete_data.groupby('state')[['good_share', 'bad_share', 'median_aqi']].mean().reset_index())

#Now we sort the states so as to see the best and worst ones

best_states = (state_summary.sort_values(['good_share', 'median_aqi'], ascending=[False, True]).head(10))

worst_states = (state_summary.sort_values(['bad_share', 'median_aqi'], ascending=[False, True]).head(10))

best_states
worst_states

plt.bar(best_states['state'], best_states['good_share'])
plt.xticks(rotation=45)
plt.show()

plt.bar(worst_states['state'], worst_states['bad_share'])
plt.xticks(rotation=45)
plt.show()

#What happens in good vs bad states that affects the air?

#3) Pollution Drivers: Which pollutants most strongly predict poor AQI outcomes?

pollutant_cols = ['days_co', 'days_no2', 'days_ozone', 'days_pm2.5', 'days_pm10']
outcome_cols = ['bad_share', 'median_aqi', 'max_aqi']

correlation = complete_data[pollutant_cols + outcome_cols].corr()

pollutant_drivers = correlation.loc[pollutant_cols, outcome_cols]

import matplotlib.pyplot as plt

corr_median = pollutant_drivers['median_aqi']

plt.bar(corr_median.index, corr_median.values)
plt.title("Correlation with Median AQI by Pollutant")
plt.ylabel("Correlation")
plt.xticks(rotation=45)
plt.show()


corr_bad = pollutant_drivers['bad_share']

plt.bar(corr_bad.index, corr_bad.values)
plt.title("Correlation with Bad Share by Pollutant")
plt.ylabel("Correlation")
plt.xticks(rotation=45)
plt.show()


corr_max = pollutant_drivers['max_aqi']

plt.bar(corr_max.index, corr_max.values)
plt.title("Correlation with Max AQI by Pollutant")
plt.ylabel("Correlation")
plt.xticks(rotation=45)
plt.show()


abs_corr = pollutant_drivers.abs()

best_for_bad    = abs_corr['bad_share'].idxmax()
best_for_median = abs_corr['median_aqi'].idxmax()
best_for_max    = abs_corr['max_aqi'].idxmax()

best_for_bad, best_for_median, best_for_max



X = complete_data[pollutant_cols]
y = complete_data['median_aqi']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

model.summary()



# Health-risk days are days when AQI is Unhealthy or worse (EPA: health warnings)
complete_data['health_risk_days'] = (
    complete_data['unhealthy_days'] +
    complete_data['very_unhealthy_days'] +
    complete_data['hazardous_days']
)

complete_data['health_risk_share'] = (
    complete_data['health_risk_days'] / complete_data['days_with_aqi']
)


national_health = (
    complete_data
      .groupby('year')[['health_risk_days', 'health_risk_share']]
      .mean()
      .reset_index()
)

plt.plot('year', 'health_risk_share', data=national_health, marker='o')
plt.title("National Trend in Health-Risk Air Days (2020–2025)")
plt.xlabel("Year")
plt.ylabel("Average Share of Health-Risk Days")
plt.grid(True)
plt.show()


state_health = (
    complete_data
      .groupby('state')[['health_risk_share', 'median_aqi']]
      .mean()
      .reset_index()
)

# Worst states by health risk
worst_health_states = (
    state_health
      .sort_values(['health_risk_share', 'median_aqi'], ascending=[False, False])
      .head(10)
)

worst_health_states


plt.bar(worst_health_states['state'], worst_health_states['health_risk_share'])
plt.xticks(rotation=45)
plt.ylabel("Average Share of Health-Risk Days")
plt.title("States with Highest Exposure to Unhealthy Air (2020–2025)")
plt.show()


outcome_cols = ['health_risk_share', 'median_aqi', 'max_aqi']

correlation = complete_data[pollutant_cols + outcome_cols].corr()
pollutant_drivers = correlation.loc[pollutant_cols, outcome_cols]
pollutant_drivers


corr_health = pollutant_drivers['health_risk_share']

plt.bar(corr_health.index, corr_health.values)
plt.title("Correlation with Health-Risk Share by Pollutant")
plt.ylabel("Correlation")
plt.xticks(rotation=45)
plt.show()



X = complete_data[pollutant_cols]
y = complete_data['median_aqi']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()


coef = model.params
coef

