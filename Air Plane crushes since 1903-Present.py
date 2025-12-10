import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

data = pd.read_csv("c:/Users/herbe/Documents/Wrangling-Project/Airplane_Crashes_and_Fatalities_Since_1908_t0_2023.csv", encoding='latin-1')

data.columns

data.describe(include='all')

data.info()

# 1) How have plane crashes changed over time?
# 2) How have fatalities per year changed over time?
# 3) How deadly are crashes in terms of survival outcomes?
# 4) Which operators have caused the highest number of fatalities?
# 5) What are the overall characteristics of global plane crashes?


#Data Cleaning Process

data.columns = (data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('#', 'number').str.replace('/', '_'))


#We now need to convert our data types

data['Date'] = pd.to_datetime(data['date'], errors='coerce')

#Convert to numeric columns

num_cols = ['abroad', 'abroad_passengers', 'abroad_crew', 'fatalities', 'fatalities_passengers', 'passengers_crew', 'ground']

for col in num_cols: data[col] = pd.to_numeric(data[col], errors='coerce')


#Creating time_based features

data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day_of_week'] = data['Date'].dt.day_name()
data['decade'] = (data['year'] // 10) * 10


#Where were these planes going?

data[['origin', 'destination']] = data['route'].str.split('-', n=1, expand=True)

#Now we create Survival rate

data['survival_rate'] = data['aboard'] - data['fatalities']
data.loc[data['aboard'] <=0, 'survival_rate'] = None

#Survival category

def classify(row):
    if pd.isna(row['survival_rate']):
        return "Unknown"
    if row['survival_rate'] == 0:
        return "No survivors"
    if row['survival_rate'] == 1:
        return "All survived"
    return "Some survived"

data['survival_category'] = data.apply(classify, axis=1)


#Descriptive statistics

crashes_per_year = data.groupby('year').size()

fatalities_per_year = data.groupby('year')['fatalities'].sum()

operator_counts = data['operator'].value_counts()

deadliest_operators = data.groupby('operator')['fatalities'].sum().sort_values(ascending=False)

data['fatalities_per_crash'] = data['fatalities']

survival_distribution = data['survival_category'].value_counts()

#Visualizations

#Crashes Per Year

plt.figure(figsize=(12,5))
crashes_per_year.plot()
plt.title("Crashes Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Crashes")
plt.show()

#Fatalities Per Year
plt.figure(figsize=(12,5))
fatalities_per_year.plot(color='red')
plt.title("Fatalities Per Year")
plt.xlabel("Year")
plt.ylabel("Total Fatalities")
plt.show()

#Survival Categories

data['survival_category'].value_counts().plot(kind='bar')
plt.title("Crash Survival Outcomes")
plt.ylabel("Number of Crashes")
plt.xticks(rotation=360, ha='right')
plt.show()

#Top 10 Deadliest Operators
deadliest_operators.head(5).plot(kind='bar', figsize=(8,6))
plt.title("Top 10 Deadliest Operators")
plt.ylabel("Total Fatalities")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.show()



#Summary Statistics

total_crashes = len(data)
total_fatalities = data['fatalities'].sum()
avg_fatalities_per_crash = data['fatalities'].mean()

crashes_2000_onward = data[data['year'] >= 2000].shape[0]
fatalities_2000_onward = data[data['year'] >= 2000]['fatalities'].sum()

total_crashes
total_fatalities
avg_fatalities_per_crash
crashes_2000_onward 
fatalities_2000_onward