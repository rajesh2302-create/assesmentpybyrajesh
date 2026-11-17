import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data set of 200 customers
np.random.seed(42)
n=200
today=pd.to_datetime('2026-11-17')
signup_dates=pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0,1000,n),unit='D')
last_purchase_dates=signup_dates + pd.to_timedelta(np.random.randint(1,400,n),unit='D')
last_purchase_dates=np.minimum(last_purchase_dates,today)

total_spend=np.random.exponential(scale=500, size=n).round(2)
visits=np.random.randint(1,50,size=n)

days_inactive=(today-last_purchase_dates).days
churn_prob=np.where(days_inactive > 180,0.75,0.15)
churned=np.random.binomial(1, churn_prob)

df=pd.DataFrame({'signup_date': signup_dates,'last_purchase_date':last_purchase_dates,'total_spend':total_spend,
                'visits':visits,'churned':churned })
df['signup_dates']=pd.to_datetime(df['signup_date'])
df['last_purchase_date']=pd.to_datetime(df['last_purchase_date'])

df['signup_year'] = df['signup_dates'].dt.year
df['signup_month'] = df['signup_dates'].dt.month
df['signup_weekday'] = df['signup_dates'].dt.weekday

df['days_since_last_purchase'] = (today - df['last_purchase_date']).dt.days
df['spend_per_visit'] = (df['total_spend'] / df['visits']).round(2)

# 3. Log Transform & Plot Before/After Histograms
df['log_total_spend'] = np.log1p(df['total_spend'])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['total_spend'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_title('Before Log Transform (Skewed)')
axes[0].set_xlabel('Total Spend')
axes[0].set_ylabel('Frequency')

axes[1].hist(df['log_total_spend'], bins=20, color='lightgreen', edgecolor='black')
axes[1].set_title('After Log Transform (Normalized)')
axes[1].set_xlabel('Log Total Spend')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# 4. Check Correlation with Target ('churned')
features = ['signup_year', 'signup_month', 'signup_weekday', 
            'days_since_last_purchase', 'spend_per_visit', 'log_total_spend']

corr_table = df[features + ['churned']].corr()[['churned']].sort_values(by='churned', ascending=False)
print("--- Correlation with Churned ---")
print(corr_table)
