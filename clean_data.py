import pandas as pd
import numpy as np
np.random.seed(42)
base_ids=[f"EMP_{1000+i}"for i in range(1470)]
base_age= np.random.choice([25,30,45,np.nan,35,150],size=1470,p=[0.4,0.3,0.2,0.05,0.04,0.01])
base_salary=np.random.choice(['50000','60000','75000','missing','1200'],size=1470)
base_date=np.random.choice(['2023-01-15','15-08-2022','2024/05/20'],size=1470)
base_dept=np.random.choice(['sales','Sales','IT','it','hr',np.nan],size=1470)
df_base=pd.DataFrame({'employee_id':base_ids,'age':base_age,'salary':base_salary,'join_date':base_date,'department':base_dept})
df_dups=df_base.iloc[:10].copy()
df=pd.concat([df_base,df_dups],ignore_index=True)

#df=pd.DataFrame(data)
initial_row=len(df)
initial_missing=df.isnull().sum().sum() + (df['salary'].str.strip()=='missing').sum()
initial_duplicates=df.duplicated().sum()
df=df.drop_duplicates().reset_index(drop=True)
df['age']=df['age'].fillna(df['age'].median())
df['salary']=pd.to_numeric(df['salary'],errors='coerce')
df['salary']=df['salary'].fillna(df['salary'].median())
df['join_date']=pd.to_datetime(df['join_date'],format='mixed').dt.strftime('%Y-%m-%d')
df['department']=df['department'].replace(['nan','NAN','Nan'],np.nan)
most_frequent_dept=df['department'].mode()[0]
df['department']=df['department'].fillna(most_frequent_dept)
final_rows=len(df)
final_missing=df.isnull().sum().sum()
final_duplicates=df.duplicated().sum()
df.to_csv('clean_data.csv',index=False)
summary_data={'metric':['total rows','missing values','duplicated rows'],
              'before cleaning': [initial_row,initial_missing,initial_duplicates],
              'after cleaning':[final_rows,final_missing,final_duplicates]}
summary_df=pd.DataFrame(summary_data)
print("data cleaning summary")
print(summary_df.to_string(index=False))
print("cleaned data successfully saved to 'clean_data.csv'")