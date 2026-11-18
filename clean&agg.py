import io
import pandas as pd
import numpy as np
orders_cv="""id,customerid,amount
1,101,500
2,102,
3,101,700
4,103,300
5,104,900"""
with open("customer.csv","w")as f:
    f.write(orders_cv)
df=pd.read_csv('customer.csv')    
print(f"missing amount{df['amount'].isna().sum()}|duplicate:{df['customerid'].duplicated().sum()}")
median_value=df['amount'].median()
df['amount']=df['amount'].fillna(median_value)
print("median",median_value)
df=df.drop_duplicates()
df['id']=df['id'].astype(str)
df['amount']=pd.to_numeric(df['amount'])
result=df.groupby('customerid').agg(total=('amount','sum'),orders=('id','count'))
print(result)

