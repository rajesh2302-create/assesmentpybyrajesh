import pandas as pd
import numpy as np
df=pd.DataFrame({
                  'employee_id':[1,2,3,4],
                 'name':['rajesh','priya','vikram','sneha',],
                 'dept':['it','hr','finance','sales',],
                 'city':['chennai','mumbai','chennai','bangalore',],
                 'gender':['m','f','m','f'],
                 'contract_type':['full-time','part-time','full-time','part-time'],
                 'salary':[56000,89000,98000,76000],
                 'experience':[5.0,3.5,2.0,3.0],
                 'performance':[82,78,93,65]})
df_budget=pd.DataFrame({
    'dept':['it','hr','sales','finance'],
    'annual_budg_used':[500000,200000,500000,800000]
})
print("summary statics")
summary_status=df[['salary','experience','performance']].describe().loc[['mean','50%','min','max','std']]
print(summary_status,"\n")
print("2. group comparison by dept")
dept_comparison=df.groupby('dept').agg(avg_salary=('salary','mean'),
                                       avg_performance=('performance','mean')).reset_index()
print(dept_comparison,"\n")
print("\n" + "="*60 + "\n")
print("---3.correlation top relation")
corr_matrix=df[['salary','experience','performance']].corr()
print(corr_matrix,"\n")
print("merged table")
merge_df=pd.merge(df,df_budget,on='dept',how='inner')
print(merge_df[['name','dept','salary','annual_budg_used']],"\n")
print("5.pivot_table")
pvt_table=pd.pivot_table(df,values='salary',index='dept',columns='gender',aggfunc='mean',fill_value=0)
print(pvt_table)