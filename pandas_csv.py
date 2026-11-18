import io
import pandas as pd 
import numpy as np
csv_data="""id,name,department,salary,city
1,arun,sales,30000,chennai
2,rajesh,hr,25000,karaikal
3,suthan,finance,35000,peralam
4,aakash,IT,60000,chennai
5,ajesh,IT,70000,chennai
6,mukesh,sales,35000,kollumangudi
7,ramesh,IT,70000,peralam
8,vennila,hr,40000,thanjavur"""
with open("employess.csv","w")as f:
    f.write(csv_data)
df=pd.read_csv('employess.csv')
sal_above_50k=len(df[df["salary"]>50000])
print(f"sal above 50k:{sal_above_50k}employees")
chennai_employees=df.loc[df["city"]=="chennai","name"]
print(f"chennai:{','.join(chennai_employees)}")
it_dept=df.loc[df["department"]=="IT","name"]
print(f"it department:{','.join(it_dept)}")