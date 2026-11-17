import pandas as pd
import numpy as np
from sklearn .impute import SimpleImputer,KNNImputer
from sklearn .preprocessing import OneHotEncoder,OrdinalEncoder,LabelEncoder

df=pd.DataFrame({'age':[25,None,35,45,None],
                 'income':[25000,35000,None,80000,70000],
                 'city':['chennai','mumbai',None,'chennai','karaikal'],
                 'size':['s','m','l','s','l'],
                 'purchased':['no','yes','yes','no','yes']})
print(df)

df_simple=df.copy()
median_imputer=SimpleImputer(strategy='median')
df_simple[['age','income']]=median_imputer.fit_transform(df_simple[['age','income']])
print("simple imputer")
print(df_simple)

df_knn=pd.DataFrame({'age':[25,None,35,45,None],
                 'income':[25000,35000,None,80000,70000],})
knn_imputer=KNNImputer(n_neighbors=2)
df_knn[['age','income']]=knn_imputer.fit_transform(df_knn[['age','income']])
print("knn imputer")
print(df_knn[['age','income']])

df_encoded=df_simple.copy()
ohe=OneHotEncoder(sparse_output=False)
city_encoded=ohe.fit_transform(df_encoded[['city']])
city_cols=ohe.get_feature_names_out(['city'])
city_df=pd.DataFrame(city_encoded,columns=city_cols)
df_encoded=pd.concat([df_encoded.drop('city',axis=1),city_df],axis=1)
print("encoded")
print(df_encoded)
ordinal_enc=OrdinalEncoder(categories=[['s','m','l']])
df_encoded[['size']]=ordinal_enc.fit_transform(df_encoded[['size']])
print("after ordinal encoding")
print(df_encoded)
le=LabelEncoder()
df_encoded['purchased']=le.fit_transform(df_encoded['purchased'])
print("final encoded data target")
print(df_encoded)