import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data=pd.DataFrame({'age':[22,25,30,35,40,45,50,55,60,65],
                   'income':[25000,32000,45000,52000,60000,75000,82000,95000,110000,125000]})
x_tr_demo,x_te_demo=train_test_split(data,test_size=0.3,random_state=42)
std_scaler=StandardScaler()
x_tr_std=std_scaler.fit_transform(x_tr_demo)
x_te_std=std_scaler.fit_transform(x_te_demo)
print("standard scaler (train set)")
print("mean:",np.round(x_tr_std.mean(axis=0),4))
print("std dev:",np.round(x_tr_std.std(axis=0),4))

mm_scaler=MinMaxScaler()
x_tr_mm=mm_scaler.fit_transform(x_tr_demo)
x_te_mm=mm_scaler.transform(x_te_demo)
print("minmaxscaler")
print("min:",np.round(x_tr_mm.min(axis=0),4))
print("max:",np.round(x_tr_mm.max(axis=0),4))

cancer=load_breast_cancer()
x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,test_size=0.25,random_state=8)

knn_raw=KNeighborsClassifier(n_neighbors=5).fit(x_train,y_train)
dt_raw=DecisionTreeClassifier(random_state=42).fit(x_train,y_train)

acc_knn_raw=accuracy_score(y_test,knn_raw.predict(x_test))
acc_dt_raw=accuracy_score(y_test,dt_raw.predict(x_test))

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

knn_scaled=KNeighborsClassifier(n_neighbors=5).fit(x_train_scaled,y_train)
dt_scaled=DecisionTreeClassifier(random_state=42).fit(x_train_scaled,y_train)

acc_knn_scaled=accuracy_score(y_test,knn_scaled.predict(x_test_scaled))
acc_dt_scaled=accuracy_score(y_test,dt_scaled.predict(x_test_scaled))

print("3&4--model comparison")
print(f"knnwithoutscaling:{acc_knn_raw:.2f}|with scaling:{acc_knn_scaled:.2f}")
print(f"decisiontreewithoutscaling:{acc_dt_raw:.2f}|with scaling:{acc_dt_scaled:.2f}")