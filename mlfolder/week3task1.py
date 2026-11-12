import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

#load data set split 80/20
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
#supervised learning
model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

acc=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred,average='macro')
cm=confusion_matrix(y_test,y_pred)

print(f"accuracy:{acc:.2f}|F1(macro):{f1:.2f}")
print("confusion matrix:")
print(cm)

#unsupervised learning
kmeans=KMeans(n_clusters=3,random_state=0,n_init=10)
clusters=kmeans.fit_predict(x)
_, counts=np.unique(clusters,return_counts=True)
print("kmeans ckuster size:",counts.tolist())

#compare cluster with real spieces
print("\ncluster vs real spieces")
print(pd.crosstab(clusters,y, rownames=['cluster'],colnames=['real spieces']))