#Importing data and libraries
import numpy as np
import matplotlib.pyplot as plt
import mglearn
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
# print(data['feature_names'])
# print(f"Type of data {type(data['data'])}")

# print(f"The shape of the array {data['data'][51:57]}")
# print(f"The shape of the array {data['target'][51:57:]}")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], test_size=0.2, random_state=0
)

# print(f"{X_train.shape}")
# print(f"{X_test.shape}")
# print(f"{y_train.shape}")
# print(f"{y_test.shape}")

#create a dataframe from data in X_train
from pandas.plotting import scatter_matrix 
df = pd.DataFrame(X_train, columns=data.feature_names)

#Create a scatter matrix from the datsets
grr = scatter_matrix(df, c=y_train, figsize=(15, 15), marker='o', 
                        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

X_new = np.array([[6.4, 3.2 ,4.5, 10]])
# print(f"X_new.shape {X_new.shape}")

prediction = knn.predict(X_new)
# print(f"Prediction: {prediction}")
# print(f"Predicted Target: {data['target_names'][prediction]}")

y_pred = knn.predict(X_test)
# print(f"Test set prediction:\n{y_pred}")

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred)
print("--------------------------")
print(f"Mean-Squared Error: {mse}")
print("Test set score: {:.2f}".format(np.mean(y_pred==y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
print("--------------------------")

sepal_l = float(input("Enter the length(cm) of sepal: "))
sepal_w = float(input("Enter the width(cm) of sepal: "))
petal_l = float(input("Enter the length(cm) of petal: "))
petal_w = float(input("Enter the width(cm) of petal: "))

X_input = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
y_pred = knn.predict(X_input)
print(f"New set prediction:\n{y_pred}")
print(f"Predicted Target: {data['target_names'][y_pred]}")

