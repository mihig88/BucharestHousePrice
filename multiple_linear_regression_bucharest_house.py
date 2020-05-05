# Multiple Linear Regression

# Importing the main libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Bucharest_HousePriceDataset.csv')

# Analysis and preprocessing
sns.pairplot(dataset, height=2.5)
plt.tight_layout()
corr_mat = dataset.corr()
hm = sns.heatmap(corr_mat, annot=True, annot_kws={'size': 10})
bottom, top = hm.get_ylim()
hm.set_ylim(bottom + 0.5, top - 0.5)

# Choosing the features and dependent variable
X = dataset.iloc[:, [0,1,4,5]].values
y = dataset.iloc[:, 6].values
print("Bucharest housing has {} data points with {} variables each.\nWe'll use {} feature variables to predict the price.".format(*dataset.shape,X.shape[1]))

# Encoding categorical data - for this model Sector and Score
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Multiple Linear Regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Evluating the model with R^2 score and visualizing the test and predicted results together
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true (y_true) and predicted (y_predict) values based on the metric chosen. """  
    score = r2_score(y_true, y_predict)    
    return score
print("R2 score is = ", performance_metric(y_test, y_pred))
viz_res = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)
print(viz_res)
