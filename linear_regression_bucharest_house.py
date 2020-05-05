# Simple Linear Regression

# Importing the main libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
X = dataset.iloc[:, [1]].values
y = dataset.iloc[:, 6].values
print("Bucharest housing has {} data points with {} variables each.\nWe will use {} feature variable to predict the price".format(*dataset.shape,X.shape[1]))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results; y_pred is a vector of the predicted values
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Surface vs Price (Training set)')
plt.xlabel('Surface')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Surface vs Price (Test set)')
plt.xlabel('Surface')
plt.ylabel('Price')
plt.show()

# Evluating the model using the R^2 score
regressor.score(X_test,y_test)
