# BucharestHousePrice
Repository with machine learning models to predict the house price in Bucharest

==Introduction==

We will predict the house price based on some or all the features in the dataset translated as follows:

Nr Camere   = No of Rooms
Suprafata   = Surface
Etaj        = Floor
Total Etaje = Total Floors
Sector      = Sector (Bucharest is divided into 6 zones/sectors)
Score       = Score
Pret        = Price

* Dataset source: https://www.kaggle.com/denisadutca/bucharest-house-price-dataset#Bucharest_HousePriceDataset.csv
* The code design is based on the templates from Udemy course "Machine Learning A-Z: Hands-On Python & R In Data Science"


==Observations==

1) Afeter running the analysis and preprocessing part we identified that:
 - there is a strong positive correlation between the Surface-Price and Rooms-Price
 - there is a negative correlation between Sector-Price and Score-Price
 - there is a weak correlation between Floor-Price and NoFloors-Price
 Therefore we will take into considerations all the features for some models (Rand Forest and SVR) and only some features for other models

2) The feature Sector needs to be encoded as it's deffinitelly categorical. The Score can be considered categorical or not, so it doesn't necesaary nedd to be encoded. In Random Forest it's not encoded and in SVR it's encoded.


==Short explanations==

regressor - is the machine and it learns from the training set that we feed it with
y_pred - is a vector of the predicted values


==Results==

Simple Linear Regression
R2 = 0.6353670298177932

Multiple Linear Regression
R2 = 0.7432251693402356

SVR
R2 = 0.7922608014919587

Random Forest Regression
R2 = 0.7692560015250947 with 10 trees
R2 = 0.7956232443301254 with 300 trees
R2 = 0.7961313191363139 with 500 trees
R2 = 0.7963226764851472 with 700 trees -> optimum choice that maximizes the R2
R2 = 0.7959752371152921 with 1000 trees
