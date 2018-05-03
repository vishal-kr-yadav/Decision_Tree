import pandas as pd

file_path='./kc_house_data.csv'

data=pd.read_csv(file_path)

# this is called X
attribute_taken_for_predictor=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','sqft_living15']
X=data[attribute_taken_for_predictor]
Y=data.price

from sklearn.tree import DecisionTreeRegressor

# Defining the model
housing_model=DecisionTreeRegressor()

# Fit the model
housing_model.fit(X,Y)

DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_impurity_decrease=0.0,
           min_impurity_split=None, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           presort=False, random_state=None, splitter='best')
predicted_home_prices = housing_model.predict(X)
from sklearn.metrics import mean_absolute_error

# print(mean_absolute_error(Y, predicted_home_prices))
# validation data.
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, Y,random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print("******Printing the mean_absolute_error---->",mean_absolute_error(val_y, val_predictions))
