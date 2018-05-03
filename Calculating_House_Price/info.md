[Reference link](https://www.kaggle.com/dansbecker/your-first-scikit-learn-model)

columns_name=['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'].
       
You will use the scikit-learn library to create your models. When coding, this library is written as sklearn, as you will see in the sample code. Scikit-learn is easily the most popular library for modeling the types of data typically stored in DataFrames.
The steps to building and using a model are:

        Define:What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
        Fit: Capture patterns from provided data. This is the heart of modeling.
        Predict: Just what it sounds like
        Evaluate: Determine how accurate the model's predictions are.

###Model_Validation
        There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE). Let's break down this metric starting with the last word, error.
        The prediction error for each house is: 
        error=actual−predicted
        With the MAE metric, we take the absolute value of each error. This converts each error to a positive number. We then take the average of those absolute errors. 
###validation data.
        use those to test the model's accuracy on data it hasn't seen before. This data is called validation data.

#####if the depth of the DT is very deep then ,leaf node become huge due to that lots of condition is there for predicting a house price...So its a problem of a overfitting...where a model matches the training data almost perfectly, but does poorly in validation and other new data.

####When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting



#########we want to find the sweet spot between underfitting and overfitting. 

       But the max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting.
       
       We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:

