# Neural_Network_Charity_Analysis

## Results:
### Data Preprocessing:

* The target variable for the model is "IS_SUCCESSFUL", which indicates whether a loan application was approved or not.
* The features for the model include all of the remaining variables in the dataset, such as the applicant's income, employment status, and credit score.
* Variables that were removed from the input data include "ID" and "NAME", as they do not provide any useful information for the model.

## Compiling, Training, and Evaluating the Model:

* For the neural network model, I selected 80 neurons for the first hidden layer, 30 neurons for the second hidden layer, and 1 neuron for the output layer. An average of 2 neurons per input, the activation function used for all layers is "relu" and "sigmoid" respectively.
* The model was able to achieve a target performance of around 72% accuracy on the test data. Unfortunately, I was not able to optimize the model to perform better than 75%
* To try and increase model performance, I used the early stopping method and optimized the number of neurons and layers, and also used different activation functions. I even tried using RandomForestClassifier and Logistic Regression but not able to achieve results.

## Summary:
The deep learning model created for AlphabetSoup was able to predict loan application success with an accuracy of around 72%. While this is an ok performance, there may be room for improvement by trying other models such as Random Forest, or by gathering more data to train the model on. Overall, this model can be a useful tool for AlphabetSoup to predict the success of loan applications.