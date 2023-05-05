![Banner](EmployeeChurnBanner.png)

# Employee Churn Prediction

This repository aims to predict employee churn using five different machine learning models (KNN, Random Forest, Naive Bayes, Logistic Regression, and a simple MLP Neural Network) and analyze their results using the [HR Analytics](https://www.kaggle.com/datasets/giripujar/hr-analytics) dataset from Kaggle.

## Overview

This project aims to predict employee churn using five different machine learning models: KNN, Random Forest, Naive Bayes, Logistic Regression, and a simple MLP Neural Network. The challenge is to develop models that can accurately predict which employees are most likely to leave a company, based on a range of input features such as average monthly work hours, recent promotion statuses, and salary. Our approach formulates the problem as a classification task, using the five models mentioned as predictors with various hyperparameters. We evaluated each model's performance using metrics such as accuracy, precision, recall, and F1 score, and compared their results. Our best model was able to achieve an accuracy of 99% on the test data, outperforming the other models.

## Summary of Workdone

### Data

* Data:
  * Input: CSV file of 10 features (satisfaction level, last evaluation, number of projects, average monthly hours, time spent at the company, work accidents, left, promotion in the last 5 years, department, and salary)
  * Output: Predicted binary label on whether an employee will churn or not.
  * Size: 566.79 kB; The original dataset contains 14,999 instances and 10 features.
  * Instances: The data was split into a training set containing 70% of the original data, and a testing set containing 30% of the original data.

#### Preprocessing / Clean up

These preprocessing steps help in cleaning and transforming the dataset in a format that can be used for the machine learning algorithms.
* Missing values are removed using df_raw.dropna() function.
* Boxplots are created for numerical features to identify any outliers or extreme values.
* Categorical variables are encoded (Label encoding replaces each category with a numerical value, with the same value being assigned to all instances of that category and allows the algorithm to process the data)
* The target variable/feature 'left' is separated from the rest of the dataset and stored as the target variable.
* The remaining dataset is stored once target variable is separated

#### Data Visualization

* The correlation matrix heat map can show which features are strongly correlated with the employee churn target variable.
* With the map we can see the strongest correlation, -0.39, with the "left" target feature would be the employee's "satisfaction level"

![](CorrCoef_Heatmap.png)


### Problem Formulation

  * Input: CSV data includes various features: satisfaction level, last evaluation, number of projects, average monthly hours, time spent at the company, work accidents, left, promotion in the last 5 years, department, and salary
  * Output: Predicted binary label on whether an employee will churn or not. 0 represents an employee who is still with the company, while 1 represents an employee who has left the company.
  * Models:
  
   <br />  1) K-Nearest Neighbors (KNN)
  * This model was chosen because it's a simple yet effective classification algorithm for small datasets.
  * This is a non-parametric algorithm that classifies a new observation by looking at its k-nearest neighbors in the training set. The classification is based on the most common class among the k-nearest neighbors. 
  * The hyperparameters used in the code are:
      * n_neighbors: The number of neighbors to consider. In this code, it is set to 2.
      * metric: The distance metric used to compute the distance between two observations. In this code, it is set to 'euclidean'.
     
   <br />  2) Random Forest
  * This model was chosen because it's an ensemble learning method that can handle nonlinear relationships, interactions, and high dimensional data well.
  * This is an ensemble learning algorithm that combines multiple decision trees to improve the model's accuracy and reduce overfitting. 
  * The hyperparameters used in the code are:
      * n_estimators: The number of trees in the forest. In this code, the default value is used.
      
   <br />  3) Naive Bayes
  * This model was chosen because it's a probabilistic classifier that assumes independence among the predictors, and works well with high-dimensional data.
  * This is a probabilistic algorithm that assumes the independence of each feature and predicts the class based on the joint probability of the features.
  * No hyperparameters are used for Gaussian Naive Bayes in this code.
      
   <br />  4) Logistic Regression
  * This model was chosen because it's a widely used classification algorithm that's easy to interpret and can handle binary and multi-class problems.
  * This is a parametric algorithm that models the probability of an observation belonging to a class using a logistic function.
  * The hyperparameters used in the code are:
      * max_iter: The maximum number of iterations for the solver to converge. In this code, it is set to 1000.
  
   <br />  5) Multilayer Perceptron (MLP) Neural Network
  * This model was chosen because it can capture complex nonlinear relationships and interactions among the predictors, and can handle both binary and multi-class problems.
  * This is a feedforward neural network that consists of multiple layers of nodes, each with a set of weights and biases. The layers are connected in a way that each node in one layer is connected to every node in the next layer. 
  * The hyperparameters used in the code are:
      * epochs: The number of times the training data is used to update the weights. In this code, it is set to 50.
      * batch_size: The number of samples to be used for each update. In this code, it is set to 32.
      * loss: The loss function to be optimized during training. In this code, binary cross-entropy is used.
      * optimizer: The optimizer used to update the weights during training. In this code, Adam optimizer is used.

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* Provide any references.







