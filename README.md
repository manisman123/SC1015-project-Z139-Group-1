# SC1015-project-Z139-Group-1

SC1015 is an introductory module offered by NTU giving an insight to the field of Data Science and Artificial Intelligence, and basic skills in those domains. We have been assigned to identify a suitable dataset, and from there, use data scientific techniques to answer a problem statement.

## Dataset
Our [dataset](https://www.who.int/data/gho/publications/world-health-statistics) was sourced from WHS and contains the latest edition of data (as of 2023) of life expectancy data of different countries, with other corresponding data such as age-specific mortality, presence of diseases in countries, etc.

## Problem definition
We focused on attempting to identify the years of healthy life expectancy at birth (of both sexes) of a country, based on corresponding data.

---
## Contributors
* Zhang Mingkang
* Garren Wee Qiming
* Gaius de Souza
---
# Models used 
* [Linear Regression](https://github.com/manisman123/SC1015-project-Z139-Group-1/blob/main/Regression/Linear%20Regression.ipynb)
* [Random Forest Regression](https://github.com/manisman123/SC1015-project-Z139-Group-1/blob/main/Regression/Random%20Forest%20Regression.ipynb)
* [Lasso Regression](https://github.com/manisman123/SC1015-project-Z139-Group-1/blob/main/Regression/Lasso%20Regression.ipynb)
* [Ridge Regression](https://github.com/manisman123/SC1015-project-Z139-Group-1/blob/main/Regression/Ridge%20regression.ipynb)
* [Support Vector Regression](https://github.com/manisman123/SC1015-project-Z139-Group-1/blob/main/Regression/SVR.ipynb)
* [Neural Network Regression/Multi-layer Perceptron Regression](https://github.com/manisman123/SC1015-project-Z139-Group-1/blob/main/Regression/Multi-layer%20Perceptron%20regression.ipynb)

## Cleaning the dataset
* Our dataset started off as an .xlsx file obtained from the WHS website quoted earlier. The dataset contained multiple annexes of data, which we manually processed by combining all of the data into one page.
* Then, we addressed formatting issues directly through Excel such as removing extra rows at the top of the page to make the resulting CSV file easier to work with in the form of a Pandas database.
* Missing values marked as "-" in the excel file were replaced with the mean using [SciKit Learn's SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) as a means of maintaining our quantity of data without having to drop rows from the database.

## Exploratory data analysis
Making use of a correlation heatmap, we were able to find data with the strongest correlation (<-0.7 or >0.7) with our target variable, years of healthy life expectancy at birth. The variables we found are as follows:
* UHC: Service coverage index
* Age-standardized mortality rate attributed to household and ambient air pollution (per 100 000 population)
* Proportion of population with primary reliance on clean fuels and technology (%) 
* Proportion of ever-partnered women and girls aged 15–49 years subjected to physical and/or sexual violence by a current or former intimate partner in the previous 12 months (%)
* Prevalence of anaemia in women of reproductive age (15–49 years) (%)
* Density of medical doctors (per 10 000 population)
* Maternal mortality ratio (per 100 000 live births)
* Neonatal mortality rate (per 1000 live births)
* Probability of dying from any of CVD, cancer, diabetes, CRD between age 30 and exact age 70 (%)
* Mortality rate attributed to exposure to unsafe WASH services (per 100 000 population)
* Mortality rate from unintentional poisoning (per 100 000 population)

## Comparison of accuracy between different models
From there, we used data from a combination of different X-variables with that of our target y-variable to predict healthy life expectancy. 
### Means of evaluating accuracy of model
Since we are performing multiple regression models, we are relying on two different readily available numbers that are commonly used to gauge the accuracy of a model on a test dataset - specifically the explained variance (R^2) and mean squared error (MSE). 

By comparing these scores attained by performing regression with these five different models, we will then gauge the various accuracies of these different models by presenting all of their scores as follows.

All values are given to 5 decimal places.

|Model|Training R^2|Training MSE|Test R^2|Test MSE|
|---|---|---|---|---|
|Linear Regression|0.93367|2.45887|0.94032|2.03050|
|Random Forest Regression|0.98764|2.45887|0.87346|2.03050|
|Lasso Regression|0.92760|2.45887|0.93787|2.03050|
|Ridge Regression|0.93367|2.45887|0.94035|2.03050|
|Support Vector Regression|0.76365|2.45887|0.74665|2.03050|
|Neural Network Regression|0.93367|2.45887|0.94034|2.02967|

## General comments
Comparing across different regression models, notable changes in R-squared scores between both training and test values have been observed. Possibly due to similar predictions made by the different models, the observed values for mean squared errors remain mostly unchanged throughout, with minor fluctuations observed in developing the model for neural network regression. Hence, R-squared values were referenced as our main benchmark for the performance of each model. 

## Linear regression
Linear regression functions via determining the equation of a best fit straight line through values from our chosen variables. We started off experimenting with a simple linear regression model, which factored in all of the correlated variables with respect to our target, using scikit-learn's LinearRegression function. The result was a linear model with the following equation:

In terms of R-squared score, This method resulted in a high performance of 0.94032. - second only to Ridge regression, which beat the former only marginally. It is hypothesized that, due to the high correlation nature of the data that we have chosen to make this model, this has resulted in a high degree of accuracy among the selected models.

## Random forest
Random forest is a variation of ensemble learning, which combines the predictions of multiple models to generate one with higher accuracy. Using individual decision trees as regression models results in a high volatility and low reliability, especially when attempting to translate the functionality of the model across different datasets. In the context of random forest, multiple decision trees ensure potection from individual errors that may arise. Generally for random forest regression, results from multiple models are aggregated, and a final optimal result is presented.

In our context, random forest resulted in the highest R-squared value Of 0.98764, with respect to the training dataset. However, when the model was applied to a test data set instead, this resulted in one of the lowest R-squared scores among the models that we have selected overall, of 0.87346. Given that a test R-squared score is more valid than a training R-squared score in evaluating the accuracy of our regression models, these observations compromise the perceived accuracy of our model.

## Lasso regression
In this notebook, we used the same variables as the other models but fitted it into a Lasso Regression model in hopes that the lasso regression would minimise the prediction error for our response variables by imposing a constraint on the model parameters that causes regression coefficients for some variables to shrink toward zero. However, this did not improve the results from the linear regression model as seen by the lower R^2 value.

## Ridge regression
In this notebook, we used the same variables as the other models but fitted it into a Ridge Regression model which is used to analyse any data that suffers from multicollinearity as when the issue of multicollinearity occurs, least-squares are unbiased, and variances are large, this results in predicted values being far away from the actual values. However, the results were similar to that of the linear regression model and thus we concluded that our data does not suffer from multicollinearity.

## Support vector regression
Support vector regression uses kernel functions to perform regression in higher dimensional spaces. This model allows complex relationships to be inferred from datasets, especially if the heuristic perceivability of a relationship is compromised in chaotic or noisy data. As an experiment, we decided to try SVR ourselves, even though there were perceived correlations in our dataset, to still evaluate its ability in predicting our target variable with the respective model generated.

This resulted in the lowest R-squared scores for both training and test datasets, with 0.76365 and 0.74665 respectively. It is possible that experimenting with altering parameters could increase the possible accuracy of this model. However, it could be also plausible that specific to our isolated data set, simpler regression models such as linear regression are enough to predict our target variable to an already notable extent.

## Multi-layer Perception Regression
In this notebook, Multi-layer Perceptron Regression (MLP-R) model was used with the same variable as the other models. MLP-R with default parameter was not suitable as Explained Variance R^2 of the model was less than 0 while Mean Squared Error (MSE) was extremely high compared to other models. 
Several parameters were adjusted to obtain a model that was comparable to others. These were the parameters: Activation function, solver for weight optimization, number of iterations, hidden layer and nodes in hidden layer.

Activation function was changed to no-op activation from rectified linear unit function as the later one tends to overfit the model.

Solver was changed to an optimizer in the family of quasi-Newton methods as it converged faster and performed better as our dataset was relatively small.

Number of iterations, hidden layer and nodes in hidden layer were adjusted to optimise the model. It was observed that as the numbers increases, run time on computer increases drastically. Therefore, it can be concluded that MLP-R has the potential to be the best model, however it is limited by the computational power of the computer.




