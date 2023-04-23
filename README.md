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
|Random Forest Regression|0.98764|0.47668|0.86753|4.50751|
|Lasso Regression|0.92760|2.68364|0.93787|2.11410|
|Ridge Regression|0.93367|2.45890|0.94035|2.02971|
|Support Vector Regression|0.76365|8.76129|0.74665|8.62036|
|Neural Network Regression|0.93367|2.45887|0.94032|2.03070|

## General comments
Comparing across different regression models, notable changes in both R-squared and mean squared error values were observed with respect to different choices of different regression models. Thus, we primarily evaluated each model based on these statistics. R-squared values range from a score of 0 to 1 in terms of measuring accuracy, and mean square error values depend on magnitude with respect to the data set that we are utilizing for this test.

## Linear regression
Linear regression functions via determining the equation of a best fit straight line through values from our chosen variables. We started off experimenting with a simple linear regression model, which factored in all of the correlated variables with respect to our target, using scikit-learn's LinearRegression function. 

Starting off with this model led us to a good baseline for which we could compare other models against its perfromance. Linear regression was successful in which it achieved a very good fit, with a low mean squared error. 

## Random forest
Random forest is a variation of ensemble learning, which combines the predictions of multiple models to generate one with higher accuracy. Using individual decision trees as regression models results in a high volatility and low reliability, especially when attempting to translate the functionality of the model across different datasets. In the context of random forest, multiple decision trees ensure potection from individual errors that may arise. Generally for random forest regression, results from multiple models are aggregated, and a final optimal result is presented.

In our context, our model is able to generate relatively acceptable results without hyper-parameter tuning. However, it did suffer from overfitting due to not enough trees that we provided in the model, as noted by a higher R-squared value achieved in training data compared to test data, as well as a dramatic increase in MSE under similar comparisons. For reasons with respect to efficiency, it was not practical to generate more trees, but instead, to look to other solutions in other algorithms that we had lined up as competitors to this choice.

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

# Conclusion
In the end, we came to a consensus that our most preferred model among the several was that of linear regression, as it resulted in a reasonable degree of accuracy even with a low computational time. The implementation of other advanced models proved to be unnecessary in a simple regression model, such as our case study. For instance, correlation between our variables was not nebulous enough to warrant the use of support vector regression, or non-linear to warrant the use of neural network regression systems such as multi-layer perceptron regression. Eventually, we decided that our initial baseline for using a regression model to make simple predictions was already accurate to answer our questions, in our particular extent.

There were other interesting observations that we have made with regards to what we have learned from correlated variables with healthy life expectancy - such as the correlations between histories of sexual violence, or the dependency on clean energy with that of healthy life expectancy. We hypothesize that the ability of our model to make predictions on healthy life expectancy, with respect to its relationship with these variables, would help to make stronger insights on how policies affecting these variables will benefit the quality of life within a country, from this particular perspective.

In addition to these insights, we also learned how to use different models from outside of syllabus. These include all models besides linear regression, such as ridge, lasso and support vector regression. Easy implementations of these models were possible with the simplicity of scikit-learn's library, which also allowed us to utilize special data-cleaning functions such as SimpleInputer for the first time.

