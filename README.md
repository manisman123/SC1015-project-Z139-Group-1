# SC1015-project-Z139-Group-1

SC1015 is an introductory module offered by NTU giving an insight to the field of Data Science and Artificial Intelligence, and basic skills in those domains. We have been assigned to identify a suitable dataset, and from there, use data scientific techniques to answer a problem statement.

## Dataset
Our [dataset]() was sourced from WHS and contains the latest edition of data (as of 2023) of life expectancy data of different countries, with other corresponding data such as age-specific mortality, presence of diseases in countries, etc.

## Problem definition
We focused on attempting to identify the years of healthy life expectancy at birth (of both sexes) of a country, based on corresponding data.

---
## Contributors
* Zhang Mingkang (@...)
* Garren Wee Qiming (@...)
* Gaius de Souza (@...)
---
# Models used 
* [Linear Regression]()
* [Random Forest Regression]()
* [Lasso Regression]()
* [Ridge Regression]()
* [Support Vector Regression]()
* [Neural Network Regression]()

## Cleaning the dataset
* Our dataset started off as an .xlsx file obtained from the WHS website quoted earlier. The dataset contained multiple annexes of data, which we manually processed by combining all of the data into one page.
* Then, we addressed formatting issues directly through Excel such as removing extra rows at the top of the page to make the resulting CSV file easier to work with in the form of a Pandas database.
* Missing values marked as "-" in the excel file were replaced with the mean using [SciKit Learn's SimpleImputer]() as a means of maintaining our quantity of data without having to drop rows from the database.

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
|Neural Network Regression|
