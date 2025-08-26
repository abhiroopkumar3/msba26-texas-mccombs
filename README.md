### Report: Prediction Contest (Group 2)

***

**Authors:** Samantha Feinberg, Alina Hota, Abhiroop Kumar, Connor Therrien, and Janani Vakkanti  

***

### Table of Contents

* [Objective](#objective)
* [Data Preparation and Feature Engineering](#data-preparation-and-feature-engineering)
    1.  [X/Y Graphs and Correlation Analysis](#xy-graphs-and-correlation-analysis)
    2.  [Data Preparation](#data-preparation)
    3.  [Feature Engineering](#feature-engineering)
* [Models](#models)
* [Final Prediction & Conclusions](#final-prediction--conclusions)
* [Table: Comparison of Original and Predicted Prices](#table-comparison-of-original-and-predicted-prices)

***

### Objective

The objective was to predict Austin house prices using the 'latestPrice' variable from the 'austinhouses_holdout.csv' dataset and the 'austinhouses.csv' dataset. The analysis compared different modeling approaches on expanded predictors.  

***

### Data Preparation and Feature Engineering

1.  #### **X/Y Graphs and Correlation Analysis:**
    The Austin housing dataset was analyzed using X/Y graphs and a correlation matrix to understand the relationship between different features and house prices. These tools helped identify key features for predicting prices.  
2.  #### **Data Preparation:**
    This process involved excluding unique identifiers and selecting specific predictors like `streetAddress`, `homeType`, and `avgSchoolSize`. Missing values were handled by omitting "NA" values for categorical data, and some features were converted to factors for specific modeling techniques.  
3.  #### **Feature Engineering:**
    Custom columns were created from existing features to improve model performance, provide more accurate price predictions, and achieve a lower RMSE. These custom columns included:  
    * `age when sold` to capture a property's lifecycle  
    * `area_ratio` to measure house density on its lot  
    * `total rooms` as a metric for size  
    * `school_quality_score` to represent a combination of school quality and proximity  
    * `log_latestPrice` as the logarithm of the `latestPrice` variable  

***

### Models

Nine models were run on the target variable, `log_latestPrice`. The results were as follows:

* **Linear Regression:** Served as a baseline model with an RMSE of 0.26.
* **Ridge and Lasso:** Both used regularization to prevent overfitting and had an RMSE of 0.31.
* **Stepwise:** Achieved an RMSE of 0.31.
* **Bagging:** Performed better with an RMSE of 0.25.
* **Random Forest:** Had an RMSE of 0.28.
* **Boosting:** Performed better with an RMSE of 0.25.
* **Unpruned Regression Tree:** Performed the best, with the lowest RMSE of 0.18, outperforming its pruned counterpart which had an RMSE of 0.24.

***

### Final Prediction & Conclusions

The **unpruned regression tree** was chosen as the final model due to its lowest RMSE of 0.18. The model's predictions on the holdout dataset demonstrated strong performance, with the predicted prices showing similar skewness to the original prices, which indicates robust performance.

***

### **Table: Comparison of Original and Predicted Prices**

| | **Original Price** | **Predicted Price** |
| :--- | :--- | :--- |
| **Min.** | 5.8 | 92.59 |
| **1st Qu.** | 310 | 314.25 |
| **Median** | 400 | 399.28 |
| **Mean** | 486.6 | 479.53 |
| **3rd Qu.** | 550 | 554.43 |
| **Max.** | 6250 | 3558.04 |