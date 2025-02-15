# Golden State Housing Report: A predictive analysis of California's housing market

## **Project Overview**

### *Package Requirements*

`pip install x` ; where 'x' is the package listed below:
* datetime 
* pandas
* hvplot
* matplotlib
* numpy
* cpi
* sklearn.linear_model
* sklearn.preprocessing
* seaborn
* scipy.stats
* warnings

### *File Navigation, Installation, Usage, Demo & Slideshow*

**File Navigation**
* Code: Code - Directory containing all of the the code
  * Housing Price Prediction using Zillow Data 
    * Navigate to [Link to Housing Price Prediction using Zillow Data](code/cg_arima.ipynb) 
  * Predicted Interest Rates 
    * Navigate to [Link to Predicted Interest Rates](code/dx_pred_int_rates.ipynb)      
* Content: 
  * Navigate to [Resources/content](resources/content) - Directory containing all images of plots created in Jupyter Notebook and demos.
* Data: 
  * Navigate to [Resources/data](resources/data) - Directory containing all of the csv files used by the code
      
**Installation**
  * Clone the repository from here.. [Clone Me](https://github.com/xraySMULu/golden-state-housing-report) 

**Usage**
  * Open Jupyter notebook, click 'Run all' to execute all code blocks.

**Demo**
* Housing Price Prediction 
  * Navigate to [Demo](resources/content/cg_demo.gif)   
* Interest Rate Prediction 
  * Navigate to [Demo](resources/content/dex_demo.gif)   

**Slideshow**
* Project #2 - Team #6 - Slideshow 
  * Navigate to [Slideshow PDF](Resources/content/?.pdf)   

### *Purpose of Use*   
* The primary goal of our project, Golden State Housing Insights, is to predict housing prices in the state of California. Our team aims to achieve this by leveraging machine learning models to analyze various factors, including investor return on investment (ROI), affordability, distance between homes and cities, and specific home features. By integrating these elements, we strive to provide accurate and actionable insights into the California housing market, aiding investors, homebuyers, and policymakers in making informed decisions.

## Data Pre-Processing & Machine Learning
* Housing Price Prediction using Zillow Data Analysis
  * Our team employed a comprehensive data pre-processing approach to ensure the accuracy and reliability of our housing price predictions. We utilized powerful libraries such as NumPy and Pandas for efficient data manipulation and analysis. Matplotlib was used for visualizing data trends and patterns. We also applied data melting techniques to reshape our datasets, making them more suitable for analysis. Additionally, we incorporated time series analysis to account for temporal trends and seasonality in housing prices. This robust pre-processing framework enabled us to prepare our data effectively for machine learning modeling. Our team conducted an extensive exploratory data analysis (EDA) to uncover underlying patterns and relationships within the housing data. This initial step allowed us to gain valuable insights and informed our subsequent modeling approach. We employed both ARIMA (AutoRegressive Integrated Moving Average) and Auto-ARIMA models to predict housing prices. The ARIMA model helped us understand the temporal dependencies and trends in the data, while the Auto-ARIMA model automated the process of identifying the optimal parameters for our time series forecasting. By comparing the results from these models, we were able to enhance the accuracy and robustness of our housing price predictions.
* Housing Feature Analysis
  * To prepare the dataset for analysis, we first removed outliers by filtering extreme values in the features AveRooms, AveBedrms, Population, and AveOccup. Next, we performed feature selection, choosing key variables such as MedInc, HouseAge, and Population. The data was then split into training and testing sets, with 80% allocated for training and 20% for testing using the train_test_split() function. Finally, we standardized the values using StandardScaler() to ensure consistent scaling across features. We employed both Linear Regression and Random Forest models for our analysis. The models were trained using the model.fit(X_train, y_train) method, which allowed them to learn from the training data. Once trained, we made predictions on the test data using the model.predict(X_test) function. This approach enabled us to evaluate the performance and accuracy of each model.
  * To predict housing market trends and analyze influential home features, we utilized the Ames dataset, assuming Californians have similar preferences. We began by preprocessing the dataset, removing features with sparse data and encoding non-numerical features. A correlation analysis was then conducted to identify the 12-15 most influential home features based on their correlation with home prices. To delve deeper into feature contributions, we trained models using Linear Regression and Random Forest, employing recursive feature elimination to iteratively remove less important features. Model performance was evaluated using RMSE, R², and MAE metrics, providing a comprehensive understanding of the factors influencing home prices.
* Interest Rate Prediction Analysis
 * To predict future interest rates up to February 2025, the code utilizes historical data from a CSV file ('fed-rates.csv'). The data spans from January 2017 to December 2024, and is cleaned to calculate average monthly rates. The machine learning techniques employed involve using the trained models to predict interest rates for the months leading up to February 2025. These predictions are then integrated with the existing data for visualization purposes. The visualization includes three key elements: the actual historical interest rates, represented as a line with circle markers; the Linear Regression predictions, depicted as a dashed line with 'x' markers, which illustrate the general trend but may miss some finer details; and the K-Nearest Neighbors (KNN) predictions, shown as a dotted line with square markers, which are potentially more responsive to recent changes in interest rates.

## Visuals and Explanations
* We used various visualizations to depict our data, including line graphs for trends, bar charts for categorical comparisons, scatter plots for correlations, and heatmaps for data density and relationships. These visualizations were presented in Google Slides for easy sharing and in Streamlit for an interactive experience, effectively communicating our findings.

**Housing Price ROI**
![image](resources/content/cg_roi.png)
The ROI for California top metro areas.

**Housing Price Prediction**
![image](resources/content/cg_prediction.png)
The home value predictions for California top metro areas.

**Housing Price Prediction Top Metro**
![image](resources/content/cg_mtop.png)
The home value predictions for California top metro.

**Housing Feature Analysis chart 1**
![image](resources/content/dex_lr1.png)
Feature analysis 1.

**Housing Feature Analysis chart 2**
![image](resources/content/dex_lr1.png)
Feature analysis 2.

**Interest Rate Prediction Analysis chart 1**
![image](resources/content/dex_lr1.png)
linear Regression 1.

**Interest Rate Prediction Analysis chart 2**
![image](resources/content/dex_lr2.png)
linear Regression 2.

## Additional Explanations and Major Findings

**Major findings**
* Housing Price Prediction using Zillow Data Analysis
* Housing Feature Analysis  
* Interest Rate Prediction Analysis
  * The plot illustrates how the models predict interest rates will change, indicating whether they expect rates to rise, fall, or remain stable. By comparing these predictions to historical rates, we can evaluate the models' performance. The code concludes by displaying future prediction values in a table (future_predictions_df), enabling users to assess and make informed decisions based on these predictions.

## Additional questions that surfaced and plan for future development


## Conclusion


## References

* https://www.eia.gov/ - U.S. Energy Information Administration
* https://fred.stlouisfed.org/ - Federal Reserve Economic Data
* https://www.bls.gov/ - U.S. Bureau of Labor Statistics
* https://chatgpt.com
* https://finance.yahoo.com/ - SP500
* https://fred.stlouisfed.org/series/CSUSHPISA - Housing
* https://fred.stlouisfed.org/seriesBeta/FPCPITOTLZGUSA - CPI yearly
* https://fred.stlouisfed.org/seriesBeta/CPIAUCSL - CPI monthly
* Coordinates of 50 states: https://gist.github.com/dikaio/0ce2a7e9f7088918f8c6ff24436fd035
