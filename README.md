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
* Content: 
  * Navigate to [Resources/content](Resources/content) - Directory containing all images of plots created in Jupyter Notebook and demos.
* Data: 
  * Navigate to [Resources/data](Resources/data) - Directory containing all of the csv files used by the code
  
**Installation**
  * Clone the repository from here.. [Clone Me](https://github.com/xraySMULu/golden-state-housing-report) 

**Usage**
  * Open Jupyter notebook, click 'Run all' to execute all code blocks.

**Demo**
* Housing Price Prediction using Zillow Data 
  * Navigate to [Demo](Resources/content/cg_demo.gif)    

**Slideshow**
* Project #2 - Team #6 - Slideshow 
  * Navigate to [Slideshow PDF](Resources/content/?.pdf)   

### *Purpose of Use*   
* The primary goal of our project, Golden State Housing Insights, is to predict housing prices in the state of California. Our team aims to achieve this by leveraging machine learning models to analyze various factors, including investor return on investment (ROI), affordability, distance between homes and cities, and specific home features. By integrating these elements, we strive to provide accurate and actionable insights into the California housing market, aiding investors, homebuyers, and policymakers in making informed decisions.

## Data Pre-Processing and Machine Learning Steps
* Housing Price Prediction using Zillow Data Analysis
  * Our team employed a comprehensive data pre-processing approach to ensure the accuracy and reliability of our housing price predictions. We utilized powerful libraries such as NumPy and Pandas for efficient data manipulation and analysis. Matplotlib was used for visualizing data trends and patterns. We also applied data melting techniques to reshape our datasets, making them more suitable for analysis. Additionally, we incorporated time series analysis to account for temporal trends and seasonality in housing prices. This robust pre-processing framework enabled us to prepare our data effectively for machine learning modeling.

  * Our team conducted an extensive exploratory data analysis (EDA) to uncover underlying patterns and relationships within the housing data. This initial step allowed us to gain valuable insights and informed our subsequent modeling approach. We employed both ARIMA (AutoRegressive Integrated Moving Average) and Auto-ARIMA models to predict housing prices. The ARIMA model helped us understand the temporal dependencies and trends in the data, while the Auto-ARIMA model automated the process of identifying the optimal parameters for our time series forecasting. By comparing the results from these models, we were able to enhance the accuracy and robustness of our housing price predictions.
 
 * Feature Engineering Analysis
  * To prepare the dataset for analysis, we first removed outliers by filtering extreme values in the features AveRooms, AveBedrms, Population, and AveOccup. Next, we performed feature selection, choosing key variables such as MedInc, HouseAge, and Population. The data was then split into training and testing sets, with 80% allocated for training and 20% for testing using the train_test_split() function. Finally, we standardized the values using StandardScaler() to ensure consistent scaling across features. 

  * We employed both Linear Regression and Random Forest models for our analysis. The models were trained using the model.fit(X_train, y_train) method, which allowed them to learn from the training data. Once trained, we made predictions on the test data using the model.predict(X_test) function. This approach enabled us to evaluate the performance and accuracy of each model.
  
  * To predict housing market trends and analyze influential home features, we utilized the Ames dataset, assuming Californians have similar preferences. We began by preprocessing the dataset, removing features with sparse data and encoding non-numerical features. A correlation analysis was then conducted to identify the 12-15 most influential home features based on their correlation with home prices. 
  
  * To delve deeper into feature contributions, we trained models using Linear Regression and Random Forest, employing recursive feature elimination to iteratively remove less important features. Model performance was evaluated using RMSE, R², and MAE metrics, providing a comprehensive understanding of the factors influencing home prices.
  


## Visuals and Explanations
* To effectively depict our data, we utilized a variety of visualizations, including line graphs, bar charts, scatter plots, and heatmaps. Line graphs were employed to illustrate trends over time, while bar charts provided a clear comparison of categorical data. Scatter plots helped us identify correlations between different variables, and heatmaps offered a comprehensive view of data density and relationships. We finalized our data display by presenting these visualizations in Google Slides for easy sharing and collaboration, and in Streamlit for an interactive and dynamic user experience. These visual tools enabled us to communicate our findings clearly and effectively to our audience.

**Chart???**
![image](Resources/content/?.png)
The description.


## Additional Explanations and Major Findings

**Major findings**
* Housing Price Prediction using Zillow Data Analysis
  * As part of our effort to enhance the predictive capabilities of our time series model, we explored the use of Seasonal Autoregressive Integrated Moving Average (SARIMA) to account for seasonality in our dataset. This research involved evaluating the limitations of our existing ARIMA model, which assumes stationarity without explicit seasonal adjustments. By implementing SARIMA, we aimed to better capture recurring patterns in the data, leveraging seasonal differencing and additional parameters to refine our forecasts. We experimented with automated parameter tuning using auto_arima and conducted multiple iterations to optimize the seasonal components (P, D, Q, m). However, this required extensive computational time and deeper fine-tuning, making it challenging to integrate effectively within our project’s timeline.
  * Given these constraints, we made the strategic decision to revert to ARIMA, ensuring timely completion while maintaining a robust forecasting approach. Despite stepping back from SARIMA, the research process provided valuable insights into time series modeling, specifically the trade-offs between model complexity and practicality in real-world applications. The exploration of SARIMA deepened our understanding of seasonal trends and the challenges of working with large datasets, equipping us with knowledge for potential future refinements. While ARIMA will be our final model, the SARIMA research effort has strengthened our ability to make data-driven decisions and assess the best methodologies based on project requirements and constraints.
* ??
  

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
