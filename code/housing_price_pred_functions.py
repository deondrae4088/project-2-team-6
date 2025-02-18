# Importing what we need 
import numpy as np 
e = np.e
import pandas as pd 
import matplotlib
import statsmodels.api as sm 
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mango.tuner import Tuner
import warnings
from math import sqrt

def melt_data(df):
    """converts dataset to long format
    """
    melted = pd.melt(df, id_vars=['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value':'mean'})

def run_auto_arima(series_i):    
    '''ARIMA (Autoregressive Integrated Moving Average) is specifically designed to model time 
    series data, meaning it analyzes patterns within a sequence of data points ordered by time, 
    allowing prediction of future values based on past trends. 
    Runs a grid search on the series passed in, then instantiates and fits 
    an ARIMA model with those hyperparameters, then returns that fit model. 
    Parameters:
    series_i: a pandas series of time series data
    Returns:
    model: a fitted ARIMA model   
    '''
    
    gridsearch = auto_arima(series_i,
                            start_p = 0,
                            max_p = 1,
                            d = 0, 
                            max_d = 0, 
                            start_q = 0,
                            max_q = 2,
                            seasonal=True,
                            m = 12,
                            suppress_warnings=True)
    
    model = ARIMA(series_i, 
                  order = gridsearch.order, 
                  seasonal_order = gridsearch.seasonal_order,
                  enforce_stationarity=False)      
    
    return model.fit()

def run_arima_model(i, steps, df):
       
    '''This function takes i, representing the index of one of our time series,
    steps, which is the number of periods after the end of the 
    sample you want to make a prediction for, and df, the dataframe the series
    is stored in. It log transforms the series, runs run_auto_arima, gets the 
    forecast from the fit model, and inverse log transforms that forecast series
    back into the original units.'''
    
    series = df.iloc[:, i:i+1]
    
    name = series.columns[0]
    
    log_series = log_transform(series)
    
    model = run_auto_arima(log_series)   
     
    log_forecast = model.get_forecast(steps)
    forecast_series = e ** log_forecast.summary_frame()
    
    return name, series, forecast_series

def log_transform(series_i):
    
    '''Takes in a series and returns the log transformed version of that series'''
    
    log_transformed = np.log(series_i)
    dropped_nans = log_transformed.dropna()
    return dropped_nans

def evaluate_models(df1, df2):
    
    '''This function takes in two dataframes (train and test in our case), 
    and returns a dataframe with how accurate the models fit to the train 
    set were in predicting the test set values.'''

    names = []
    actuals = []
    preds = []
    perc_errors = []
    
    for i in range(len(df1.columns)):
        
        name, series, forecast_series = run_arima_model(i, 24, df1)
        
        clean_name = name[:-4]
        
        actual_val = df2[name][-1]
        predicted_val = forecast_series.iloc[23, 0]
        error = abs(actual_val - predicted_val)
        percent_error = (error/ actual_val) * 100
        
        names.append(clean_name)
        actuals.append(f'{round(actual_val):,}')
        preds.append(f'{round(predicted_val):,}')
        perc_errors.append(round(percent_error, 2))
        
        #print(train.columns[i][:-4], 'done', f'{i+1}/26')
        
    
    results_df = pd.DataFrame(index=names)
    results_df['2024 Actual'] = actuals 
    results_df['2024 Predicted'] = preds
    results_df['% Error'] = perc_errors
    results_df.sort_values(by='% Error', inplace=True)
    
    return results_df

def generate_predictions(df, steps):
    
    '''Similar to evaluate_models(), this function takes in a dataframe,
    and a specific number of steps, and returns a dataframe of the 
    future predictions the specified number of steps past the end of 
    the sample.'''
    
    names = []
    current_vals = []
    pred_vals = []
    net_profits = []
    ROI_strings = []
    
    count = 0
    for i in range(len(df.columns)):
        
        count += 1
        
        name, series, forecast = run_arima_model(i, steps, df)
        
        clean_name = name[:-4]
        print(clean_name)
        
        cur_val = series.iloc[-1, 0]
        pred_val = forecast.iloc[steps-1, 0]
        net_prof = round(pred_val - cur_val , 2)
        roi = int(round(((pred_val - cur_val) / cur_val) * 100, 2))
        
        names.append(clean_name)
        current_vals.append(f'{round(cur_val):,}')
        pred_vals.append(f'{round(pred_val):,}')
        net_profits.append(f'{round(net_prof):,}')
        ROI_strings.append(f'{roi}%') 
        
        if count == 26:
            break
    
    
    results_df = pd.DataFrame()
    results_df['City'] = names
    results_df.set_index(['City'])
    results_df['Current Value'] = current_vals
    results_df['Predicted Value'] = pred_vals
    results_df['Net Profit'] = net_profits
    results_df['ROI'] = ROI_strings
    
    return results_df

def plot_results(i, steps, df):    
    '''plot_results runs run_arima_model() and plots the results.'''
    
    name, original_series, forecast_series = run_arima_model(i, steps, df)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(original_series)
    plt.plot(forecast_series['mean'])
    ax.fill_between(forecast_series.index, forecast_series['mean_ci_lower'], 
                    forecast_series['mean_ci_upper'], color='k', alpha=0.1)
    plt.title(name)
    plt.legend(['Original','Predicted'], loc='lower right')
    plt.xlabel('Year')
    plt.ylabel('Median Home Price')
    plt.show()   

    forecast = round(forecast_series['mean'][11])
    low_int =  round(forecast_series['mean_ci_lower'][11])
    high_int = round(forecast_series['mean_ci_upper'][11])
    
    print(f'12 month forecast: {forecast}')
    print(f'95% confidence that the true future value is between {low_int}, and {high_int}')   

def check_stationarity(timeseries):
    # Perform the Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    p_value = result[1]
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {p_value}')
    print('Stationary' if p_value < 0.05 else 'Non-Stationary')

    # application -  check_stationarity(df_2018['Los Angeles, CA'])
    # result - 
    # ADF Statistic: -0.6767289537979833
    # p-value: 0.8526855399082687
    # Non-Stationary

def stationarity_check(TS, column, plot_std=True):
    '''Outputs a plot of the Rolling Mean and Standard Deviation and prints results of the Dickey-Fuller Test
      TS: Time Series, this is the dataframe from which you are pulling your information
      column: This is the column within the TS that you are interested in
      plot_std: optional to plot the standard deviation or not'''
    
    # Calculate rolling statistics
    rolmean = TS[column].rolling(window = 8, center = False).mean()
    rolstd = TS[column].rolling(window = 8, center = False).std()
    
    # Perform the Dickey Fuller Test
    dftest = adfuller(TS[column].dropna())
    
    # Formatting
    fig = plt.figure(figsize=(14,8))
    
    #Plot rolling statistics:
    orig = plt.plot(TS[column], color='blue',label='Original Home Price')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        # Optional plot of standard deviation
    if plot_std:
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.title('Rolling Mean & Standard Deviation for {}'.format(column)) # alternative title
    else:
        plt.title('Rolling Mean for {}'.format(column)) # alternative title
    
    # Legend and show
    plt.legend(loc='best')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results
    print ('Results of Dickey-Fuller Test:')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value {}'.format(key)] = value
    print (dfoutput)    

def run_auto_sarima(series_i):   
    """SARIMA model parameters based off manual tuning in analysing model.summary fits the SARIMA model."""
    model = SARIMAX(
        series_i,
        order=(3,1,1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit()    

def run_sarima_model(i, steps, df):
    """Runs SARIMA on the selected time series, returning forecast results."""
    series = df.iloc[:, i:i+1]
    name = series.columns[0]
    log_series = np.log(series).dropna()
    
    model = run_auto_sarima(log_series)
    log_forecast = model.get_forecast(steps)
    forecast_series = np.exp(log_forecast.summary_frame())
    
    return name, series, forecast_series

def plot_sarima_results(i, steps, df):
    name, original_series, forecast_series = run_sarima_model(i, steps, df)
    
    plt.figure(figsize=(15, 7))
    plt.plot(original_series, label='Original')
    plt.plot(forecast_series['mean'], label='Predicted')
    plt.fill_between(
        forecast_series.index, 
        forecast_series['mean_ci_lower'], 
        forecast_series['mean_ci_upper'], 
        color='gray', alpha=0.2
    )
    plt.title(name)
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Median Home Price')
    plt.show()
    
    forecast = round(forecast_series['mean'][steps - 1])
    low_int = round(forecast_series['mean_ci_lower'][steps - 1])
    high_int = round(forecast_series['mean_ci_upper'][steps - 1])
    print(f"{steps}-month forecast: {forecast}")
    print(f"95% confidence interval: {low_int} - {high_int}")

def evaluate_sarima_models(df1, df2):
    
    '''This function takes in two dataframes (train and test in our case), 
    and returns a dataframe with how accurate the models fit to the train 
    set were in predicting the test set values.'''

    names = []
    actuals = []
    preds = []
    perc_errors = []
    
    for i in range(len(train.columns)):
        
        name, series, forecast_series = run_sarima_model(i, 25, df1)
        
        clean_name = name[:-4]
        
        actual_val = df2[name][-1]
        predicted_val = forecast_series.iloc[23, 0]
        error = abs(actual_val - predicted_val)
        percent_error = (error/ actual_val) * 100
        
        names.append(clean_name)
        actuals.append(f'{round(actual_val):,}')
        preds.append(f'{round(predicted_val):,}')
        perc_errors.append(round(percent_error, 2))
        
        #print(train.columns[i][:-4], 'done', f'{i+1}/26')
        
    
    results_df = pd.DataFrame(index=names)
    results_df['2024 Actual'] = actuals 
    results_df['2024 Predicted'] = preds
    results_df['% Error'] = perc_errors
    results_df.sort_values(by='% Error', inplace=True)
    
    return results_df

def generate_sarima_predictions(df, steps):
    
    '''Similar to evaluate_models(), this function takes in a dataframe,
    and a specific number of steps, and returns a dataframe of the 
    future predictions the specified number of steps past the end of 
    the sample.'''
    
    names = []
    current_vals = []
    pred_vals = []
    net_profits = []
    ROI_strings = []
    
    count = 0
    for i in range(len(df.columns)):
        
        count += 1
        
        name, series, forecast = run_sarima_model(i, steps, df)
        
        clean_name = name[:-4]
        print(clean_name)
        
        cur_val = series.iloc[-1, 0]
        pred_val = forecast.iloc[steps-1, 0]
        net_prof = round(pred_val - cur_val , 2)
        roi = int(round(((pred_val - cur_val) / cur_val) * 100, 2))
        
        names.append(clean_name)
        current_vals.append(f'{round(cur_val):,}')
        pred_vals.append(f'{round(pred_val):,}')
        net_profits.append(f'{round(net_prof):,}')
        ROI_strings.append(f'{roi}%') 
        
        if count == 26:
            break
    
    
    results_df = pd.DataFrame()
    results_df['City'] = names
    results_df.set_index(['City'])
    results_df['Current Value'] = current_vals
    results_df['Predicted Value'] = pred_vals
    results_df['Net Profit'] = net_profits
    results_df['ROI'] = ROI_strings
    
    return results_df

def plot_sarima_results(i, steps, df):
    
    '''plot_results runs run_arima_model() and plots the results.'''
    
    name, original_series, forecast_series = run_sarima_model(i, steps, df)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(original_series)
    plt.plot(forecast_series['mean'])
    ax.fill_between(forecast_series.index, forecast_series['mean_ci_lower'], 
                    forecast_series['mean_ci_upper'], color='k', alpha=0.1)
    plt.title(name)
    plt.legend(['Original','Predicted'], loc='lower right')
    plt.xlabel('Year')
    plt.ylabel('Median Home Price')
    plt.show()
    
    forecast = round(forecast_series['mean'][11])
    low_int =  round(forecast_series['mean_ci_lower'][11])
    high_int = round(forecast_series['mean_ci_upper'][11])
    
    print(f'12 month forecast: {forecast}')
    print(f'95% confidence that the true future value is between {low_int}, and {high_int}')

def arima_objective_function(args_list):

    global data_values
    
    params_evaluated = []
    results = []
    
    for params in args_list:
        try:
            p,d,q = params['p'],params['d'], params['q']
            trend = params['trend']
            
            model = ARIMA(data_values, order=(p,d,q), trend = trend)
            predictions = model.fit()

            mse = mean_squared_error(data_values, predictions.fittedvalues)   
            params_evaluated.append(params)
            results.append(mse)
        except:
            #print(f"Exception raised for {params}")
            #pass 
            params_evaluated.append(params)
            results.append(1e5)
        
        #print(params_evaluated, mse)
    return params_evaluated, results

def evaluate_arima_model(X, arima_order):
    # evaluate an ARIMA model for a given order (p,d,q)
    # prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse
 # evaluate combinations of p, d and q values for an ARIMA model

def evaluate_for_perf_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

def initialize_data(csvpath):
    # Load the data
    df = pd.read_csv(csvpath)
    return df

def detrend_test(TS, alpha=0.05, maxlag=4, suppress_output=False):

    '''Selecting the best method for detrending timeseries based on lowest p-value of the augmented Dickey-Fuller.
       TS: timeseries dataframe
       alpha: alpha value for Dickey-Fuller '''
    
    new_TS = pd.DataFrame()
    
    plist = []  
    plist_zips = []
    
    # Keep track of which metro require a log 1st difference transformation
    log_1diff = []
    
    for column in list(TS.columns):  #go through each metro in the DF
        p_values = []
        
        # First Difference
            # find the first difference for each row in the metro
        data_1diff = TS[column].diff(periods=1) 
            # perform Dickey Fuller test on first difference
        dftest = adfuller(data_1diff.dropna(),maxlag=maxlag)
            # Place first 4 outputs of the Dickey Fuller test in a dataframe and label the outputs appropriately in the index
        dfoutput_1diff = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput_1diff['Critical Values {}'.format(key)] = value
            # Add p-value of first difference Dickey-Fuller p-value list
        p_values.append(dfoutput_1diff[1])
    
        # Log First Difference
            # find the log first difference for each row in the metro
        data_log_1diff = TS[column].apply(lambda x: np.log(x)) - TS[column].apply(lambda x: np.log(x)).shift(1)
            # perform Dickey Fuller test on first difference
        dftest = adfuller(data_log_1diff.dropna(),maxlag=maxlag)
            # Take first 4 outputs of the Dickey Fuller test and label the outputs appropriately
        dfoutput_log_1diff = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput_log_1diff['Critical Values {}'.format(key)] = value
            # Add p-value of log first difference Dickey-Fuller p-value list
        p_values.append(dfoutput_log_1diff[1])

        # If first difference performed better, print the Dickey-Fuller results and plot
        """ if np.argmin(p_values)==0:
            data_1diff.plot(figsize=(20,6))
            plt.title('{} First Difference'.format(column))
            plt.show();
            print(dfoutput_1diff)
            new_TS[column]=data_1diff """
        
        # If log first difference performed better, print the Dickey-Fuller results and plot
        """ elif np.argmin(p_values)==1:
            log_1diff.append(column)
            data_log_1diff.plot(figsize=(20,6))
            plt.title('{} Log First Difference'.format(column))
            plt.show();
            print(dfoutput_log_1diff)
            new_TS[column]=data_log_1diff """
        
        # Add the smallest p value from tests, to the plist
        plist.append(min(p_values))
        # Add metros with high p-values to plist_zips
        if min(p_values)>alpha:
            plist_zips.append(column)
    
    
    if suppress_output==False:
        print('\nNumber of p-values above alpha of {}:'.format(alpha),(np.array(plist)>alpha).sum())
        print('\nMetro Areas with p-values above alpha of {}'.format(alpha), plist_zips)        
        return new_TS, log_1diff
    
def append_to_list_and_create_csv(data, filename):
    # Initialize an empty list
    data_list = []

    # Append data to the list through a loop
    for item in data:
        data_list.append(item)

    # Create a CSV file from the list
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Data'])  # Write header
        for item in data_list:
            writer.writerow([item])

# Example usage
data = ['apple', 'banana', 'cherry', 'date', 'elderberry']
filename = 'fruits.csv'
append_to_list_and_create_csv(data, filename)

print(f"CSV file '{filename}' created successfully.")