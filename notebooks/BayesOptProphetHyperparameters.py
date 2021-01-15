#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from bayes_opt import BayesianOptimization

# silence scipy/optimize/_numdiff.py:519: RuntimeWarnings
import warnings; 
warnings.simplefilter("ignore", RuntimeWarning)

# silence prophet INFO messages
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)

import random
random.seed(42)
np.random.seed(42)


# # Bayesian Optimisation of Prophet Hyperparameters
# 
# This notebook illustrates optimisation of continuous hyperparamaters of 
# [prophet](https://facebook.github.io/prophet/) 
# time series models using the 
# [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) package.
# 
# The notebook is organised into the following sections:
#  * Importing Data
#  * Building Simple Model
#  * Cross-validating Model
#  * Tuning Discrete Prophet Hyperparameters
#  * Bayesian Optimisation of Continuous Prophet Hyperparameters
#  * Conclusion
# 
# 
# ## Import Data
# 
# Data has been cleaned but may still have issues.  See the [cleaning section](https://github.com/makeyourownmaker/CambridgeTemperatureModel#Cleaning) in my [Cambridge Temperature Model repository](https://github.com/makeyourownmaker/CambridgeTemperatureModel) for details.
# 
# The `y` variable is temperature * 10.  I'm primarily interested in very short term forecasts (less than 2 hours)
# but forecasts over 24 hours are also interesting.

# In[3]:


df = pd.read_csv("../data/CamUKWeather.csv", parse_dates=True)
print("Shape:")
print(df.shape)
print("\nInfo:")
print(df.info())
print("\nSummary stats:")
display(df.describe())
print("\nRaw data:")
df


# Create train and test data.  Will perform [rolling origin forecasting](https://otexts.com/fpp2/accuracy.html#time-series-cross-validation) later.

# In[4]:


THRESHOLD = 2019
df_test  = df[df['year'] >= THRESHOLD]
df_train = df[df['year'] <  THRESHOLD]


# ---
# 
# 
# ## Build Simple Model
# 
# First, build a simple model with flat growth and no weekly seasonality.  This is a quick sanity check.  Results should be similar to my previous R version.
# 
# One reason for using the python prophet version over R is to check the 
# [flat growth](https://facebook.github.io/prophet/docs/additional_topics.html#flat-trend-and-custom-trends)
# option, (only available in python), with 
# [linear and logistic growth I used earlier in R](https://github.com/makeyourownmaker/CambridgeTemperatureModel/blob/master/4.02-prophet.R).
# 
# Seasonality mode defaults to additive for both daily and yearly.  Yearly seasonality is set to use 2 Fourier terms to enforce smooth annual cyclicality.  Yearly seasonality shows over-fitting if `yearly_seasonality = 'auto'` is used.
# It _may_ be better to use `yearly_seasonality = 'auto'` and tune `seasonality_prior_scale` instead of setting the number of Fourier terms.

# In[5]:


m = Prophet(growth = 'flat',
            daily_seasonality  = True,
            weekly_seasonality = False,
            yearly_seasonality = 2)
m.fit(df_train);


# ### Make forecast
# 
# Use `df_test` data created earlier to make forecast.  `df_test` contains data in 2019 and after.
# `yhat_lower` and `yhat_upper` are the 80% uncertainty intervals.

# In[7]:


forecast = m.predict(df_test)
pd.concat([forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().reset_index(drop = True), 
           df_test['y'].tail().reset_index(drop = True)], axis = 1)


# ### Plot forecast

# In[8]:


fig1 = m.plot(forecast)


# ### Plot components of simple model forecast
# 
# As expected the trend is flat and is close to the mean `y` (temperature in C * 10) value from `df.describe()` above (101.93).
# 
# The seasonalities are smoothly varying and cyclic.
# The daily and yearly seasonalities resemble the 
# [seasonalities obtained with R](https://github.com/makeyourownmaker/CambridgeTemperatureModel#Seasonality).

# In[9]:


fig2 = m.plot_components(forecast)


# ---
# 
# ## Cross-validate Model
# 
# Second, cross-validate the simple model to get some indication of performance.
# 
# To perform [rolling origin forecasting](https://otexts.com/fpp2/accuracy.html#time-series-cross-validation) first build a prophet model on `df` instead of `df_train`.  
# Then run cross-validation on a horizon of 1 hour, starting with 90,000 hours (over 10 years) of training data in the first cutoff and then making predictions every 1,000 hours.  On this 11 year time series, this corresponds to 11 total forecasts between 2018-11-25 09:00:00 and 2020-01-16 01:00:00.  1,000 hours is used to get get a range of forecasts throughout the year.  This is a small validation set.
# 
# I'm primarily interested in making "nowcasts" (forecasts in the next 1 to 2 hours) because I live very close to the data source and the UK met office still only update the forecasts on their web site every 2 hours. 

# In[10]:


m = Prophet(growth = 'flat',
            daily_seasonality  = True,
            weekly_seasonality = False,
            yearly_seasonality = 2)
m.fit(df)

df_cv = cross_validation(m,                                                  
                         initial = '90000 hours',
                         period  = '1000 hours',
                         horizon = '1 hours')
df_cv


# In[12]:


df_p = performance_metrics(df_cv)
df_p[['horizon', 'rmse', 'mae', 'mape']]


# These metrics are comparable to the basic linear and logistic model results previously obtained in R.

# ---
# 
# 
# ## Tune Discrete Prophet Hyperparameters
# 
# Third, tune the categorical parameters.
# 
# In general, Gaussian process-based Bayesian optimisation does not support discrete parameters.  Discrete parameters should probably be tuned before continuous parameters.
# 
# I previously added two additional regressors:
#  * dew point
#  * humidity
# 
# Dew point is temperature to which air must be cooled to become saturated with water vapor, and
# humdidity is atmospheric moisture.
# 
# These discrete hyperparameters were tuned:
#  * growth
#  * daily seasonality mode
#  * yearly seasonality mode
#  * dew point mode
#  * humidity mode
#  
# Final hyperparameters from 
# [R version](https://github.com/makeyourownmaker/CambridgeTemperatureModel/blob/master/4.02-prophet.R):
# ```R
# params <- data.frame(growth = 'linear',
#                      n.changepoints = 0,
#                      daily.mode  = 'multiplicative',
#                      yearly.mode = 'additive',
#                      daily.fo  = 2,
#                      yearly.fo = 2,
#                      daily.prior  = 0.01,
#                      yearly.prior = 0.01,
#                      dp.mode   = 'multiplicative',
#                      hum.mode  = 'multiplicative',
#                      dp.prior  = 0.01,
#                      hum.prior = 0.01,
#                      stringsAsFactors = FALSE)
# ```
# 
# When setting `n.changepoints = 0` the trend often showed small amounts of postive or negative growth.
# 
# The seasonality and regressor priors were tuned over a small range of values \[0.01, 0.1, 1, 10\].
# The low number of Fourier terms and low prior values results in the seasonality components being dominated by the regressors.  This suggests the seasonality is not crucial for such short-term forecasts.
# It will be interesting to see at what horizon seasonality becomes less dominated.  The tuning grid should be extended to 100 or higher.
# 
# Nonetheless, these parameters give much improved results compared to the simple model above.
# 
# | horizon  | rmse     | mae      | mape     |
# |----------|----------|----------|----------|
# | 00:30:00 | 6.784561 | 5.127664 | 0.069588 |
# | 01:00:00 | 8.080058 | 5.304410 | 0.076183 |
# 
# I won't translate the R code to python.  See the [prophet docs](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)
# for a python grid search example.

# ---
# 
# 
# ## Bayesian Optimisation of Continuous Prophet Hyperparameters
# 
# Fourth, tune the continuous parameters.
# 
# Below I give an example of Bayesian optimisation of prophet seasonality and regressor parameters.
# 
# These continuous parameters are optimised:
#  * daily_prior_scale
#  * yearly_prior_scale
#  * dew point prior scale
#  * humidity prior scale
#  
# The python [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
# library is used.  It is an implementation of constrained global optimization with Gaussian processes.
# 
# Next, I define the prophet model to optimise.

# In[15]:


def prophet_f(daily_prior, yearly_prior, hum_prior, dp_prior, metric = 'rmse', period = '1000 hours'):
    """
    Implements the prophet model to be optimised and performs cross-validation
    
    Args:
        daily_prior:  daily seasonality prior scale
        yearly_prior: yearly seasonality prior scale
        hum_prior:    humidity regressor prior scale 
        dp_prior:     dew.point regressor prior scale
        metric:       metric(s) to return - 'rmse' or ['horizon', 'rmse', 'mae', 'mape']
        period:       cross-validation period

    Returns:
        negative of root mean square error
    """
                     
    m = Prophet(growth = 'flat',
                weekly_seasonality = False)
    
    m.add_seasonality(name   = 'daily',
                      period = 1,
                      mode   = 'multiplicative',
                      prior_scale   = 10 ** daily_prior,
                      fourier_order = 2)
    m.add_seasonality(name   = 'yearly',
                      period = 365.25,
                      mode   = 'additive',
                      prior_scale   = 10 ** yearly_prior,
                      fourier_order = 2)
    
    m.add_regressor('humidity',
                    mode = 'multiplicative',
                    prior_scale = 10 ** hum_prior)
    m.add_regressor('dew.point',
                    mode = 'multiplicative',
                    prior_scale = 10 ** dp_prior)
    
    m.fit(df)
    df_cv = cross_validation(m,                                                  
                             initial = '90000 hours',
                             period  = period,
                             horizon = '1 hours')
    
    if metric == 'rmse':
        df_cv_rmse = ((df_cv.y - df_cv.yhat) ** 2).mean() ** .5
        return - df_cv_rmse
    elif metric == 'all':
        df_p = performance_metrics(df_cv)
        return m, df_p[['horizon', 'rmse', 'mae', 'mape']]
    


# **WARNING** Next cell may take quite a while to run.
# 
# Run time can be reduced by 
#  * decreasing `init_points` and/or `n_iter` in the `optimizer.maximize` call below
#  or
#  * increasing `period` (up to maximum of '15000 hours') in the `cross_validation` call in the `prophet_f` function above.
#  
# **NOTE** It may be necessary to scroll down through the runtime messages to find the optimised parameters.
# Or, set `verbose = 0` in the `BayesianOptimization` call below.

# In[16]:


# daily_prior_scale calculated as 10 ** daily_prior in prophet_f
pbounds = {'daily_prior': (-2, 2), 'yearly_prior': (-2, 2),
           'hum_prior':   (-2, 2), 'dp_prior':     (-2, 2)}

optimizer = BayesianOptimization(
    f = prophet_f,
    pbounds = pbounds,
    verbose = 2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state = 1
)

optimizer.maximize(
    init_points = 10,
    n_iter = 10)

print("\nMax params:")
print(optimizer.max)


# `yearly_prior` has converged to the maximum limit (2), `dp_prior` and `hum_prior` have converged to the minimum limit (-2).
# Anecdotally, it seemed like the Bayesian optimisation was bouncing around in a shallow basin.
# `daily_prior` may or may not have converged.  We can check by setting new bounds to optimise only `daily_prior`.

# In[17]:


# daily_prior_scale calculated as 10 ** daily_prior in prophet_f
pbounds_red = {'daily_prior': (-2, 2),  'yearly_prior': (2, 2),
               'hum_prior':   (-2, -2), 'dp_prior':     (-2, -2)}

optimizer.set_bounds(new_bounds = pbounds_red)

optimizer.maximize(
    init_points = 0,
    n_iter = 10)

print("\nMax params:")
print(optimizer.max)


# There was a marginal improvement in rmse value from setting new bounds to optimise only `daily_prior`.  We are firmly into the _micro-optimisation theatre_ regime, so time to stop optimisation.
# 
# ### Compare optimised model with earlier models
# 
# How does the optimised model compare with the earlier models?

# In[18]:


print("Simple model:")
display(df_p[['horizon', 'rmse', 'mae', 'mape']])


# Model with tuned discrete parameters and small grid search over continuous parameters:
# 
# | horizon  | rmse     | mae      | mape     |
# |----------|----------|----------|----------|
# | 00:30:00 | 6.784561 | 5.127664 | 0.069588 |
# | 01:00:00 | 8.080058 | 5.304410 | 0.076183 |
# 

# In[19]:


m_opt, m_diags_opt = prophet_f(optimizer.max['params']['daily_prior'], 
                               optimizer.max['params']['yearly_prior'], 
                               optimizer.max['params']['hum_prior'],
                               optimizer.max['params']['dp_prior'],
                               metric = 'all',
                               period = '250 hours')

print("Model with Bayesian optimised parameters:")
m_diags_opt


# ### Plot components of forecast from optimised model
# 
# Finally, make a forecast with the optimised model and plot model components.

# In[20]:


forecast_opt = m_opt.predict(df_test)
fig3 = m_opt.plot_components(forecast_opt)


# ---
# 
# ## Conclusion
# 
# Findings:
#  * Bayesian optimised parameters
#    * it's notable that `yearly_prior` has converged to the maximum limit (2) and `dp_prior` plus `hum_prior` converged to the lower limit (-2)
#  * Components
#    * yearly and daily seasonalities appear overfitted
#      * I would expect a smooth cycle between the maximum and minimum values
#    * there may be some seasonality present in the extra regressors
#      * check for seasonality in performance of the model
#  * Diagnostics
#    * the Bayesian optimised model is superior to the simple model
#    * the Bayesian optimised model is comparable to the partially optimised grid search model
#  * Number of model evaluations
#    * unfortunately, we cannot make a good comparison between the R grid search and the Bayesian optimisation
#      * I used a more restricted parameter range in the R grid search
#      * a comparable grid would be [0.01, 0.1, 0, 1, 10, 100] meaning potentially 625 (5 ** 4) model evaluations for these 4 continuous parameters
#      * this compares well with the 30 model evaluations performed for Bayesian optimisation
#    * presumably a comparable random search would not give better results than Bayesian optimisation
# 
# It's disappointing that the Bayesian optimised model has worse performance than the 
# [simple exponential smoothing baseline model](https://github.com/makeyourownmaker/CambridgeTemperatureModel#one-step-ahead-baselines) :-(
# 
# Future work could include:
#  * Plot individual effect of each regressor as done in this [weather related prophet notebook](https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb)
#  * Consider deseasonalising the regressors
#  * Explore addition of lagged regressors
#  * Expand cross-validation horizon to 2 hours
#    * limited to 1 hour here to reduce compute time
#  * Plot optimisation progress
#    * the bayes_opt package does not include any built-in plots
#    * plots could show if there are any unexplored areas of parameter space
#  * Cross-validation would benefit from being 
# [parallelised](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)
# 
# In conclusion, the Bayesian optimised model is much improved but the possible seasonality in the regressors needs further investigation.

# ---
# 
# 
# ## Archival
# 
# Archive code, markdown, history and formatted notebooks.
# 
# Assumes all pdf, html, latex etc dependencies are installed.
# 
# **WARNING** Will overwrite existing files.

# In[21]:


notebook = "BayesOptProphetHyperparameters.ipynb"
# !jupyter nbconvert --to script {notebook}
# !jupyter nbconvert --execute --to html {notebook}
# !jupyter nbconvert --execute --to pdf {notebook}
# !jupyter nbconvert --to pdf {notebook}

get_ipython().run_line_magic('rm', 'history.txt')
get_ipython().run_line_magic('history', '-f history.txt')

get_ipython().system('jupyter nbconvert --to python {notebook}')
sleep(5)
get_ipython().system('jupyter nbconvert --to markdown {notebook}')
sleep(5)
get_ipython().system('jupyter nbconvert --to html {notebook}')


# ---
# 
# ## Metadata
# 
# Python and Jupyter versions plus modules imported and their version strings.
# This is the poor man's python equivalent of R's [`sessionInfo()`](https://stat.ethz.ch/R-manual/R-patched/library/utils/html/sessionInfo.html).
# 
# Code for imported modules and versions adapted from this [stackoverflow answer](https://stackoverflow.com/a/49199019/100129).  There are simpler alternatives,
# such as [watermark](https://github.com/rasbt/watermark),
# but they all require installation.

# In[22]:


import sys
import IPython

print("Python version:")
print(sys.executable)
print(sys.version)
print("\nIPython version:")
print(IPython.__version__)


# In[23]:


import pkg_resources
import types

def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names.  Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name.  You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL":       "Pillow",
            "sklearn":   "scikit-learn",
            "bayes_opt": "bayesian-optimization",
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name

imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name != "pip":
        requirements.append((m.project_name, m.version))

reqs = pd.DataFrame(requirements, columns = ['name', 'version'])
print("Imported modules:")
reqs.style.hide_index()


# In[24]:


get_ipython().system('date')

