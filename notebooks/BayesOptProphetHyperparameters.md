```python
%matplotlib inline

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from fbprophet import Prophet
from bayes_opt import BayesianOptimization

random.seed(42)
np.random.seed(42)
```

    Importing plotly failed. Interactive plots will not work.


# Bayesian Optimisation of Prophet Hyperparameters

This notebook illustrates optimisation of continuous hyperparamaters of prophet models using the 
[BayesianOptimization](https://github.com/fmfn/BayesianOptimization) package.

The notebook is organised into the following sections:
 * Importing Data
 * Building Simple Model
 * Cross-validating Model
 * Tuning Discrete Prophet Hyperparameters
 * Bayesian Optimisation of Continuous Prophet Hyperparameters
 * Conclusion


## Import Data

Data has been cleaned but may still have issues.  See the [cleaning section](https://github.com/makeyourownmaker/CambridgeTemperatureModel#Cleaning) in my [Cambridge Temperature Model repository](https://github.com/makeyourownmaker/CambridgeTemperatureModel) for details.

The `y` variable is temperature * 10.  I'm primarily interested in very short term forecasts (less than 2 hours)
but forecasts over 24 hours are also interesting.


```python
df = pd.read_csv("../data/CamUKWeather.csv", parse_dates=True)
print("Shape:")
print(df.shape)
print("\nInfo:")
print(df.info())
print("\nSummary stats:")
display(df.describe())
print("\nRaw data:")
df
```

    Shape:
    (192885, 11)
    
    Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 192885 entries, 0 to 192884
    Data columns (total 11 columns):
     #   Column             Non-Null Count   Dtype 
    ---  ------             --------------   ----- 
     0   ds                 192885 non-null  object
     1   year               192885 non-null  int64 
     2   doy                192885 non-null  int64 
     3   time               192885 non-null  object
     4   y                  192885 non-null  int64 
     5   humidity           192885 non-null  int64 
     6   dew.point          192885 non-null  int64 
     7   pressure           192885 non-null  int64 
     8   wind.speed.mean    192885 non-null  int64 
     9   wind.bearing.mean  192885 non-null  int64 
     10  wind.speed.max     192885 non-null  int64 
    dtypes: int64(9), object(2)
    memory usage: 16.2+ MB
    None
    
    Summary stats:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>doy</th>
      <th>y</th>
      <th>humidity</th>
      <th>dew.point</th>
      <th>pressure</th>
      <th>wind.speed.mean</th>
      <th>wind.bearing.mean</th>
      <th>wind.speed.max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>192885.000000</td>
      <td>192885.000000</td>
      <td>192885.000000</td>
      <td>192885.000000</td>
      <td>192885.000000</td>
      <td>192885.000000</td>
      <td>192885.000000</td>
      <td>192885.000000</td>
      <td>192885.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2013.895803</td>
      <td>186.882298</td>
      <td>101.096819</td>
      <td>79.239951</td>
      <td>62.135174</td>
      <td>1014.404153</td>
      <td>44.588148</td>
      <td>196.223423</td>
      <td>117.140369</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.283992</td>
      <td>106.486420</td>
      <td>64.465602</td>
      <td>16.908724</td>
      <td>51.016879</td>
      <td>11.823922</td>
      <td>40.025546</td>
      <td>82.458390</td>
      <td>80.116199</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2008.000000</td>
      <td>1.000000</td>
      <td>-138.000000</td>
      <td>25.000000</td>
      <td>-143.000000</td>
      <td>963.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2011.000000</td>
      <td>94.000000</td>
      <td>52.000000</td>
      <td>69.000000</td>
      <td>25.000000</td>
      <td>1008.000000</td>
      <td>12.000000</td>
      <td>135.000000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2014.000000</td>
      <td>191.000000</td>
      <td>100.000000</td>
      <td>83.000000</td>
      <td>64.000000</td>
      <td>1016.000000</td>
      <td>35.000000</td>
      <td>225.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2017.000000</td>
      <td>280.000000</td>
      <td>145.000000</td>
      <td>92.000000</td>
      <td>100.000000</td>
      <td>1023.000000</td>
      <td>67.000000</td>
      <td>270.000000</td>
      <td>160.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2020.000000</td>
      <td>366.000000</td>
      <td>361.000000</td>
      <td>100.000000</td>
      <td>216.000000</td>
      <td>1048.000000</td>
      <td>291.000000</td>
      <td>315.000000</td>
      <td>580.000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Raw data:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>year</th>
      <th>doy</th>
      <th>time</th>
      <th>y</th>
      <th>humidity</th>
      <th>dew.point</th>
      <th>pressure</th>
      <th>wind.speed.mean</th>
      <th>wind.bearing.mean</th>
      <th>wind.speed.max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008-08-01 08:30:00</td>
      <td>2008</td>
      <td>214</td>
      <td>09:30:00</td>
      <td>186</td>
      <td>69</td>
      <td>128</td>
      <td>1010</td>
      <td>123</td>
      <td>180</td>
      <td>280</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008-08-01 09:00:00</td>
      <td>2008</td>
      <td>214</td>
      <td>10:00:00</td>
      <td>191</td>
      <td>70</td>
      <td>135</td>
      <td>1010</td>
      <td>137</td>
      <td>180</td>
      <td>260</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2008-08-01 09:30:00</td>
      <td>2008</td>
      <td>214</td>
      <td>10:30:00</td>
      <td>195</td>
      <td>68</td>
      <td>134</td>
      <td>1010</td>
      <td>133</td>
      <td>180</td>
      <td>260</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2008-08-01 10:00:00</td>
      <td>2008</td>
      <td>214</td>
      <td>11:00:00</td>
      <td>200</td>
      <td>68</td>
      <td>139</td>
      <td>1010</td>
      <td>129</td>
      <td>180</td>
      <td>240</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2008-08-01 10:30:00</td>
      <td>2008</td>
      <td>214</td>
      <td>11:30:00</td>
      <td>213</td>
      <td>61</td>
      <td>135</td>
      <td>1010</td>
      <td>145</td>
      <td>180</td>
      <td>260</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>192880</th>
      <td>2020-01-16 00:00:00</td>
      <td>2020</td>
      <td>16</td>
      <td>00:00:00</td>
      <td>40</td>
      <td>78</td>
      <td>5</td>
      <td>1017</td>
      <td>45</td>
      <td>180</td>
      <td>100</td>
    </tr>
    <tr>
      <th>192881</th>
      <td>2020-01-16 00:30:00</td>
      <td>2020</td>
      <td>16</td>
      <td>00:30:00</td>
      <td>36</td>
      <td>86</td>
      <td>15</td>
      <td>1018</td>
      <td>25</td>
      <td>180</td>
      <td>120</td>
    </tr>
    <tr>
      <th>192882</th>
      <td>2020-01-16 01:00:00</td>
      <td>2020</td>
      <td>16</td>
      <td>01:00:00</td>
      <td>36</td>
      <td>85</td>
      <td>13</td>
      <td>1018</td>
      <td>28</td>
      <td>180</td>
      <td>80</td>
    </tr>
    <tr>
      <th>192883</th>
      <td>2020-01-16 01:30:00</td>
      <td>2020</td>
      <td>16</td>
      <td>01:30:00</td>
      <td>36</td>
      <td>82</td>
      <td>8</td>
      <td>1018</td>
      <td>17</td>
      <td>180</td>
      <td>80</td>
    </tr>
    <tr>
      <th>192884</th>
      <td>2020-01-16 02:00:00</td>
      <td>2020</td>
      <td>16</td>
      <td>02:00:00</td>
      <td>36</td>
      <td>89</td>
      <td>20</td>
      <td>1018</td>
      <td>22</td>
      <td>180</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
<p>192885 rows × 11 columns</p>
</div>



Create train and test data.  Will perform [rolling origin forecasting](https://otexts.com/fpp2/accuracy.html#time-series-cross-validation) later.


```python
threshold = 2019
df_test  = df[df['year'] >= threshold]
df_train = df[df['year'] <  threshold]
```

---


## Build Simple Model

First, start by building a simple model with flat growth and no weekly seasonality.  This is a quick sanity check.  Results should be similar to my previous R version.

One reason for using the python prophet version over R is to check the 
[flat growth](https://facebook.github.io/prophet/docs/additional_topics.html#flat-trend-and-custom-trends)
option, (only available in python), with 
[linear and logistic growth I used earlier in R](https://github.com/makeyourownmaker/CambridgeTemperatureModel/blob/master/4.02-prophet.R).

Seasonality mode defaults to additive for both daily and yearly.  Yearly seasonality is set to use 2 Fourier terms to enforce smooth annual cyclicality.  Yearly seasonality shows over-fitting if `yearly_seasonality='auto'` is used.
It _may_ be better to use `yearly_seasonality='auto'` and tune `seasonality_prior_scale` instead of setting the number of Fourier terms.


```python
m = Prophet(growth='flat',
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=2)
m.fit(df_train);
```

### Make forecast

Use `df_test` data created earlier to make forecast.  `df_test` contains data in 2019 and after.
`yhat_lower` and `yhat_upper` are the 80% uncertainty intervals.


```python
forecast = m.predict(df_test)
pd.concat([forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().reset_index(drop=True), 
           df_test['y'].tail().reset_index(drop=True)], axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-16 00:00:00</td>
      <td>17.667960</td>
      <td>-29.847775</td>
      <td>62.817637</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-16 00:30:00</td>
      <td>16.180201</td>
      <td>-29.955690</td>
      <td>60.990527</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-16 01:00:00</td>
      <td>14.807732</td>
      <td>-31.797248</td>
      <td>64.647189</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-16 01:30:00</td>
      <td>13.508647</td>
      <td>-32.661017</td>
      <td>61.800024</td>
      <td>36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-16 02:00:00</td>
      <td>12.256809</td>
      <td>-31.535405</td>
      <td>59.283150</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>



### Plot forecast


```python
fig1 = m.plot(forecast)
```


    
![png](BayesOptProphetHyperparameters_files/BayesOptProphetHyperparameters_10_0.png)
    


### Plot components of simple model forecast

As expected the trend is flat and is close to the mean `y` (temperature in C * 10) value from `df.describe()` above (101.93).

The seasonalities are smoothly varying and cyclic.
The daily and yearly seasonalities resemble the 
[seasonalities obtained with R](https://github.com/makeyourownmaker/CambridgeTemperatureModel#Seasonality).


```python
fig2 = m.plot_components(forecast)
```


    
![png](BayesOptProphetHyperparameters_files/BayesOptProphetHyperparameters_12_0.png)
    


---

## Cross-validate Model

Second, cross-validate the simple model to get some indication of performance.

To perform [rolling origin forecasting](https://otexts.com/fpp2/accuracy.html#time-series-cross-validation) first build a prophet model on `df` instead of `df_train`.  
Then run cross-validation on a horizon of 1 hour, starting with 90000 hours (over 10 years) of training data in the first cutoff and then making predictions every 1000 hours. On this 11 year time series, this corresponds to 11 total forecasts between 2018-11-25 09:00:00 and 2020-01-16 01:00:00.  This is a small validation set.

I'm primarily interested in making "nowcasts" (forecasts in the next 1 to 2 hours) because I live very close to the data source and the UK met office only update the forecasts on their web site every 2 hours. 


```python
from fbprophet.diagnostics import cross_validation, performance_metrics

m = Prophet(growth='flat',
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=2)
m.fit(df)

df_cv = cross_validation(m,                                                  
                         initial='90000 hours',
                         period='1000 hours',
                         horizon='1 hours')
df_cv
```

    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
      <th>y</th>
      <th>cutoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-11-25 09:30:00</td>
      <td>69.739130</td>
      <td>19.884369</td>
      <td>117.277538</td>
      <td>40</td>
      <td>2018-11-25 09:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-11-25 10:00:00</td>
      <td>74.736026</td>
      <td>26.711922</td>
      <td>122.259926</td>
      <td>44</td>
      <td>2018-11-25 09:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-01-06 01:30:00</td>
      <td>14.903101</td>
      <td>-29.389721</td>
      <td>59.957277</td>
      <td>32</td>
      <td>2019-01-06 01:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-01-06 02:00:00</td>
      <td>13.648006</td>
      <td>-28.144470</td>
      <td>58.607385</td>
      <td>32</td>
      <td>2019-01-06 01:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-02-16 17:30:00</td>
      <td>60.864809</td>
      <td>16.395474</td>
      <td>109.247482</td>
      <td>92</td>
      <td>2019-02-16 17:00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-02-16 18:00:00</td>
      <td>57.318057</td>
      <td>13.104304</td>
      <td>104.092568</td>
      <td>88</td>
      <td>2019-02-16 17:00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2019-03-30 09:30:00</td>
      <td>78.062718</td>
      <td>31.079447</td>
      <td>124.812803</td>
      <td>123</td>
      <td>2019-03-30 09:00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2019-03-30 10:00:00</td>
      <td>83.093376</td>
      <td>35.050668</td>
      <td>129.022228</td>
      <td>141</td>
      <td>2019-03-30 09:00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2019-05-11 01:30:00</td>
      <td>91.757079</td>
      <td>46.765174</td>
      <td>135.244288</td>
      <td>56</td>
      <td>2019-05-11 01:00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-05-11 02:00:00</td>
      <td>90.536030</td>
      <td>44.309768</td>
      <td>139.393141</td>
      <td>56</td>
      <td>2019-05-11 01:00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2019-06-21 17:30:00</td>
      <td>177.967667</td>
      <td>133.398878</td>
      <td>223.433699</td>
      <td>186</td>
      <td>2019-06-21 17:00:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2019-06-21 18:00:00</td>
      <td>174.412222</td>
      <td>128.890993</td>
      <td>220.406266</td>
      <td>177</td>
      <td>2019-06-21 17:00:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019-08-02 09:30:00</td>
      <td>183.581257</td>
      <td>137.855582</td>
      <td>228.725586</td>
      <td>186</td>
      <td>2019-08-02 09:00:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019-08-02 10:00:00</td>
      <td>188.600265</td>
      <td>141.070477</td>
      <td>237.462060</td>
      <td>182</td>
      <td>2019-08-02 09:00:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2019-09-13 01:30:00</td>
      <td>126.944368</td>
      <td>79.875884</td>
      <td>174.356863</td>
      <td>155</td>
      <td>2019-09-13 01:00:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2019-09-13 02:00:00</td>
      <td>125.625959</td>
      <td>78.915122</td>
      <td>170.545562</td>
      <td>145</td>
      <td>2019-09-13 01:00:00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2019-10-24 17:30:00</td>
      <td>117.622934</td>
      <td>73.074388</td>
      <td>164.331569</td>
      <td>88</td>
      <td>2019-10-24 17:00:00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2019-10-24 18:00:00</td>
      <td>113.996222</td>
      <td>68.131090</td>
      <td>160.379466</td>
      <td>84</td>
      <td>2019-10-24 17:00:00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2019-12-05 09:30:00</td>
      <td>60.190404</td>
      <td>14.470598</td>
      <td>105.778553</td>
      <td>12</td>
      <td>2019-12-05 09:00:00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2019-12-05 10:00:00</td>
      <td>65.223833</td>
      <td>22.867112</td>
      <td>110.798999</td>
      <td>20</td>
      <td>2019-12-05 09:00:00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2020-01-16 01:30:00</td>
      <td>13.998328</td>
      <td>-34.842082</td>
      <td>58.477476</td>
      <td>36</td>
      <td>2020-01-16 01:00:00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2020-01-16 02:00:00</td>
      <td>12.720485</td>
      <td>-31.944373</td>
      <td>58.185447</td>
      <td>36</td>
      <td>2020-01-16 01:00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_p = performance_metrics(df_cv)
df_p[['horizon','rmse','mae','mape']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>horizon</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00:30:00</td>
      <td>30.129179</td>
      <td>26.998845</td>
      <td>0.710990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01:00:00</td>
      <td>31.161329</td>
      <td>27.206752</td>
      <td>0.554322</td>
    </tr>
  </tbody>
</table>
</div>



These metrics are comparable to the basic linear and logistic model results previously obtained in R.

---


## Tune Discrete Prophet Hyperparameters

Third, tune the categorical parameters.

In general, Gaussian process-based Bayesian optimisation does not support discrete parameters.  Discrete parameters should probably be tuned before continuous parameters.

I previously added two additional regressors:
 * dew point
 * humidity

Dew point is temperature to which air must be cooled to become saturated with water vapor, and
humdidity is atmospheric moisture.

These discrete hyperparameters were tuned:
 * growth
 * daily seasonality mode
 * yearly seasonality mode
 * dew point mode
 * humidity mode
 
Final hyperparameters from 
[R version](https://github.com/makeyourownmaker/CambridgeTemperatureModel/blob/master/4.02-prophet.R):
```R
params <- data.frame(growth='linear',
                     n.changepoints=0,
                     daily.mode='multiplicative',
                     yearly.mode='additive',
                     daily.fo=2,
                     yearly.fo=2,
                     daily.prior=0.01,
                     yearly.prior=0.01,
                     dp.mode='multiplicative',
                     hum.mode='multiplicative',
                     dp.prior=0.01,
                     hum.prior=0.01,
                     stringsAsFactors=FALSE)
```

When setting `n.changepoints=0` the trend often showed small amounts of postive or negative growth.

The seasonality and regressor priors were tuned over a small range of values \[0.01, 0.1, 1, 10\].
The low number of Fourier terms and low prior values results in the seasonality components being dominated by the regressors.  This suggests the seasonality is not useful for such short-term forecasts.
It will be interesting to see at what horizon seasonality becomes less dominated.  Alternatively, the tuning grid could be extended to 100 or higher.

Nonetheless, these parameters give much improved results compared to the simple model above.

| horizon  | rmse     | mae      | mape     |
|----------|----------|----------|----------|
| 00:30:00 | 6.784561 | 5.127664 | 0.069588 |
| 01:00:00 | 8.080058 | 5.304410 | 0.076183 |

I won't translate the R code to python.  See the [prophet docs](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning)
for a python grid search example.

---


## Bayesian Optimisation of Continuous Prophet Hyperparameters

Fourth, tune the continuous parameters.

Below I give an example of Bayesian optimisation of seasonality and regressor parameters.

These continuous parameters are optimised:
 * daily_prior_scale
 * yearly_prior_scale
 * dew point prior scale
 * humidity prior scale
 
The python [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
library is used.  It is an implementation of constrained global optimization with Gaussian processes.

Next, I define the prophet model to optimise.


```python
def prophet_f(daily_prior, yearly_prior, hum_prior, dp_prior, metric='rmse', period='1000 hours'):
    """
    Implements the prophet model to be optimised and performs cross-validation
    
    Args:
        daily_prior:  daily seasonality prior scale
        yearly_prior: yearly seasonality prior scale
        hum_prior:    humidity regressor prior scale 
        dp_prior:     dew.point regressor prior scale

    Returns:
        negative of root mean square error
    """
                     
    m = Prophet(growth='flat',
                weekly_seasonality=False)
    
    m.add_seasonality(name='daily',
                      period=1,
                      fourier_order=2,
                      mode='multiplicative',
                      prior_scale=10 ** daily_prior)
    m.add_seasonality(name='yearly',
                      period=365.25,
                      fourier_order=2,
                      mode='additive',
                      prior_scale=10 ** yearly_prior)
    
    m.add_regressor('humidity',
                    mode='multiplicative',
                    prior_scale=10 ** hum_prior)
    m.add_regressor('dew.point',
                    mode='multiplicative',
                    prior_scale=10 ** dp_prior)
    
    m.fit(df)
    df_cv = cross_validation(m,                                                  
                             initial='90000 hours',
                             period=period,
                             horizon='1 hours')
    
    if metric == 'rmse':
        df_cv_rmse = ((df_cv.y - df_cv.yhat) ** 2).mean() ** .5
        return - df_cv_rmse
    else:
        df_p = performance_metrics(df_cv)
        return m, df_p[['horizon','rmse','mae','mape']]
    
```

**WARNING** Next cell may take quite a while to run.

Run time can be reduced by 
 * decreasing `init_points` and/or `n_iter` in the `optimizer.maximize` call below
 or
 * increasing `period` (up to maximum of '15000 hours') in the `cross_validation` call in the `prophet_f` function above.
 
**NOTE** It may be necessary to scroll down through the runtime messages to find the optimised parameters.
Or, set `verbose=0` in the `BayesianOptimization` call below.


```python
# daily_prior_scale calculated as 10 ** daily_prior in prophet_f
pbounds = {'daily_prior': (-2, 2), 'yearly_prior': (-2, 2),
           'hum_prior':   (-2, 2), 'dp_prior':     (-2, 2)}

optimizer = BayesianOptimization(
    f=prophet_f,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1
)

optimizer.maximize(
    init_points=10,
    n_iter=10)

print("\nMax params:")
print(optimizer.max)

```

    |   iter    |  target   | daily_... | dp_prior  | hum_prior | yearly... |
    -------------------------------------------------------------------------


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 1       [0m | [0m-5.892   [0m | [0m-0.3319  [0m | [0m 0.8813  [0m | [0m-2.0     [0m | [0m-0.7907  [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 2       [0m | [0m-5.894   [0m | [0m-1.413   [0m | [0m-1.631   [0m | [0m-1.255   [0m | [0m-0.6178  [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 3       [0m | [0m-5.896   [0m | [0m-0.4129  [0m | [0m 0.1553  [0m | [0m-0.3232  [0m | [0m 0.7409  [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 4       [0m | [0m-5.893   [0m | [0m-1.182   [0m | [0m 1.512   [0m | [0m-1.89    [0m | [0m 0.6819  [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 5       [0m | [0m-5.896   [0m | [0m-0.3308  [0m | [0m 0.2348  [0m | [0m-1.438   [0m | [0m-1.208   [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 6       [0m | [0m-5.896   [0m | [0m 1.203   [0m | [0m 1.873   [0m | [0m-0.7463  [0m | [0m 0.7693  [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 7       [0m | [0m-5.895   [0m | [0m 1.506   [0m | [0m 1.578   [0m | [0m-1.66    [0m | [0m-1.844   [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 8       [0m | [0m-5.896   [0m | [0m-1.321   [0m | [0m 1.513   [0m | [0m-1.607   [0m | [0m-0.3156  [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 9       [0m | [0m-5.896   [0m | [0m 1.832   [0m | [0m 0.1327  [0m | [0m 0.7675  [0m | [0m-0.7379  [0m |


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 10      [0m | [0m-5.893   [0m | [0m 0.746   [0m | [0m 1.339   [0m | [0m-1.927   [0m | [0m 1.001   [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 11      [0m | [0m-5.896   [0m | [0m-1.969   [0m | [0m-1.353   [0m | [0m 0.8375  [0m | [0m 1.059   [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 12      [0m | [0m-5.896   [0m | [0m-0.2298  [0m | [0m-0.3461  [0m | [0m 0.1912  [0m | [0m 0.9768  [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 13      [0m | [0m-5.896   [0m | [0m 0.4935  [0m | [0m-1.107   [0m | [0m 1.836   [0m | [0m 1.219   [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 14      [0m | [0m-5.896   [0m | [0m-1.915   [0m | [0m 1.232   [0m | [0m 1.859   [0m | [0m 0.8083  [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [95m 15      [0m | [95m-5.891   [0m | [95m-0.5423  [0m | [95m-1.91    [0m | [95m-1.39    [0m | [95m 1.886   [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 16      [0m | [0m-5.892   [0m | [0m-0.2874  [0m | [0m 0.8332  [0m | [0m-1.989   [0m | [0m-0.8146  [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [95m 17      [0m | [95m-5.886   [0m | [95m-0.3065  [0m | [95m-2.0     [0m | [95m-1.753   [0m | [95m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [95m 18      [0m | [95m-5.885   [0m | [95m-0.1552  [0m | [95m-1.97    [0m | [95m-1.977   [0m | [95m 1.861   [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [95m 19      [0m | [95m-5.883   [0m | [95m 0.4908  [0m | [95m-2.0     [0m | [95m-2.0     [0m | [95m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2.  2.]
     [-2.  2.]
     [-2.  2.]]


    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 20      [0m | [0m-5.883   [0m | [0m 1.768   [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    =========================================================================
    
    Max params:
    {'target': -5.883217196442895, 'params': {'daily_prior': 0.4908015707468804, 'dp_prior': -2.0, 'hum_prior': -2.0, 'yearly_prior': 2.0}}


`yearly_prior` has converged to the maximum limit (2), `dp_prior` and `hum_prior` have converged to the minimum limit (-2).
Anecdotally, it seemed like the Bayesian optimisation was bouncing around in a shallow basin.
`daily_prior` may or may not have converged.  We can check by setting new bounds to optimise only `daily_prior`.


```python
# daily_prior_scale calculated as 10 ** daily_prior in prophet_f
pbounds_red = {'daily_prior': (-2, 2), 'yearly_prior': (2, 2),
               'hum_prior':   (-2, -2),    'dp_prior': (-2, -2)}

optimizer.set_bounds(new_bounds=pbounds_red)

optimizer.maximize(
    init_points=0,
    n_iter=10)

print("\nMax params:")
print(optimizer.max)

```

    |   iter    |  target   | daily_... | dp_prior  | hum_prior | yearly... |
    -------------------------------------------------------------------------
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 21      [0m | [0m-5.883   [0m | [0m 1.186   [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 22      [0m | [0m-5.883   [0m | [0m 2.0     [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [95m 23      [0m | [95m-5.883   [0m | [95m 0.7832  [0m | [95m-2.0     [0m | [95m-2.0     [0m | [95m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 24      [0m | [0m-5.883   [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 25      [0m | [0m-5.883   [0m | [0m-1.364   [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 26      [0m | [0m-5.883   [0m | [0m-1.708   [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 27      [0m | [0m-5.883   [0m | [0m 0.6975  [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 28      [0m | [0m-5.883   [0m | [0m 0.879   [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 29      [0m | [0m-5.883   [0m | [0m-0.7728  [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    bounds: 
    [[-2.  2.]
     [-2. -2.]
     [-2. -2.]
     [ 2.  2.]]


    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    /usr/local/lib/python3.8/site-packages/scipy/optimize/_numdiff.py:519: RuntimeWarning: invalid value encountered in true_divide
      J_transposed[i] = df / dx
    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 11 forecasts with cutoffs between 2018-11-25 09:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    | [0m 30      [0m | [0m-5.884   [0m | [0m 0.09288 [0m | [0m-2.0     [0m | [0m-2.0     [0m | [0m 2.0     [0m |
    =========================================================================
    
    Max params:
    {'target': -5.8830215248022, 'params': {'daily_prior': 0.7832480635982564, 'dp_prior': -2.0, 'hum_prior': -2.0, 'yearly_prior': 2.0}}


### Compare optimised model with earlier models

There was a marginal improvement in rmse value from setting new bounds to optimise only `daily_prior`.  We are firmly into the _micro-optimisation theatre_ regime, so time to stop optimisation.

How does the optimised model compare with the earlier models?


```python
print("Simple model:")
display(df_p[['horizon','rmse','mae','mape']])
```

    Simple model:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>horizon</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00:30:00</td>
      <td>30.129179</td>
      <td>26.998845</td>
      <td>0.710990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01:00:00</td>
      <td>31.161329</td>
      <td>27.206752</td>
      <td>0.554322</td>
    </tr>
  </tbody>
</table>
</div>


Model with tuned discrete parameters and small grid search over continuous parameters:

| horizon  | rmse     | mae      | mape     |
|----------|----------|----------|----------|
| 00:30:00 | 6.784561 | 5.127664 | 0.069588 |
| 01:00:00 | 8.080058 | 5.304410 | 0.076183 |



```python
m_opt, m_diags_opt = prophet_f(optimizer.max['params']['daily_prior'], 
                               optimizer.max['params']['yearly_prior'], 
                               optimizer.max['params']['hum_prior'],
                               optimizer.max['params']['dp_prior'],
                               metric='all',
                               period='250 hours')

print("Bayesian optimised parameters:")
m_diags_opt
```

    INFO:fbprophet:Found custom seasonality named 'yearly', disabling built-in 'yearly' seasonality.
    INFO:fbprophet:Found custom seasonality named 'daily', disabling built-in 'daily' seasonality.
    INFO:fbprophet:Making 42 forecasts with cutoffs between 2018-11-14 23:00:00 and 2020-01-16 01:00:00



    HBox(children=(FloatProgress(value=0.0, max=42.0), HTML(value='')))



```python
pd.DataFrame(m_diags_opt.mean()[['rmse','mae','mape']])
```

### Plot components of forecast from optimised model

Finally, make a forecast with the optimised model and plot model components.


```python
forecast_opt = m_opt.predict(df_test)
fig3 = m_opt.plot_components(forecast_opt)
```

---

## Conclusion

Findings:
 * Bayesian optimised parameters
   * it's notable that `yearly_prior` has converged to the maximum limit (2) and `dp_prior` plus `hum_prior` converged to the lower limit (-2)
 * Components
   * yearly and daily seasonalities appear overfitted to me
     * I would expect a smooth cycle with a single maximum and a single minimum
   * there may be some seasonality present in the extra regressors
     * check for seasonality in performance of the model
 * Diagnostics
   * the Bayesian optimised model is superior to the simple model
   * the Bayesian optimised model is comparable to the partially optimised grid search model
 * Number of model evaluations
   * unfortunately, we cannot make a good comparison between the R grid search and the Bayesian optimisation
     * I used a more restricted parameter range in the R grid search
     * a comparable grid would be [0.01, 0.1, 0, 1, 10, 100] meaning potentially 625 (5 ** 4) model evaluations for these 4 continuous parameters
     * this compares well with the 30 model evaluations performed for Bayesian optimisation
   * a comparable random search may not give better results than Bayesian optimisation

It's disappointing that the Bayesian optimised model has worse performance than the 
[simple exponential smoothing baseline model](https://github.com/makeyourownmaker/CambridgeTemperatureModel#one-step-ahead-baselines) :-(

Future work could include:
 * Plot individual affect of each regressor as done in this [weather related prophet notebook](https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb)
 * Consider deseasonalising the regressors
 * Explore addition of lagged regressors
 * Expand horizon to 2 hours
   * limited to 1 hour here to reduce compute time
 * Plot optimisation progress
   * the bayes_opt package does not include any built-in plots
   * plots could show if there are any unexplored areas of parameter space
 * Cross-validation would benefit from being 
[parallelised](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)

In conclusion, the Bayesian optimised model is much improved but the possible seasonality in the regressors needs further investigation.

---


## Archival

Archive code, markdown, history and formatted notebooks.

Assumes all pdf, html, latex etc dependencies are installed.

**WARNING** Will overwrite existing files.


```python
notebook = "BayesOptProphetHyperparameters.ipynb"
# !jupyter nbconvert --to script {notebook}
# !jupyter nbconvert --execute --to html {notebook}
# !jupyter nbconvert --execute --to pdf {notebook}
# !jupyter nbconvert --to pdf {notebook}

%rm history.txt
%history -f history.txt

!jupyter nbconvert --to python {notebook}
sleep(5)
!jupyter nbconvert --to markdown {notebook}
sleep(5)
!jupyter nbconvert --to html {notebook}
```

---

## Metadata

Python and Jupyter versions plus modules imported and their version strings.
This is the poor man's python equivalent of R's [`sessionInfo()`](https://stat.ethz.ch/R-manual/R-patched/library/utils/html/sessionInfo.html).

Code for imported modules and versions adapted from this [stackoverflow answer](https://stackoverflow.com/a/49199019/100129).  There are simpler alternatives,
such as [watermark](https://github.com/rasbt/watermark),
but they all require installation.


```python
import sys
import IPython

print("Python version:")
print(sys.executable)
print(sys.version)
print("\nIPython version:")
print(IPython.__version__)
```


```python
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

reqs = pd.DataFrame(requirements, columns=['name', 'version'])
print("Imported modules:")
reqs.style.hide_index()
```


```python
!date
```
