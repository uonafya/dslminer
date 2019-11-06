import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import time
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from . import db

_db=db.database()
connection=_db.get_db_con()[0]
cursor=_db.get_db_con()[1]



SQL_Query = pd.read_sql_query(
'''SELECT  distinct startdate, kpivalue
    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"=61901 and "Org unit id"=18 order by startdate asc''', connection)


data = pd.DataFrame(SQL_Query)
_db.close_db_con()
# overwriting data after changing format
data["startdate"] = pd.to_datetime(data["startdate"])
data=data.set_index("startdate")


ts=data

#decomposing data
decomposition = sm.tsa.seasonal_decompose(ts.kpivalue, freq=20)
#fig = decomposition.plot()

# Determing rolling statistics
rolmean=ts.rolling(12).mean()
rolstd = ts.rolling(12).std()

# Plot rolling statistics:
#orig = ts.plot()
#mean = plt.plot(rolmean,color='red', label='Rolling Mean')
#std = plt.plot(rolstd,color='black', label='Rolling Std')
# print("deviation=====>")
# print(ts.head())
# print("deviation=====>")
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')

# Perform Dickey-Fuller test:
'Results of Dickey-Fuller Test:'
dftest = adfuller(ts['kpivalue'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print("engagement =======>")
print (dfoutput)

#Differencing
ts_log = np.log(ts)
ts_log_diff = ts_log - ts_log.shift()

#plt.plot(ts_log_diff)



#prediction # fit model
model = ARIMA(ts_log, order=(1,1,1))
results_ARIMA = model.fit(disp=1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title("title")

#copy predicted values
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)


#cummulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()


#original values
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#print("predictions =======>")
#print(predictions_ARIMA_log.head())

#compare with original graph
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f')

#moving_avg.plot()
#plt.plot(moving_avg, color='red')

#ts.plot()
plt.show()



