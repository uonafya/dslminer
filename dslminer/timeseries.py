import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import time
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet

from . import db


class predictor:
    def predict(self):

        _db = db.database()
        connection = _db.get_db_con()[0]
        cursor = _db.get_db_con()[1]

        SQL_Query = pd.read_sql_query(
        '''SELECT  distinct startdate, kpivalue
            FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"=61901 and "Org unit id"=18 order by startdate asc''', connection)

        data = pd.DataFrame(SQL_Query)
        _db.close_db_con()
        decomposition = sm.tsa.seasonal_decompose(data.kpivalue, freq=20)
        m = Prophet()
        data.columns = ['ds','y']

        m.fit(data)
        future = m.make_future_dataframe(periods = 3,freq = "Y")
        forecast = m.predict(future)
        print(decomposition.seasonal)
        print(decomposition.resid)
        return forecast
        #print(forecast.ds)


