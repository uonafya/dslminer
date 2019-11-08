import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import time
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
from . import utils
from . import db
from flask import Flask

log = Flask(__name__)

class predictor:
    def predict(self,indicatorid,ouid=None,periodtype=None,periodspan=None):
        _db = db.database()
        connection = _db.get_db_con()[0]
        cursor = _db.get_db_con()[1]

        query_string = '''SELECT  distinct startdate, kpivalue
            FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"=%s and "Org unit id"=%s order by startdate asc''' %(indicatorid,ouid)
        SQL_Query = pd.read_sql_query(query_string, connection)
        print(query_string)

        data = pd.DataFrame(SQL_Query)
        _db.close_db_con()
        #decomposition = sm.tsa.seasonal_decompose(data.kpivalue, freq=20)
        m = Prophet()
        data.columns = ['ds','y']
        m.fit(data)

        if(periodspan==None):
            periodspan=int(2)

        if(periodtype==None):
            periodtype = 'Y'
        elif(periodtype=='yearly'):
            periodtype='Y'
        elif(periodtype=='monthly'):
            periodtype='M'


        future = m.make_future_dataframe(periods = periodspan,freq = periodtype)

        forecast = m.predict(future)
        log.logger.info("======> 1")
        log.logger.info(forecast)
        #log.logger.info(type(forecast.trend))
        log.logger.info(len(forecast.ds))
        trend = []
        projection = []
        yearly_trend = []
        orgunit_series_data={}
        indicator_series_data={}
        series_data_envelop = {}
        for index in range(len(forecast.ds)):
            month=str(forecast.ds[index].month)
            year=forecast.ds[index].year
            if(len(month)==1):
                month='0%s'%month
            _date=str(year)+str(month)

            _trend={}
            _trend["time"]=_date
            _trend["value"]=forecast.trend[index]
            trend.append(_trend)

            _yearly_trend = {}
            _yearly_trend["time"] = _date
            _yearly_trend["value"] = forecast.yearly[index]
            yearly_trend.append(_yearly_trend)

            _projection = {}
            _projection["time"] = _date
            _projection["value"] = forecast.yhat[index]
            projection.append(_projection)

        _orgunit_series_data ={
            "trend": trend,
            "projection": projection,
            "yearly": yearly_trend
        }
        orgunit_series_data[ouid]=_orgunit_series_data
        indicator_series_data[indicatorid] = orgunit_series_data
        series_data_envelop["data"]=indicator_series_data
            # print(decomposition.seasonal)
        return series_data_envelop


