import logging
from builtins import int

import pandas as pd
from . import db
from dateutil.relativedelta import *
import datetime
from sklearn import linear_model
from fbprophet import Prophet
import numpy
from. import utils

months_between = utils.months_between


# configurations
log = logging.getLogger("weather correlation")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)


class WeatherCorrelation:
    def __init__(self):
        self.begin_year=2010
        self.end_year=2019
        self._db = db.database()
        self.variables={}
        self.stationarity_iterations_count = 0
        self.max_indicator_dates = {}
        self.max_weather_dates = {}
        self._indicator_id= None

    def set_max_min_period(self,orgunit_id,indictor_id):
        """Sets the begin and end period to query data based on availablility of data from the given indicator
                Parameters
                ----------
                indictor_id : str
                    indicator id

                orgunit_id : str
                    org unit id

                Returns
                ----------
                void
                """
        max_min_period=''' SELECT  date_part('year',max(startdate)) as mx, date_part('year',min(startdate)) as mn
                    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"='''+str(indictor_id)+''' and "Org unit id"='''+str(orgunit_id)+ ''' and startdate>'2009-12-31' '''

        cursor = self._db.get_db_con()[1]
        log.info(max_min_period)
        cursor.execute(max_min_period)
        row = cursor.fetchall()
        log.info("============================")
        if (len(row) != 0):
            log.info("============================")
            self.end_year=int(row[0][0])
            if (int(row[0][0] < 2010)):
                pass
            else:
                self.begin_year = int(row[0][1])
            log.info("end year "+str(row[0][0]))
            log.info("start year " + str(self.begin_year))


    def get_indicator_data(self,ouid,indicatorid):

        query_string = '''SELECT  distinct startdate, ROUND(kpivalue,3), "Indicator ID" , _datecreated , lastupdated , "Indicator description" ,
                    "Organisation Unit Name" , "Org unit id" , "Indicator name" 
                    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"=%s and "Org unit id"=%s and startdate>='%s' and enddate<='%s' order by startdate asc''' % (
        indicatorid,ouid,str(self.begin_year)+"-01-01", str(self.end_year)+"-12-31" )


        cursor = self._db.get_db_con()[1]
        log.info(query_string)
        cursor.execute(query_string)
        rows = cursor.fetchall()
        indic_meta_list=[]
        org_meta_list = []

        indic_frame_values = {'startdate': [], "kpivalue": []}
        indic_payload_values = {}
        is_indict_meta_initialized = False
        for row in rows:
            if(not is_indict_meta_initialized):
                indicator_meta = {}
                self.variables[str(row[2])]=str(row[8])
                indicator_meta['name'] = row[8]
                indicator_meta['id'] = str(row[2])
                indicator_meta['date_created'] = str(row[3])
                indicator_meta['last_updated'] = str(row[4])
                indicator_meta['description'] = row[5]
                indicator_meta['source'] = "KHIS"
                indic_meta_list.append(indicator_meta)

                orgunit_meta = {}
                orgunit_meta['name']=row[6]
                orgunit_meta['id']=str(row[7])
                org_meta_list.append(orgunit_meta)

                is_indict_meta_initialized = True

            indic_frame_values['startdate'].append(row[0])
            indic_frame_values['kpivalue'].append(row[1])

            if row[2] in indic_payload_values:
                indict_list=indic_payload_values[row[2]]
                indic_val = {"date": str(row[0]), "value": str(row[1]), "ouid":  str(row[7])}
                indict_list.append(indic_val)
            else:
                indic_payload_values[row[2]] = []
                indict_list = indic_payload_values[row[2]]
                indic_val = {"date": str(row[0]), "value": str(row[1]), "ouid": str(row[7])}
                indict_list.append(indic_val)

            if (row[0] != None):
                self.max_indicator_dates[indicatorid] = row[0]
        indicator_df = pd.DataFrame(indic_frame_values)

        return (indicator_df,indic_payload_values,indic_meta_list,org_meta_list)


    def get_weather_by_year(self,ouid,weather_id=None):
        weather_rep = "in (1, 2, 3, 5)"
        if weather_id is not None:
            weather_rep ="in (%s)" %weather_id

        query_string = '''select distinct  TO_DATE(year || '-' || month  || '-' || 01, 'YYYY-MM-DD') as startdate, weather_type, ROUND(value,3),w_type_id, org_id from
                          (select avg(value) as value,EXTRACT(month from period) as month,EXTRACT(year from period) as year, org_id,weather_type,w_type_id
                          from weather_comm_org_unit where org_id =%s and period>='%s' and w_type_id %s 
                          group by org_id,w_type_id,EXTRACT(month from period),EXTRACT(year from period),weather_type) as weather order by startdate asc ''' % (
                          ouid, str(self.begin_year) + "-01-01" ,weather_rep)
        data_list = []
        data_list_above_indicator_last = []
        columns = ['startdate', 'dew_point', 'humidity', 'temperature', 'pressure']
        cursor = self._db.get_db_con()[1]
        log.info(query_string)
        cursor.execute(query_string)
        rows = cursor.fetchall()
        weather_data = {} #date: ['date','dew point','humidity','temperature','pressure']
        data_above_indicator_last = {} #date: ['date','dew point','humidity','temperature','pressure']
        weather_payload = {}
        colmns_indexes= {'dew point': 1, 'humidity': 2, 'temperature': 3, 'pressure': 4}
        weather_meta = {}

        end_date = str(self.end_year)+"-12-31"
        for row in rows:
            self.max_weather_dates[int(row[3])] = row[0]
            if row[0] < self.max_indicator_dates[self._indicator_id]: #segment data in same range with indicator last data date and  afterwards
                if row[3] not in self.variables:
                    self.variables[str(row[3])]=str(row[1])
                weather_meta[row[3]] =row[1]
                if(row[3] in weather_payload):
                    weather_val = {"date": str(row[0]), "value": str(row[2]), "ouid": str(row[4])}
                    weather_payload[row[3]].append(weather_val)
                else:
                    weather_payload[row[3]]=[]
                    weather_val = {"date": str(row[0]), "value": str(row[2]),  "ouid": str(row[4])}
                    weather_payload[row[3]].append(weather_val)


                start_dt = row[0]
                if row[1] == 'precipitation': # we do not analyse precipitation as all data reads 0 in db
                    continue
                if(start_dt in weather_data):
                    weather_data[start_dt][colmns_indexes[row[1]]] = row[2]
                    weather_data[start_dt][0] = start_dt
                else:
                    weather_data[start_dt] = [None,None,None,None,None]
                    weather_data[start_dt][colmns_indexes[row[1]]] = row[2]
                    weather_data[start_dt][0] = start_dt
            else:
                start_dt = row[0]
                #self.max_weather_dates[int(row[3])] = start_dt
                if row[1] == 'precipitation':
                    continue
                if (start_dt in data_above_indicator_last):
                    data_above_indicator_last[start_dt][colmns_indexes[row[1]]] = row[2]
                    data_above_indicator_last[start_dt][0] = start_dt
                else:
                    data_above_indicator_last[start_dt] = [None, None, None, None, None]
                    data_above_indicator_last[start_dt][colmns_indexes[row[1]]] = row[2]
                    data_above_indicator_last[start_dt][0] = start_dt


        for key in weather_data:
            data_list.append(weather_data[key])

        for key in data_above_indicator_last:
            data_list_above_indicator_last.append(data_above_indicator_last[key])

        # Create the pandas DataFrame
        weather_df = pd.DataFrame(data_list, columns=columns)
        data_list_above_indicator_last_df =pd.DataFrame(data_list_above_indicator_last, columns=columns )
        # log.info(weather_df.head())
        return (weather_df,weather_payload,weather_meta,data_list_above_indicator_last_df)


    def run_correlation(self,indicator_id,orgunit_id):
        self._indicator_id=indicator_id
        self.set_max_min_period(orgunit_id,indicator_id)

        indic_data = self.get_indicator_data(orgunit_id, indicator_id)
        indicator_df = indic_data[0]

        weather_data = self.get_weather_by_year(orgunit_id,weather_id=None)
        weather_df = weather_data[0]

        weather_df['startdate'] = weather_df['startdate'].astype('datetime64[ns]')
        indicator_df['startdate'] = indicator_df['startdate'].astype('datetime64[ns]')

        indicator_df = indicator_df.set_index('startdate') # make startdate index to allow concatination axes reference
        weather_df = weather_df.set_index('startdate') # make startdate index to allow concatination axes reference
        final_df = pd.concat([indicator_df, weather_df], axis=1, sort=False)

        final_df['kpivalue'] = final_df['kpivalue'].astype('float64')
        final_df['dew_point'] = final_df['dew_point'].astype('float64')
        final_df['humidity'] = final_df['humidity'].astype('float64')
        final_df['temperature'] = final_df['temperature'].astype('float64')
        final_df['pressure'] = final_df['pressure'].astype('float64')

        #analyses metadata
        period_span = {"start_date":  str(self.begin_year), "end_date": str(self.end_year)}
        correlation_dimension = ['kpivalue','dew_point','humidity','temperature','pressure']


        correlation_payload = {'kpivalue': None,'dew_point': None,'humidity': None,'temperature': None,'pressure': None }
        count = 1
        for index, row in final_df.corr().iterrows():
            vals = str(round(row['kpivalue'], 2)), str(round(row['dew_point'], 2)), str(round(row['humidity'], 2)), str(round(row['temperature'], 2)), str(round(row['pressure'], 2))

            if count==1:
                correlation_payload['kpivalue']=vals
            elif count==2:
                correlation_payload['dew_point']=vals
            elif count==3:
                correlation_payload['humidity']=vals
            elif count==4:
                correlation_payload['temperature']=vals
            elif count==5:
                correlation_payload['pressure']= vals

            count+=1

        self._db.close_db_con()

        #assemble result
        dictionary = {
            "analyses": {
                "period_span": period_span,
                "correlation_coeffient":  "Pearson's correlation",
                "period_type": "monthly",
                "variables": self.variables,
                "correlation_dimension": correlation_dimension
            },
            "orgunits": indic_data[3],
            "indicators": indic_data[2],
            "weather": weather_data[2]
        }
        data = {
            "weather": weather_data[1],
            "indicator": indic_data[1],
            "correlation": correlation_payload
        }
        result = {"dictionary": dictionary, "data": data}

        return result


    '''
        projects missing values from given series data
    '''
    def fill_missing_vals(self,series, leap):
        series=series.reset_index()
        m = Prophet()
        series.columns = ['ds','y']
        m.fit(series)
        future = m.make_future_dataframe(periods=leap, freq='M',include_history = False)
        forecast = m.predict(future)
        return forecast


    '''
        forecast weather data needed for predicition if need be
    '''
    def _forecast_independent_variable(self,last_weather_forecast_diff,wether_df,weather_id,last_date_forecast):

        if last_weather_forecast_diff>0: # forecast weather data if date of last predict higher than available data
            forecasted_indepn_var = self.fill_missing_vals(wether_df,last_weather_forecast_diff)
            predicted_values_to_append = forecasted_indepn_var.yhat  # actual forecasted values

            predicted_values_to_append = predicted_values_to_append.to_frame()  # to dataframe

            predicted_values_to_append.insert(0, "date",
                                              pd.date_range(start=self.max_weather_dates[weather_id] + relativedelta(months=+1),
                                                            periods=last_weather_forecast_diff,
                                                            freq='MS'))  # generate new time periods for forecasted data.
            wether_df = wether_df.reset_index()
            wether_df.columns = ['date', 'value']
            wether_df=wether_df.set_index('date')
            predicted_values_to_append.columns = ['date', 'value']
            predicted_values_to_append=predicted_values_to_append.set_index('date')
            concatnated_df = pd.concat([wether_df, predicted_values_to_append])
            concatnated_df=concatnated_df.reset_index()
            concatnated_df=concatnated_df.set_index('date', drop=True)
            concatnated_df.index = pd.DatetimeIndex(concatnated_df.index)  # make index datetime
            return concatnated_df
        else:
            log.info("truncating excess data ========>")
            wether_df=wether_df.truncate(after=last_date_forecast) #if data has more data that last indicator date data available, truncante
            log.info("truncanted data %s " % wether_df.tail())

            wether_df = wether_df.reset_index()
            wether_df.columns = ['date', 'value']

            wether_df = wether_df.set_index('date', drop=True)
            return wether_df


    '''
       implementation of multivariate prediction analysis.
       1. Generally gets weather and indicator data from storage. Formats into Dataframes and merges(left join, left on indicator data) along the periods(dates).
       2. Does forecasting on the independent variable(weather) upto the requeted time range.
       3. Greps data from the forecast data from last indicator period to end of forecast date request. This data is used to regress with the dependent variable to get corresponding
          data points
    '''
    def run_var_prediction(self, indicator_id,orgunit_id, weather_id, time_range):
        wther_dict = {2: 'dew_point', 3: 'humidity',1: 'temperature',5: 'pressure'}
        self.set_max_min_period(orgunit_id, indicator_id)
        indic_data = self.get_indicator_data(orgunit_id, indicator_id)
        weather_data = self.get_weather_by_year(orgunit_id,weather_id=weather_id)

        #inner join weather and indicator dataframes to allow even number of elements for multivariate regression
        a = indic_data[0]
        b = weather_data[0]

        concat_wther_indic = pd.merge(a, b, on='startdate' ,how='left')
        concat_wther_indic=concat_wther_indic.set_index('startdate', drop=True)
        harmonized_indic_df = concat_wther_indic['kpivalue']
        harmonized_indic_df.columns = ['startdate','kpivalue']
        harmonized_indic_df=harmonized_indic_df.reset_index()

        harmonized_indic_df_lst = list(indic_data)
        harmonized_indic_df_lst[0] = harmonized_indic_df
        indic_data = tuple(harmonized_indic_df_lst)

        #--------------------------------->>> now for weather df put it back
        new_weather_df = concat_wther_indic.drop(['kpivalue'], axis=1)

        temp_new_weather_list = list(weather_data)
        new_weather_df=new_weather_df.reset_index()
        temp_new_weather_list[0] = new_weather_df
        weather_data = tuple(temp_new_weather_list)

        columns_to_drop = [wther_condition for wther_condition in ['dew_point', 'humidity', 'temperature', 'pressure'] if wther_condition!=wther_dict[weather_id]]
        wther_upto_indicator_date = weather_data[0]
        wther_aftr_indicator_available=weather_data[3]

        wther_upto_indicator_date=wther_upto_indicator_date.drop(columns=columns_to_drop)
        wther_aftr_indicator_available=wther_aftr_indicator_available.drop(columns=columns_to_drop)

        wther_upto_indicator_date=wther_upto_indicator_date.set_index('startdate',drop=True)
        wther_aftr_indicator_available=wther_aftr_indicator_available.set_index('startdate',drop=True)

        weather_df_to_concant = []
        weather_df_to_concant.append(wther_upto_indicator_date)
        weather_df_to_concant.append(wther_aftr_indicator_available)
        wether_df = pd.concat(weather_df_to_concant)
        wether_df=wether_df.fillna(wether_df.mean())
        last_date_forecast = self.max_indicator_dates[indicator_id] + relativedelta(months=+time_range)

        last_weather_forecast_diff =months_between(self.max_weather_dates[weather_id],last_date_forecast)
        log.info("max weather data date vs max forecast date =========>")
        log.info("forecast %s" %last_date_forecast)
        log.info("weather %s" %self.max_weather_dates[weather_id])
        log.info("diff (-) %s" %last_weather_forecast_diff)

        forecasted_indepn_var=self._forecast_independent_variable(last_weather_forecast_diff,wether_df,weather_id,last_date_forecast)
        forecasted_indepn_var = forecasted_indepn_var[~forecasted_indepn_var.index.duplicated(keep='last')] #remove duplicates indecies if any as a result of earlier merge

        forecasted_indepn_var = forecasted_indepn_var.reset_index()
        forecasted_indepn_var=forecasted_indepn_var.fillna(forecasted_indepn_var.mean())

        forecasted_indepn_var['date'] = pd.to_datetime(forecasted_indepn_var['date']).dt.date #cast column to date type

        forcast_data_range = forecasted_indepn_var[(forecasted_indepn_var['date'] > self.max_indicator_dates[indicator_id]) & (forecasted_indepn_var['date'] <= last_date_forecast)]

        forecasted_indepn_var = forecasted_indepn_var.set_index("date", drop=True)

        indepent_var_trains = forecasted_indepn_var.truncate(after=self.max_indicator_dates[indicator_id])  # weather data to use for fitting

        indepent_var_trains = numpy.array(indepent_var_trains.values).astype(numpy.float)
        indepent_var_trains=numpy.round(indepent_var_trains,3)
        indepent_var_trains = numpy.reshape(indepent_var_trains, (-1, 1))

        forcast_data_range = numpy.array(forcast_data_range['value']).astype(float)
        forcast_data_range=numpy.round(forcast_data_range,3)

        forcast_data_range = numpy.reshape(forcast_data_range, (-1, 1))

        indicator_dframe = indic_data[0].fillna(indic_data[0].mean())
        X = indepent_var_trains
        indicator_dframe=indicator_dframe['kpivalue'].astype(float)
        Y = numpy.array(indicator_dframe.values) # value to be forecasted - multivariate
        regr = linear_model.LinearRegression()

        regr.fit(X, Y)
        forecast_data = regr.predict(forcast_data_range)
        df_forecast = pd.DataFrame(forecast_data)
        df_forecast.columns=['kpivalue']
        return (df_forecast, indic_data)


    def do_multivariate_prediction(self, indicator_id,orgunit_id, weather_id, time_range):
        time_range = int(time_range)
        weather_id= int(weather_id)
        self._indicator_id = indicator_id
        var_data = self.run_var_prediction(indicator_id,orgunit_id, weather_id, time_range)
        var_dat=var_data[1][0].set_index("startdate")
        # end_forecast_date = var_dat.index[-1] + datetime.timedelta(days=time_range)
        start_forecast_date = var_dat.index[-1] + datetime.timedelta(days=1)
        var_data[0].insert(0, "date", pd.date_range(start=start_forecast_date, periods=time_range, freq='MS'))
        predicted_dataframe = var_data[0].set_index("date")
        final_data = pd.concat([var_data[1][0], predicted_dataframe])
        period_span = {"start_date": str(self.begin_year), "end_date": str(self.end_year)}

        forecast_payload_values = {}
        forecast_payload_values['kpivalue'] = []

        for items in predicted_dataframe['kpivalue'].iteritems():
            indict_list = forecast_payload_values['kpivalue']
            indic_val = {"date": str(items[0].date()), "value": str(round(items[1], 2)), "ouid": str(orgunit_id)}
            indict_list.append(indic_val)

        # assemble result
        dictionary = {
            "analyses": {
                "period_span": period_span,
                "precition_model": "Vector Autoregression",
                "period_type": "monthly",
                "variables": self.variables
            },
            "orgunits": var_data[1][3],
            "indicators": var_data[1][2]
        }
        data = {
            "indicator": var_data[1][1],
            "forecast_values": forecast_payload_values
        }
        result = {"dictionary": dictionary, "data": data}
        return result

# r=WeatherCorrelation()
# r.do_multivariate_prediction(82257,23401,3,5)