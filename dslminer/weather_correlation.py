import logging
import json
from builtins import int

import pandas as pd
import db
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
import datetime
import math

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
        print("maximum dates =======>")
        print(self.max_indicator_dates)
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
            if row[0] < self.max_indicator_dates[self._indicator_id]:
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
                if row[1] == 'precipitation':
                    continue
                if (start_dt in data_above_indicator_last):
                    data_above_indicator_last[start_dt][colmns_indexes[row[1]]-1] = row[2]
                else:
                    data_above_indicator_last[start_dt] = [None, None, None, None]
                    data_above_indicator_last[start_dt][colmns_indexes[row[1]]-1] = row[2]


        for key in weather_data:
            data_list.append(weather_data[key])

        # Create the pandas DataFrame
        weather_df = pd.DataFrame(data_list, columns=columns)
        # log.info(weather_df.head())
        print("returning")
        print(data_above_indicator_last)
        return (weather_df,weather_payload,weather_meta,data_above_indicator_last)


    def run_correlation(self,indicator_id,orgunit_id):
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
            Makes given series (df_train) stationary to enable for timeseries analysis.
        '''

    def make_series_stationary(self, df_train):
        # ADF Test on each column
        for name, column in df_train.iteritems():
            is_stationary = self.adfuller_test(column, name=column.name)
            if is_stationary == 0 and self.stationarity_iterations_count != 4:

                df_train = df_train.diff().dropna()
                self.stationarity_iterations_count = self.stationarity_iterations_count + 1
                self.make_series_stationary(df_train)
        return df_train

    '''
       implementation of Vector autoregression model for multivariate prediction analysis.
    '''

    def run_var_prediction(self, indicator_id,orgunit_id, weather_id, time_range):

        self.set_max_min_period(orgunit_id, indicator_id)
        indic_data = self.get_indicator_data(orgunit_id, indicator_id)
        weather_data = self.get_weather_by_year(orgunit_id,weather_id=weather_id)
        weather_df = weather_data[0]

        indicator_df = indic_data[0]
        indicator_df = indicator_df.fillna(indicator_df.mean())  # fill NaN with averages.

        weather_df['startdate'] = weather_df['startdate'].astype('datetime64[ns]')
        indicator_df['startdate'] = indicator_df['startdate'].astype('datetime64[ns]')
        indicator_df = indicator_df.set_index('startdate')  # make startdate index to allow concatination axes reference
        weather_df = weather_df.set_index('startdate')  # make startdate index to allow concatination axes reference
        final_df = pd.concat([indicator_df, weather_df], axis=1, sort=False)
        final_df = final_df.fillna(final_df.mean())  # fill NaN with averages.
        final_df['kpivalue'] = final_df['kpivalue'].astype('float64')


        if (weather_id == 2):
            final_df['dew_point'] = final_df['dew_point'].astype('float64')
            final_df=final_df.drop(columns=[ 'humidity', 'temperature', 'pressure'])
        if (weather_id == 3):
           final_df['humidity'] = final_df['humidity'].astype('float64')
           final_df=final_df.drop(columns=['dew_point', 'temperature', 'pressure'])
        if (weather_id == 1):
            final_df['temperature'] = final_df['temperature'].astype('float64')
            final_df=final_df.drop(columns=['dew_point', 'humidity', 'pressure'])
        if (weather_id == 5):
            final_df['pressure'] = final_df['pressure'].astype('float64')
            final_df=final_df.drop(columns=['dew_point', 'humidity', 'temperature'])


        # we split the data frame to training and test series
        nobs = 4
        df_train, df_test = final_df[0:-nobs], final_df[-nobs:]

        # make the series stationary

        df_differenced = self.make_series_stationary(df_train)
        model = VAR(df_differenced)
        # fit model
        model_fitted = model.fit()

        # Get the lag order
        lag_order = model_fitted.k_ar

        # Input data for forecasting
        forecast_input = df_differenced.values[-lag_order:]

        # Forecast
        forecast_data = model_fitted.forecast(y=forecast_input, steps=time_range)

        df_forecast = pd.DataFrame(forecast_data, columns=final_df.columns)
        predicted_results = self.invert_transformation(df_train, df_forecast)
        return (predicted_results, indic_data)

    def invert_transformation(self, df_train, df_forecast):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:
            # Roll back Diff value
            if self.stationarity_iterations_count == 1:
                # Roll back 1st Diff
                df_fc[str(col)] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()
            elif self.stationarity_iterations_count > 1:
                for i in range(self.stationarity_iterations_count, 1, -1):
                    df_fc[str(col)] = (df_train[col].iloc[-i] - df_train[col].iloc[-(i + 1)]) + df_fc[str(col)].cumsum()
                    if i == 1:
                        df_fc[str(col)] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()

        return df_fc

    def adfuller_test(self, series, signif=0.05, name='', verbose=False):
        """Perform ADFuller to test for Stationarity of given series and print report"""
        r = adfuller(series, autolag='AIC')
        output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
        p_value = output['pvalue']

        def adjust(val, length=6):
            return str(val).ljust(length)

        # Print Summary
        print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-' * 47)
        print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f' Significance Level    = {signif}')
        print(f' Test Statistic        = {output["test_statistic"]}')
        print(f' No. Lags Chosen       = {output["n_lags"]}')

        for key, val in r[4].items():
            print(f' Critical value {adjust(key)} = {round(val, 3)}')

        if p_value <= signif:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary.")
            return 0
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary.")
            return 1

    def do_multivariate_prediction(self, indicator_id,orgunit_id, weather_id, time_range):
        time_range = int(time_range)
        weather_id= int(weather_id)
        self._indicator_id = indicator_id
        # indic_data = final_df,indic_payload_values,indic_meta_list,org_meta_list
        # var_data = predicted_results,indic_data
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

r=WeatherCorrelation()
r.do_multivariate_prediction(23185,23408,2,10)