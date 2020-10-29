import logging
import matplotlib.pyplot as plt
import pandas as pd
from . import db
from datetime import datetime
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
import datetime
import math

# configurations
log = logging.getLogger("indicator correlation")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)


class IndicatorCorrelation:
    def __init__(self):
        self.begin_year=2010
        self.end_year=2019
        self._db = db.database()
        self.variables={}
        self.stationarity_iterations_count = 0

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


    def get_indicator_data(self,ouid,compare_indicators):
        query_string = '''SELECT  distinct startdate ,ROUND(kpivalue,3) , "Indicator name" as indicator , "Organisation Unit Name" as org_unit , "Org unit id" as org_id , 
                    "Indicator ID" as indicator_id ,  _datecreated , lastupdated , "Indicator description"  
                    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID" in (%s) and "Org unit id"=%s and startdate>='%s' and enddate<='%s' order by startdate asc''' % (
        compare_indicators,ouid,str(self.begin_year)+"-01-01", str(self.end_year)+"-12-31" )
        log.info(query_string)
        columns = ["date", "indicator"]
        cursor = self._db.get_db_con()[1]
        cursor.execute(query_string)
        rows = cursor.fetchall()

        indicators_dt = {} # indicator_id: [[date, value],[date, value]]
        indic_meta_list = []
        org_meta_list = []
        indic_payload_values = {}
        added_org_units = []
        added_indics = {}
        for row in rows:

            if row[2] not in added_indics: # indicator name
                self.variables[str(row[5])]=row[2]
                indicator_meta = {}
                indicator_meta['name'] = row[2]
                indicator_meta['id'] = str(row[5])
                indicator_meta['date_created'] = str(row[6])
                indicator_meta['last_updated'] = str(row[7])
                indicator_meta['description'] = row[8]
                indicator_meta['source'] = "KHIS"
                indic_meta_list.append(indicator_meta)
                added_indics[str(row[2])] = row[2]
            if row[4] not in added_org_units:
                orgunit_meta = {}
                orgunit_meta['name'] = row[3]
                orgunit_meta['id'] = str(row[4])
                org_meta_list.append(orgunit_meta)
                added_org_units.append(row[4])


            if row[5] in indic_payload_values:
                indict_list=indic_payload_values[row[5]]
                indic_val = {"date": str(row[0]), "value": str(row[1]), "ouid":  str(row[4])}
                indict_list.append(indic_val)
            else:
                indic_payload_values[row[5]] = []
                indict_list = indic_payload_values[row[5]]
                indic_val = {"date": str(row[0]), "value": str(row[1]), "ouid": str(row[4])}
                indict_list.append(indic_val)


            indicatr_id = row[5]
            if indicatr_id in indicators_dt:
                data_element =[]
                data_element.append(row[0])
                data_element.append(row[1])
                indicators_dt[indicatr_id].append(data_element)
            else:
                indicators_dt[indicatr_id] = []
                data_element = []
                data_element.append(row[0])
                data_element.append(row[1])
                indicators_dt[indicatr_id].append(data_element)

        indicators_dataframes = []
        for key in indicators_dt:
            indict = '%s' %key
            colmns = ['date',indict]
            datafrm = pd.DataFrame(indicators_dt[key], columns=colmns)
            datafrm['date'] = datafrm['date'].astype('datetime64[ns]')
            datafrm[indict] = datafrm[indict].astype('float64')
            datafrm = datafrm.set_index('date')  # make startdate index to allow concatination axes reference
            indicators_dataframes.append(datafrm)

        final_df = pd.concat(indicators_dataframes, axis=1, sort=False)
        # log.info(final_df.head(50))
        # log.info(indicator_df.head())
        return (final_df,indic_payload_values,indic_meta_list,org_meta_list)


    def run_correlation(self,indicator_id,orgunit_id,compare_indicators):
        correlation_dimension= []
        correlation_payload= {}
        compare_indicators=  str(indicator_id)+','+ compare_indicators
        self.set_max_min_period(orgunit_id,indicator_id)
        indic_data = self.get_indicator_data(orgunit_id,compare_indicators)
        indicator_df = indic_data[0]

        for col in indicator_df.columns:
            correlation_dimension.append(str(col))
            correlation_payload[str(col)] = None
        self._db.close_db_con()


        for index, row in indicator_df.corr().iterrows():
            value_list=[]
            for x in row.tolist():
                value_list.append(str(round(x, 2)))
            correlation_payload[index] = value_list

        period_span = {"start_date": str(self.begin_year), "end_date": str(self.end_year)}
        # assemble result
        dictionary = {
            "analyses": {
                "period_span": period_span,
                "correlation_coeffient": "Pearson's correlation",
                "period_type": "monthly",
                "variables": self.variables,
                "correlation_dimension": correlation_dimension
            },
            "orgunits": indic_data[3],
            "indicators": indic_data[2]
        }
        data = {
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
            # print('\n')
            if is_stationary == 0 and self.stationarity_iterations_count != 4:
                df_train = df_train.diff().dropna()
                self.stationarity_iterations_count = self.stationarity_iterations_count + 1
                self.make_series_stationary(df_train)
        return df_train

    '''
       implementation of Vector autoregression model for multivariate prediction analysis.
    '''
    def run_var_prediction(self,indicator_id,orgunit_id,compare_indicators,time_range):

        compare_indicators = str(indicator_id) + ',' + compare_indicators
        self.set_max_min_period(orgunit_id, indicator_id)
        indic_data = self.get_indicator_data(orgunit_id, compare_indicators)
        indicator_df = indic_data[0]
        indicator_df = indicator_df.fillna(indicator_df.mean()) # fill NaN with averages.

        #we split the data frame to training and test series
        nobs = 4
        df_train, df_test = indicator_df[0:-nobs], indicator_df[-nobs:]

        #make the series stationary
        df_differenced= self.make_series_stationary(df_train)
        model = VAR(df_differenced)
        # fit model
        model_fitted = model.fit()

        # Get the lag order
        lag_order = model_fitted.k_ar

        # Input data for forecasting
        forecast_input = df_differenced.values[-lag_order:]

        # Forecast
        forecast_data = model_fitted.forecast(y=forecast_input, steps=time_range)

        df_forecast = pd.DataFrame(forecast_data, columns=indicator_df.columns)
        predicted_results = self.invert_transformation(df_train, df_forecast)

        return (predicted_results,indic_data)


    def invert_transformation(self,df_train, df_forecast):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:
            # Roll back Diff value
            if self.stationarity_iterations_count==1:
                # Roll back 1st Diff
                df_fc[str(col)] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()
            elif  self.stationarity_iterations_count>1:
                for i in range(self.stationarity_iterations_count,1,-1):
                    df_fc[str(col)] = (df_train[col].iloc[-i] - df_train[col].iloc[-(i+1)]) + df_fc[str(col)].cumsum()
                    if i==1:
                        df_fc[str(col)] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()


        return df_fc

    def adfuller_test(self,series, signif=0.05, name='', verbose=False):
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

    def do_multivariate_prediction(self,indicator_id,orgunit_id,compare_indicators,time_range):
        time_range = int(time_range)
        # indic_data = final_df,indic_payload_values,indic_meta_list,org_meta_list
        # var_data = predicted_results,indic_data
        var_data = self.run_var_prediction(indicator_id,orgunit_id,compare_indicators,time_range)
        # end_forecast_date =var_data[1][0].index[-1] + datetime.timedelta(days=time_range)
        start_forecast_date = var_data[1][0].index[-1] + datetime.timedelta(days=1)
        var_data[0].insert(0, "date", pd.date_range(start = start_forecast_date, periods=time_range, freq= 'MS')) # generate new time periods for forecast data.
        predicted_dataframe = var_data[0].set_index("date")
        final_data = pd.concat([var_data[1][0], predicted_dataframe])
        period_span = {"start_date": str(self.begin_year), "end_date": str(self.end_year)}
        print("================")
        indicator_column = '%s' %indicator_id
        forecast_payload_values = {}
        forecast_payload_values[indicator_column] = []
        for items in predicted_dataframe[indicator_column].iteritems():
            indict_list = forecast_payload_values[indicator_column]
            indic_val = {"date": str(items[0].date()), "value": str(round(items[1],2)), "ouid": str(orgunit_id)}
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
            "forecast_values":forecast_payload_values
        }
        result = {"dictionary": dictionary, "data": data}
        return result


# r=IndicatorCorrelation()
# r.do_multivariate_prediction(23185,23408,'23191,23701,31589',10)
# r.run_var_prediction(93299,23408,'32954,2449433',10) #,2449433,93323