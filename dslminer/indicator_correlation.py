import logging
import pandas as pd
from . import db
import datetime
from sklearn import linear_model
from neuralprophet import NeuralProphet
from dateutil.relativedelta import *
import numpy

from . import utils
diff_month = utils.diff_month

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
            if (int(row[0][1] < 2010)):
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
        max_indicator_dates = {}
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
            if(row[0]!=None):
                max_indicator_dates[indicatr_id]= row[0]

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
        return (final_df,indic_payload_values,indic_meta_list,org_meta_list,max_indicator_dates)


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


    def forecast_independent_variable(self,key,indicator_id,time_range,max_indicator_dates,indicator_df):
            indicator_id = int(indicator_id)
            time_diff=diff_month(max_indicator_dates[indicator_id] , max_indicator_dates[key]) # diff between last data in independent variable (indicaotor) and the predicted variable(indicator)

            time_diff=time_range+time_diff # to total number of values to be forecasted and filled to the independent variable
            df_colmn = '%s' %key

            indicator_data_to_fill = indicator_df[df_colmn].truncate(after=max_indicator_dates[key]).to_frame().reset_index() # remove null indcies starting from last date with value to current

            predicted_indicator_data = self.fill_missing_vals(indicator_data_to_fill, time_diff) # fill missing dates with data and also add forecasted with number of values to be predicted in dependent variable

            predicted_values_to_append=predicted_indicator_data.yhat #actual forecasted values

            predicted_values_to_append = predicted_values_to_append.to_frame() #to dataframe
            predicted_values_to_append.insert(0, "date", pd.date_range(start=max_indicator_dates[key]+relativedelta(months=+1), periods=time_diff,
                                                        freq='MS'))  # generate new time periods for forecasted data.
            indicator_data_to_fill.columns = ['date', 'value']
            indicator_data_to_fill.set_index('date')
            predicted_values_to_append.columns = ['date', 'value']
            predicted_values_to_append.set_index('date')
            concatnated_df = pd.concat([indicator_data_to_fill, predicted_values_to_append])
            concatnated_df.set_index('date',drop=True,inplace=True)
            concatnated_df.index = pd.DatetimeIndex(concatnated_df.index) #make index datetime
            return concatnated_df


    '''
       implementation of model for multivariate prediction analysis.
    '''
    def run_var_prediction(self,indicator_id,orgunit_id,compare_indicators,time_range):

        parameter_indicators = str(indicator_id) + ',' + compare_indicators
        self.set_max_min_period(orgunit_id, indicator_id)
        indic_data = self.get_indicator_data(orgunit_id, parameter_indicators)
        indicator_df = indic_data[0]
        max_indicator_dates = indic_data[4]

        indipendent_variable =[]
        for indic_id in max_indicator_dates:
            if (indic_id != indicator_id):
                forecasted_indepn_var=self.forecast_independent_variable(indic_id, indicator_id, time_range, max_indicator_dates, indicator_df)
                forecasted_indepn_var=forecasted_indepn_var.rename(columns={'value': indic_id})
                # forecasted_indepn_var.rename(columns={'value': indic_id}, inplace=True)
                indipendent_variable.append(forecasted_indepn_var)

        indicator_df.fillna(indicator_df.mean(),inplace=True)  # fill NaN with averages.
        indicator_colmn = '%s' % indicator_id
        indicator_df[indicator_colmn].index = pd.DatetimeIndex(indicator_df[indicator_colmn].index)

        independent_variables_vals = []
        independent_predicted_vals = []  # we  create a 2d array to hold values for the X independent to be used for against correlation model to do predcition of Y variable
        independ_correlation_count = len(indipendent_variable[0])-time_range # control to put values for fitting and for X correlation prediction separate
        correlation_arr_counter = 0

        for x in range(len(indipendent_variable[0])):
            if independ_correlation_count!=0:
                for data_frame_no in range(len(indipendent_variable)):
                    if len(independent_variables_vals) != len(indipendent_variable): # create list for each indicators data
                        independent_variables_vals.append([])

                    independent_variables_vals[data_frame_no].append(indipendent_variable[data_frame_no].values[x][0])
                independ_correlation_count = independ_correlation_count - 1
            else:
                independent_predicted_vals.append([])
                for data_frame_no in range(len(indipendent_variable)):
                    independent_predicted_vals[correlation_arr_counter].append(indipendent_variable[data_frame_no].values[x][0])
                correlation_arr_counter = correlation_arr_counter + 1

        strt_date = str(self.begin_year) + "-01-01"
        indep_dataframes = []

        for dat_frame in independent_variables_vals:
            arr = numpy.array(dat_frame)
            indep_dataframe=pd.DataFrame(arr)
            indicator_dframe = indicator_df[indicator_colmn]
            indicator_dframe=indicator_dframe.truncate(after=max_indicator_dates[indicator_id]) # remove null values if any in indicator (dependent var) to  be predicted
            #indicator_dframe.round(decimals=3)
            indep_dataframe.insert(0, "date", pd.date_range(start=strt_date, periods=len(indicator_dframe), freq='MS'))  # generate new time periods for forecasted data.
            indep_dataframe = indep_dataframe.set_index('date')
            indep_dataframe.fillna(indep_dataframe.mean(),inplace=True)
            indep_dataframes.append(indep_dataframe)

        indep_dataframes = pd.concat(indep_dataframes, axis=1, sort=False)
        X=indep_dataframes
        Y = indicator_dframe # value to be forecasted - multivariate
        regr = linear_model.LinearRegression()
        X=X.values
        Y=numpy.array(Y.values)
        regr.fit(X,Y)
        forecast_data=regr.predict(independent_predicted_vals)
        #, columns=indicator_df.columns
        df_forecast = pd.DataFrame(forecast_data)

        return (df_forecast,indic_data,indipendent_variable,max_indicator_dates)


    '''
        projects missing values from given series data
    '''
    def fill_missing_vals(self,series, leap):
        m = NeuralProphet()
        series.columns = ['ds','y']
        m.fit(series)
        future = m.make_future_dataframe(periods=leap, freq='M',include_history = False)
        forecast = m.predict(future)
        return forecast


    def do_multivariate_prediction(self,indicator_id,orgunit_id,compare_indicators,time_range):
        time_range = int(time_range)
        indicator_id =int(indicator_id)
        var_data = self.run_var_prediction(indicator_id,orgunit_id,compare_indicators,time_range)
        start_forecast_date = var_data[1][0].index[-1] + datetime.timedelta(days=1)

        forecast_payload_values = {}
        for independent_var_df in var_data[2]:
            max_indicator_dates = var_data[3]

            indepn_indic_id = independent_var_df.columns[0]
            independent_var_df=independent_var_df.round(decimals=2)
            forecast_payload_values[str(indepn_indic_id)] =[]

            for index, row  in independent_var_df[max_indicator_dates[indepn_indic_id]+relativedelta(months=+1):].iterrows():
                indict_list = forecast_payload_values[str(indepn_indic_id)]
                date_time = index.strftime("%Y-%m-%d")
                indic_val = {"date": date_time, "value": str(list(row)[0]), "ouid": str(orgunit_id)}
                indict_list.append(indic_val)

        var_data[0].insert(0, "date", pd.date_range(start = start_forecast_date, periods=time_range, freq= 'MS')) # generate new time periods for forecast data.
        predicted_dataframe = var_data[0].set_index("date")
        period_span = {"start_date": str(self.begin_year), "end_date": str(self.end_year)}
        indicator_column = '%s' %indicator_id


        forecast_payload_values[indicator_column] = []

        for index, row in predicted_dataframe.iterrows():
            indict_list = forecast_payload_values[indicator_column]
            date_time = index.strftime("%Y-%m-%d")
            indic_val = {"date": date_time, "value": str(round(row[0], 2)), "ouid": str(orgunit_id)}
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
# r.do_multivariate_prediction(32956,18,'32413',30) # 21030, '23191,23701,31589'
# r.run_var_prediction(21030,23408,'93299,23700,23191,31584',30) #,2449433,|93299,
