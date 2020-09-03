import logging
import requests
import json
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import db
from datetime import datetime


# configurations
log = logging.getLogger("weather correlation")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)


class MultiRegression:
    def __init__(self):
        self.begin_year=2010
        self.end_year=2019
        self._db = db.database()
        self.cadres=[]

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
                    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"='''+str(indictor_id)+''' and "Org unit id"='''+str(orgunit_id)

        cursor = self._db.get_db_con()[1]
        print(max_min_period)
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
        connection = self._db.get_db_con()[0]
        # cursor = _db.get_db_con()[1]
        query_string = '''SELECT  distinct startdate, kpivalue
                    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"=%s and "Org unit id"=%s and startdate>='%s' and enddate<='%s' order by startdate asc''' % (
        indicatorid,ouid,str(self.begin_year)+"-01-01", str(self.end_year)+"-12-31" )
        pd_resultset = pd.read_sql_query(query_string, connection)
        indicator_df = pd.DataFrame(pd_resultset)
        # log.info(indicator_df.head())
        return indicator_df


    def get_weather_by_year(self,ouid):
        query_string = '''select year || '-' || month  || '-' || 01 as startdate, weather_type, ROUND(value,3) from
                          (select avg(value) as value,EXTRACT(month from period) as month,EXTRACT(year from period) as year, org_id,weather_type
                          from weather_comm_org_unit where org_id =%s and period>='%s' and period<='%s' and w_type_id in(1,2,3,5,6)
                          group by org_id,EXTRACT(month from period),EXTRACT(year from period),weather_type) as weather order by startdate asc ''' % (
                          ouid, str(self.begin_year) + "-01-01", str(self.end_year) + "-12-31")
        data_list = []
        columns = ['date', 'precipitation', 'dew_point', 'humidity', 'temperature', 'pressure']
        cursor = self._db.get_db_con()[1]
        cursor.execute(query_string)
        rows = cursor.fetchall()
        weather_data = {} #date: ['date','precipitation','dew point','humidity','temperature','pressure']
        colmns_indexes= {'precipitation': 1, 'dew point': 2, 'humidity': 3, 'temperature': 4, 'pressure': 5}
        for row in rows:
            start_dt = row[0]
            if(start_dt in weather_data):
                weather_data[start_dt][colmns_indexes[row[1]]] = row[2]
                weather_data[start_dt][0] = start_dt
            else:
                weather_data[start_dt] = [None,None,None,None,None,None]
                weather_data[start_dt][colmns_indexes[row[1]]] = row[2]
                weather_data[start_dt][0] = start_dt

        for key in weather_data:
            data_list.append(weather_data[key])

        # Create the pandas DataFrame
        weather_df = pd.DataFrame(data_list, columns=columns)
        # log.info(weather_df.head())

        return weather_df

    def run_regression(self,orgunit_id,indicator_id):
        self.set_max_min_period(orgunit_id,indicator_id)
        indicator_df = self.get_indicator_data(orgunit_id, indicator_id)
        weather_df = self.get_weather_by_year(orgunit_id)

        weather_df['date'] = weather_df['date'].astype('datetime64[ns]')
        indicator_df['startdate'] = indicator_df['startdate'].astype('datetime64[ns]')

        indicator_df = indicator_df.set_index('startdate') # make startdate index to allow concatination axes reference
        weather_df = weather_df.set_index('date') # make startdate index to allow concatination axes reference

        final_df = pd.concat([indicator_df, weather_df], axis=1, sort=False)

        final_df['precipitation'] = final_df['precipitation'].astype('float64')
        final_df['dew_point'] = final_df['dew_point'].astype('float64')
        final_df['humidity'] = final_df['humidity'].astype('float64')
        final_df['temperature'] = final_df['temperature'].astype('float64')
        final_df['pressure'] = final_df['pressure'].astype('float64')

        log.info(final_df.dtypes)
        log.info("=========....")
        log.info(final_df.head())
        log.info("=========....")
        log.info(final_df.corr())
        # self.run_model(final_df,cadre_condition_list)
        self._db.close_db_con()

r=MultiRegression()
r.run_regression(23408,93333)