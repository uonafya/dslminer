import requests
import logging
import json
import numpy as np
import pandas as pd
from sklearn import linear_model
from . import db
from datetime import datetime


# configurations
log = logging.getLogger("dataloader")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)

BASE_URL="http://dsl.health.go.ke/dsl/api/"

class MultiRegression:
    def __init__(self):
        self.begin_year=2010
        self.end_year=2019
        self._db = db.database()

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
        cursor.execute(max_min_period)
        row = cursor.fetchall()
        print("============================ 7")
        print(max_min_period)
        if (len(row) != 0):
            print("============================")
            print(row[0][0])
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
        log.info(query_string)
        pd_resultset = pd.read_sql_query(query_string, connection)
        log.info(query_string)
        indicator_df = pd.DataFrame(pd_resultset)
        log.info(indicator_df.head())
        return indicator_df


    def get_cadres_by_year(self,orgunit,cadreid_list):
        data ={ "startdate": [], "cadre_value": [] }
        for year in range(self.begin_year, self.end_year):
            for cadre in cadreid_list:
                log.info(cadreid_list)
                req_url=BASE_URL + 'cadres?pe=' + str(year) + '&ouid=' + str(orgunit) + '&id=' + str(cadre) + '&periodtype=monthly'
                req = requests.get(req_url)
                if("Error Code" in req.text):
                    continue
                cadre_allocation=json.loads(req.text)
                log.debug(cadre_allocation)
                for cadre_alloc in cadre_allocation:
                    year=cadre_alloc['period'][:4]
                    month = cadre_alloc['period'][4:]
                    final_period=year+"-"+month+"-1"
                    cadre_date = datetime.strptime(final_period, '%Y-%m-%d').date()
                    data["startdate"].append(cadre_date)
                    data['cadre_value'].append(cadre_alloc['cadreCount'])
        cadre_alloc_pd = pd.DataFrame(data)
        log.info(cadre_alloc_pd.head())
        return cadre_alloc_pd


    def run_model(self,data_frame):
        # Data preprocessing/data cleaning
        # look the missing values (NaN)
        median_cadre_value = data_frame['cadre_value'].median()
        data_frame.cadre_value = data_frame.cadre_value.fillna(median_cadre_value)

        median_kpivalue = data_frame['kpivalue'].median()
        data_frame.kpivalue = data_frame.kpivalue.fillna(median_kpivalue)

        # Train the model Linear Regression
        reg = linear_model.LinearRegression()
        reg.fit(data_frame[['cadre_value']], data_frame.kpivalue)
        log.info(reg.predict([[1]]))


    def run_regression(self,orgunit_id,indicator_id,cadre_list):
        self.set_max_min_period(orgunit_id,indicator_id)
        indicator_df = self.get_indicator_data(orgunit_id, indicator_id)
        cadres_df = self.get_cadres_by_year(orgunit_id,cadre_list)
        indicator_df = indicator_df.set_index('startdate') # make startdate index to allow concatination axes reference
        cadres_df = cadres_df.set_index('startdate') # make startdate index to allow concatination axes reference
        final_df = result = pd.concat([indicator_df, cadres_df], axis=1, sort=False)
        self.run_model(final_df)
        self._db.close_db_con()

    #run_regression(23519,61901,[33])
