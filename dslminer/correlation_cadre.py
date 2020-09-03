import requests
import logging
import json
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import db
from datetime import datetime


# configurations
log = logging.getLogger("cadre correlation")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)

BASE_URL="http://dsl.health.go.ke/dsl/api/"

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
        cursor.execute(max_min_period)
        row = cursor.fetchall()
        log.info("============================")
        log.info(max_min_period)
        if (len(row) != 0):
            log.info("============================")
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
        indicator_df = pd.DataFrame(pd_resultset)
        # log.info(indicator_df.head())
        return indicator_df


    def get_cadres_by_year(self,orgunit,cadreid_list):
        data ={ "startdate": [] }
        date_control=0
        for cadre in cadreid_list:
            for year in range(self.begin_year, self.end_year):
                log.info(cadreid_list)
                req_url=BASE_URL + 'cadres?pe=' + str(year) + '&ouid=' + str(orgunit) + '&id=' + str(cadre) + '&periodtype=monthly'
                req = requests.get(req_url)
                if("Error Code" in req.text):
                    continue
                cadre_allocation=json.loads(req.text)
                log.debug(cadre_allocation)
                for cadre_alloc in cadre_allocation:
                    if(date_control == 0):
                        year=cadre_alloc['period'][:4]
                        month = cadre_alloc['period'][4:]
                        final_period=year+"-"+month+"-1"
                        cadre_date = datetime.strptime(final_period, '%Y-%m-%d').date()
                        data["startdate"].append(cadre_date)
                    if(cadre_alloc['cadre'] in data.keys()):
                        data[cadre_alloc['cadre']].append(int(cadre_alloc['cadreCount']))
                    else:
                        data[cadre_alloc['cadre']]=[]
                        self.cadres.append(cadre_alloc['cadre'].replace("^ ","").replace(" $",""))
                        data[cadre_alloc['cadre']].append(int(cadre_alloc['cadreCount']))
            date_control=1

        # pd.DataFrame(data)
        cadre_series=[]
        for x in range(len(self.cadres)):
            log.info(type(pd.Series(data[self.cadres[x]], name=self.cadres[x])))
            cadre_series.append(pd.Series(data[self.cadres[x]], name=self.cadres[x]))

        cadre_series.append(pd.Series(data["startdate"], name="startdate"))
        cadres_df = pd.concat([v for v in cadre_series], axis=1)
        cadre_alloc_pd=cadres_df
        return cadre_alloc_pd


    def run_model(self,data_frame,cadre_condition_list):
        # Data preprocessing/data cleaning
        # look the missing values (NaN)
        for key in self.cadres:
            median_cadre_value = data_frame[key].mean()
            cadre_fillna=data_frame[key]
            cadre_fillna.fillna(median_cadre_value,inplace=True)
            # data_frame.cadre_value = data_frame[key].fillna(median_cadre_value)

        median_kpivalue = data_frame['kpivalue'].mean()
        data_frame.kpivalue = data_frame.kpivalue.fillna(median_kpivalue)

        # data_frame=data_frame.astype({'cadre_value': 'int32'})
        log.info("=================> correlation scoring <=================")
        log.info(data_frame.corr())
        log.info("=================> correlation scoring <=================")
        # log.info(data_frame.isnull().sum())
        # data_frame.plot(kind='scatter', x='kpivalue', y='cadre_value', title='kpivalue vs cadre_value');
        # data_frame['cadre_value'].plot(kind='hist');
        # plt.show()
        # Train the model Linear Regression
        reg = linear_model.LinearRegression()
        reg.fit(data_frame[self.cadres], data_frame.kpivalue)
        prediction=str('%.2f' % reg.predict([cadre_condition_list])[0])
        log.info("=================> predicted value <=================")  # two decimal places
        log.info(prediction) # two decimal places
        log.info("=================> predicted value <=================")  # two decimal places
        return prediction


    def run_regression(self,orgunit_id,indicator_id,cadre_list,cadre_condition_list):
        self.set_max_min_period(orgunit_id,indicator_id)
        indicator_df = self.get_indicator_data(orgunit_id, indicator_id)
        cadres_df = self.get_cadres_by_year(orgunit_id,cadre_list)
        indicator_df = indicator_df.set_index('startdate') # make startdate index to allow concatination axes reference
        cadres_df = cadres_df.set_index('startdate') # make startdate index to allow concatination axes reference

        final_df = pd.concat([indicator_df, cadres_df], axis=1, sort=False)
        self.run_model(final_df,cadre_condition_list)
        self._db.close_db_con()

# r=MultiRegression()
# r.run_regression(23519,61901,[33,30,31,32],[2,20,10,10])