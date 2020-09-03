import logging
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import db
from datetime import datetime


# configurations
log = logging.getLogger("indicator correlation")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)


class IndicatorCorrelation:
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


    def get_indicator_data(self,ouid,indicatorid,compare_indicators):
        query_string = '''SELECT  distinct startdate, kpivalue, "Indicator name" as indicator, "Organisation Unit Name" as org_unit, "Org unit id" as org_id,
                    "Indicator ID" as indicator_id
                    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID" in (%s) and "Org unit id"=%s and startdate>='%s' and enddate<='%s' order by startdate asc''' % (
        compare_indicators,ouid,str(self.begin_year)+"-01-01", str(self.end_year)+"-12-31" )
        print(query_string)
        columns = ["date", "indicator"]
        cursor = self._db.get_db_con()[1]
        cursor.execute(query_string)
        rows = cursor.fetchall()

        indicators_dt = {} # indicator_id: [[date, value],[date, value]]
        for row in rows:
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
        log.info(final_df.head(50))
        # log.info(indicator_df.head())
        return final_df


    def run_regression(self,orgunit_id,indicator_id,compare_indicators):
        compare_indicators=  str(indicator_id)+','+ compare_indicators
        self.set_max_min_period(orgunit_id,indicator_id)
        indicator_df = self.get_indicator_data(orgunit_id, indicator_id,compare_indicators)

        log.info(indicator_df.corr())
        # self.run_model(final_df,cadre_condition_list)
        self._db.close_db_con()

r=IndicatorCorrelation()
r.run_regression(23408,23185,'23191,23701,31589')