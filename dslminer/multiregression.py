import requests
import logging
import json
import psycopg2
import numpy as np
import pandas as pd
from sklearn import linear_model
import db
from datetime import datetime


# configurations
log = logging.getLogger("dataloader")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)

BASE_URL="http://dsl.health.go.ke/dsl/api/"
BEGIN_YEAR=2009
END_YEAR=2019
_db = db.database()

def set_max_min_period(indictor_id,orgunit_id):
    """Sets the begin and end period to query data
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

    cursor = _db.get_db_con()[1]
    cursor.execute(max_min_period)
    row = cursor.fetchall()
    if (len(row) != 0):
        END_YEAR=int(row[0][0])
        if (int(row[0][0] < 2009)):
            pass
        else:
            BEGIN_YEAR = int(row[0][1])
        log.info("end year "+str(row[0][0]))
        log.info("start year " + str(BEGIN_YEAR))


def get_indicator_data(indicatorid,ouid):
    connection = _db.get_db_con()[0]
    # cursor = _db.get_db_con()[1]
    query_string = '''SELECT  distinct startdate, kpivalue
                FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"=%s and "Org unit id"=%s and startdate>='%s' and enddate<='%s' order by startdate asc''' % (
    indicatorid,ouid,str(BEGIN_YEAR)+"-01-01", str(END_YEAR)+"-12-31" )
    log.info(query_string)
    pd_resultset = pd.read_sql_query(query_string, connection)
    log.info(query_string)
    indicator_df = pd.DataFrame(pd_resultset)
    log.info(indicator_df.head())
    return indicator_df


def get_cadres_by_year(cadreid_list,orgunit,begin_year,end_year):
    data ={ "startdate": [], "cadre_value": [] }
    for year in range(begin_year, end_year):
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

set_max_min_period(61901,23519)
indicator_df=get_indicator_data(61901,23519)
cadres_df=get_cadres_by_year([33],23519,BEGIN_YEAR,END_YEAR)
log.info("=============final data frame=========")
indicator_df=df = indicator_df.set_index('startdate')
cadres_df=df = cadres_df.set_index('startdate')
final_df=result = pd.concat([indicator_df, cadres_df], axis=1, sort=False)
log.info(final_df.head())
_db.close_db_con()