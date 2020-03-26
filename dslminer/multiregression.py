import requests
import logging
import json
import psycopg2
import numpy as np
import pandas as pd
from sklearn import linear_model
import db

# configurations
log = logging.getLogger("dataloader")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.INFO)

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
    max_period=''' SELECT  date_part('year',max(startdate))
                FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"='''+str(indictor_id)+''' and "Org unit id"='''+str(orgunit_id)
    min_period = ''' SELECT  date_part('year',min(startdate))
                    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"=''' + str(indictor_id) + ''' and "Org unit id"=''' + str(orgunit_id)

    cursor = _db.get_db_con()[1]
    cursor.execute(max_period)
    row = cursor.fetchall()
    if (len(row) != 0):
        END_YEAR=int(row[0][0])
        log.info("end year "+str(row[0][0]))

    cursor.execute(min_period)
    row = cursor.fetchall()
    if (len(row) != 0):
        if(int(row[0][0]<2009)):
            pass
        else:
            BEGIN_YEAR = int(row[0][0])
        log.info("start year"+ str(row[0][0]))


def get_indicator_data(indicatorid,ouid):
    # BEGIN_YEAR = 2009
    # END_YEAR = 2019

    connection = _db.get_db_con()[0]
    cursor = _db.get_db_con()[1]
    query_string = '''SELECT  distinct startdate, kpivalue
                FROM public.vw_mohdsl_dhis_indicators where "Indicator ID"=%s and "Org unit id"=%s and startdate>='%s' and enddate<='%s' order by startdate asc''' % (
    indicatorid,ouid,str(BEGIN_YEAR)+"-01-01", str(END_YEAR)+"-12-31" )
    log.info(query_string)
    pd_resultset = pd.read_sql_query(query_string, connection)
    log.info(query_string)
    data = pd.DataFrame(pd_resultset)
    log.info(data.head())


def get_cadres_by_year(orgunit,cadreid_list,begin_year,end_year):
    for year in range(begin_year, end_year):
        for org in orgunit:
            for cadre in cadreid_list:
                log.info(org)
                log.info(cadreid_list)
                log.info(BASE_URL + 'cadres?pe=' + str(year) + '&ouid=' + orgunit + '&id=' + cadre['id'] + '&periodtype=monthly')


set_max_min_period(61901,23519)
get_indicator_data(61901,23519)
_db.close_db_con()