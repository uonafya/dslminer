import logging
import matplotlib.pyplot as plt
import pandas as pd
from . import db
from datetime import datetime


# configurations
log = logging.getLogger("indicator correlation")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',level=logging.DEBUG)


class IndicatorCorrelation:
    def __init__(self):
        self.begin_year=2010
        self.end_year=2019
        self._db = db.database()
        self.variables={}

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


    def get_indicator_data(self,ouid,compare_indicators):
        query_string = '''SELECT  distinct startdate ,ROUND(kpivalue,3) , "Indicator name" as indicator , "Organisation Unit Name" as org_unit , "Org unit id" as org_id , 
                    "Indicator ID" as indicator_id ,  _datecreated , lastupdated , "Indicator description"  
                    FROM public.vw_mohdsl_dhis_indicators where "Indicator ID" in (%s) and "Org unit id"=%s and startdate>='%s' and enddate<='%s' order by startdate asc''' % (
        compare_indicators,ouid,str(self.begin_year)+"-01-01", str(self.end_year)+"-12-31" )
        print(query_string)
        columns = ["date", "indicator"]
        cursor = self._db.get_db_con()[1]
        cursor.execute(query_string)
        rows = cursor.fetchall()

        indicators_dt = {} # indicator_id: [[date, value],[date, value]]
        indic_meta_list = []
        org_meta_list = []
        indic_payload_values = {}
        added_org_units = []
        for row in rows:

            if row[2] not in self.variables: # indicator name
                self.variables[str(row[5])]=(row[2])
                indicator_meta = {}
                indicator_meta['name'] = row[2]
                indicator_meta['id'] = str(row[5])
                indicator_meta['date_created'] = str(row[6])
                indicator_meta['last_updated'] = str(row[7])
                indicator_meta['description'] = row[8]
                indicator_meta['source'] = "KHIS"
                indic_meta_list.append(indicator_meta)

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

# r=IndicatorCorrelation()
# r.run_correlation(23185,23408,'23191,23701,31589')