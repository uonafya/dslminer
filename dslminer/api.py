from flask import Flask
from flask import request
from . import timeseries
from . import weather_correlation
import json
app = Flask(__name__)

# forecast indicator data
@app.route('/forecast/<indicatorid>/',strict_slashes=False)
def do_indicator_timeseries(indicatorid):
    periodspan = request.args.get('periodspan') # the number of dependent values to be generated based on periodtype (give prediction data to which time)
    periodtype = request.args.get('periodtype') # can yealy 'Y' or montly 'M'
    ouid = request.args.get('ouid')  # orgunit id
    p=timeseries.predictor()
    app.logger.debug("===query parameters===")
    app.logger.debug(periodspan)
    app.logger.debug(periodtype)
    app.logger.debug(ouid)
    app.logger.debug("======")
    if(ouid==None):
        ouid=18 # National org unit id
    if(periodspan!=None):
        periodspan=int(periodspan)
    data=p.predict(indicatorid,ouid,periodtype,periodspan)

    data=json.dumps(data)
    return data

# weather correlation with indicator
@app.route('/correlation/<indicatorid>/<orgunit>/',strict_slashes=False)
def do_weather_correlation(indicatorid,orgunit):
    #23185,23408
    app.logger.info(indicatorid)
    app.logger.info(orgunit)
    p=weather_correlation.WeatherCorrelation()
    app.logger.debug("===weather correlation model started===")
    data=p.run_correlation(indicatorid,orgunit)

    data=json.dumps(data)
    return data